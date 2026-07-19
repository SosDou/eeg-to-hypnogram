#include "eeg_to_hypnogram/dataset_builder.h"

#include "eeg_to_hypnogram/edf_reader.h"
#include "eeg_to_hypnogram/epoch.h"
#include "eeg_to_hypnogram/feature_extraction.h"
#include "eeg_to_hypnogram/sleep_stage.h"
#include "eeg_to_hypnogram/temporal_context.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

    constexpr double kEpochSeconds = 30.0;
    constexpr double kTargetSampleRateHz = 100.0;
    constexpr int kTargetSamplesPerEpoch = 3000;
    constexpr int kMaximumChannelCount = 7;
    constexpr double kSpectrumMinimumHz = 0.5;
    constexpr double kSpectrumMaximumHz = 30.0;
    constexpr std::int64_t kTicksPerSecond = 10000000;

    std::vector<double> ResampleLinearImpl(
        const std::vector<double> &samples,
        double sourceSampleRateHz,
        double targetSampleRateHz,
        int targetSampleCount)
    {
        if (!std::isfinite(sourceSampleRateHz) ||
            !std::isfinite(targetSampleRateHz) ||
            sourceSampleRateHz <= 0.0 ||
            targetSampleRateHz <= 0.0 ||
            targetSampleCount <= 0)
        {
            throw std::invalid_argument("Invalid resample parameters.");
        }

        if (samples.empty())
        {
            return std::vector<double>(
                static_cast<std::size_t>(targetSampleCount),
                0.0);
        }

        std::vector<double> output(
            static_cast<std::size_t>(targetSampleCount),
            samples.back());

        for (int targetIndex = 0;
             targetIndex < targetSampleCount;
             ++targetIndex)
        {
            const double targetTimeSeconds =
                static_cast<double>(targetIndex) /
                targetSampleRateHz;

            const double sourceIndex =
                targetTimeSeconds * sourceSampleRateHz;

            const std::size_t leftIndex =
                static_cast<std::size_t>(
                    std::floor(sourceIndex));

            if (leftIndex >= samples.size() - 1)
            {
                // 与旧实现一致：映射到右边界以外时复制最后一个样本。
                output[static_cast<std::size_t>(targetIndex)] =
                    samples.back();
                continue;
            }

            const std::size_t rightIndex = leftIndex + 1;
            const double fraction =
                sourceIndex - static_cast<double>(leftIndex);

            output[static_cast<std::size_t>(targetIndex)] =
                samples[leftIndex] * (1.0 - fraction) +
                samples[rightIndex] * fraction;
        }

        return output;
    }

    int StageToLabelIdImpl(const std::string &stage)
    {
        eeg_to_hypnogram::SleepStage parsedStage =
            eeg_to_hypnogram::SleepStage::Unknown;

        if (!eeg_to_hypnogram::TryParseSleepStage(
                stage,
                parsedStage))
        {
            throw std::invalid_argument(
                "Unsupported stage label: " + stage);
        }

        return eeg_to_hypnogram::SleepStageToClassLabel(
            parsedStage);
    }

    std::vector<eeg_to_hypnogram::DatasetFilePair> ConvertManifestPairs(
        const eeg_to_hypnogram::DatasetManifest &manifest)
    {
        std::vector<eeg_to_hypnogram::DatasetFilePair> pairs;
        pairs.reserve(manifest.pairs.size());

        for (const auto &pair : manifest.pairs)
        {
            pairs.push_back(
                {
                    pair.psgPath,
                    pair.hypnogramPath,
                });
        }

        return pairs;
    }

    std::vector<eeg_to_hypnogram::SleepEpoch> BuildFixedEpochs(
        double recordingDurationSeconds,
        double epochSeconds)
    {
        if (!std::isfinite(recordingDurationSeconds) ||
            !std::isfinite(epochSeconds) ||
            recordingDurationSeconds <= 0.0 ||
            epochSeconds <= 0.0)
        {
            return {};
        }

        const double epochCountValue =
            std::floor(recordingDurationSeconds / epochSeconds);

        if (epochCountValue <= 0.0)
        {
            return {};
        }

        if (epochCountValue >
            static_cast<double>(
                std::numeric_limits<std::int64_t>::max()))
        {
            throw std::runtime_error(
                "Unlabeled epoch count exceeds the supported range.");
        }

        const std::int64_t totalEpochs =
            static_cast<std::int64_t>(epochCountValue);

        std::vector<eeg_to_hypnogram::SleepEpoch> epochs;
        epochs.reserve(static_cast<std::size_t>(totalEpochs));

        for (std::int64_t epochIndex = 0;
             epochIndex < totalEpochs;
             ++epochIndex)
        {
            eeg_to_hypnogram::SleepEpoch epoch;
            epoch.index = epochIndex;
            epoch.startSeconds =
                static_cast<double>(epochIndex) * epochSeconds;
            epoch.durationSeconds = epochSeconds;
            epoch.startTicks =
                static_cast<std::int64_t>(
                    std::llround(
                        epoch.startSeconds *
                        static_cast<double>(kTicksPerSecond)));
            epoch.durationTicks =
                static_cast<std::int64_t>(
                    std::llround(
                        epochSeconds *
                        static_cast<double>(kTicksPerSecond)));
            epoch.stage = "W";
            epoch.rawText = "UNLABELED";
            epochs.push_back(std::move(epoch));
        }

        return epochs;
    }

    std::size_t ValidateFeatureMatrix(
        const std::vector<std::vector<double>> &features,
        std::optional<std::size_t> expectedRowCount,
        const std::string &where)
    {
        if (expectedRowCount.has_value() &&
            features.size() != *expectedRowCount)
        {
            throw std::runtime_error(
                "Feature row count mismatch at " + where + ".");
        }

        if (features.empty())
        {
            return 0;
        }

        const std::size_t featureDimension =
            features.front().size();

        for (const auto &row : features)
        {
            if (row.size() != featureDimension)
            {
                throw std::runtime_error(
                    "Inconsistent feature dimension at " + where + ".");
            }
        }

        return featureDimension;
    }

    void EnsureSummaryCompatible(
        const eeg_to_hypnogram::FeaturePipelineSummary &expected,
        const eeg_to_hypnogram::FeaturePipelineSummary &current,
        const std::string &where)
    {
        if (expected.epochSeconds != current.epochSeconds ||
            expected.targetSampleRateHz !=
                current.targetSampleRateHz ||
            expected.channelCount != current.channelCount ||
            expected.baseFeatureDim != current.baseFeatureDim ||
            expected.temporalFeatureDim !=
                current.temporalFeatureDim ||
            expected.channelLabels != current.channelLabels)
        {
            throw std::runtime_error(
                "Feature pipeline mismatch at " + where +
                ". Check channels, epoch length, and feature dimensions.");
        }
    }

    void BuildBaseDataset(
        const eeg_to_hypnogram::EdfReader &psgReader,
        const std::vector<eeg_to_hypnogram::SleepEpoch> &epochs,
        bool produceLabels,
        std::vector<std::vector<double>> *featuresOut,
        std::vector<int> *labelsOut,
        eeg_to_hypnogram::FeaturePipelineSummary *summaryOut)
    {
        if (featuresOut == nullptr)
        {
            throw std::invalid_argument(
                "featuresOut must not be null.");
        }

        if (produceLabels && labelsOut == nullptr)
        {
            throw std::invalid_argument(
                "labelsOut must not be null when labels are requested.");
        }

        const auto &header = psgReader.Header();

        const int channelCount =
            std::min(
                kMaximumChannelCount,
                static_cast<int>(header.signals.size()));

        if (channelCount <= 0)
        {
            throw std::runtime_error(
                "No PSG channels are available.");
        }

        eeg_to_hypnogram::EpochBatchData epochBatch;
        epochBatch.reserve(epochs.size());

        std::vector<double> resampledChannelRates(
            static_cast<std::size_t>(channelCount),
            kTargetSampleRateHz);

        std::vector<int> labels;
        if (produceLabels)
        {
            labels.reserve(epochs.size());
        }

        for (const auto &epoch : epochs)
        {
            std::vector<std::vector<double>> epochChannels;
            epochChannels.reserve(
                static_cast<std::size_t>(channelCount));

            for (int channelIndex = 0;
                 channelIndex < channelCount;
                 ++channelIndex)
            {
                const auto &signal =
                    header.signals[static_cast<std::size_t>(channelIndex)];

                const double sourceSampleRateHz =
                    signal.sampleRateHz;

                if (!std::isfinite(sourceSampleRateHz) ||
                    sourceSampleRateHz <= 0.0)
                {
                    throw std::runtime_error(
                        "Invalid channel sample rate while building features.");
                }

                const double sourceSampleCountValue =
                    std::round(
                        sourceSampleRateHz * kEpochSeconds);

                if (sourceSampleCountValue <= 1.0 ||
                    sourceSampleCountValue >
                        static_cast<double>(
                            std::numeric_limits<int>::max()))
                {
                    throw std::runtime_error(
                        "Invalid source sample count while building features.");
                }

                const int sourceSamplesPerEpoch =
                    static_cast<int>(sourceSampleCountValue);

                if (!std::isfinite(epoch.startSeconds) ||
                    epoch.startSeconds < 0.0)
                {
                    throw std::runtime_error(
                        "Invalid epoch start time while building features.");
                }

                const std::int64_t startSample =
                    static_cast<std::int64_t>(
                        std::llround(
                            epoch.startSeconds *
                            sourceSampleRateHz));

                const auto rawEpoch =
                    psgReader.ReadPhysicalSamples(
                        channelIndex,
                        startSample,
                        sourceSamplesPerEpoch);

                epochChannels.push_back(
                    ResampleLinearImpl(
                        rawEpoch,
                        sourceSampleRateHz,
                        kTargetSampleRateHz,
                        kTargetSamplesPerEpoch));
            }

            epochBatch.push_back(std::move(epochChannels));

            if (produceLabels)
            {
                labels.push_back(
                    StageToLabelIdImpl(epoch.stage));
            }
        }

        eeg_to_hypnogram::PsdConfig psdConfig;
        psdConfig.method = eeg_to_hypnogram::PsdMethod::Welch;
        psdConfig.welch.nFft = 128;
        psdConfig.welch.nOverlap = 64;

        auto features =
            eeg_to_hypnogram::ComputeEegPowerBandFeatures(
                epochBatch,
                resampledChannelRates,
                eeg_to_hypnogram::DefaultEegBands(),
                kSpectrumMinimumHz,
                kSpectrumMaximumHz,
                psdConfig);

        const std::size_t baseFeatureDimension =
            ValidateFeatureMatrix(
                features,
                epochs.size(),
                "base feature extraction");

        if (produceLabels &&
            features.size() != labels.size())
        {
            throw std::runtime_error(
                "Base feature and label row counts do not match.");
        }

        if (summaryOut != nullptr)
        {
            summaryOut->epochSeconds = kEpochSeconds;
            summaryOut->targetSampleRateHz =
                kTargetSampleRateHz;
            summaryOut->channelCount = channelCount;
            summaryOut->baseFeatureDim =
                static_cast<int>(baseFeatureDimension);
            summaryOut->temporalFeatureDim = 0;
            summaryOut->channelLabels.clear();
            summaryOut->channelSampleRatesHz.clear();
            summaryOut->channelLabels.reserve(
                static_cast<std::size_t>(channelCount));
            summaryOut->channelSampleRatesHz.reserve(
                static_cast<std::size_t>(channelCount));

            for (int channelIndex = 0;
                 channelIndex < channelCount;
                 ++channelIndex)
            {
                const auto &signal =
                    header.signals[static_cast<std::size_t>(channelIndex)];

                summaryOut->channelLabels.push_back(signal.label);
                summaryOut->channelSampleRatesHz.push_back(
                    signal.sampleRateHz);
            }
        }

        *featuresOut = std::move(features);

        if (produceLabels)
        {
            *labelsOut = std::move(labels);
        }
    }

} // namespace

namespace eeg_to_hypnogram
{

    void BuildFeaturesFromPsgFile(
        const std::string &psgPath,
        const TemporalContextConfig &contextConfig,
        std::vector<std::vector<double>> *XOut,
        FeaturePipelineSummary *summaryOut)
    {
        if (XOut == nullptr)
        {
            throw std::invalid_argument(
                "XOut must not be null.");
        }

        EdfReader psgReader;
        psgReader.Open(psgPath, true);

        const auto epochs =
            BuildFixedEpochs(
                psgReader.Header().fileDurationSeconds,
                kEpochSeconds);

        if (epochs.empty())
        {
            throw std::runtime_error(
                "No valid unlabeled epochs were generated from PSG file: " +
                psgPath);
        }

        std::vector<std::vector<double>> baseFeatures;
        FeaturePipelineSummary partSummary;

        BuildBaseDataset(
            psgReader,
            epochs,
            false,
            &baseFeatures,
            nullptr,
            &partSummary);

        auto contextualFeatures =
            BuildTemporalContextFeatures(
                baseFeatures,
                contextConfig);

        const std::size_t temporalFeatureDimension =
            ValidateFeatureMatrix(
                contextualFeatures,
                baseFeatures.size(),
                "unlabeled temporal context");

        if (contextualFeatures.empty())
        {
            throw std::runtime_error(
                "No features were generated after temporal context for PSG file: " +
                psgPath);
        }

        partSummary.temporalFeatureDim =
            static_cast<int>(temporalFeatureDimension);

        // 无标签入口保留旧行为：覆盖，而不是追加。
        *XOut = std::move(contextualFeatures);

        if (summaryOut != nullptr)
        {
            *summaryOut = std::move(partSummary);
        }
    }

    void AppendDatasetFromPairs(
        const std::vector<DatasetFilePair> &pairs,
        const std::string &splitName,
        const TemporalContextConfig &contextConfig,
        std::vector<std::vector<double>> *XOut,
        std::vector<int> *yOut,
        FeaturePipelineSummary *summaryOut)
    {
        if (XOut == nullptr || yOut == nullptr)
        {
            throw std::invalid_argument(
                "Output pointers must not be null.");
        }

        if (XOut->size() != yOut->size())
        {
            throw std::invalid_argument(
                "Existing XOut and yOut row counts must match.");
        }

        (void)ValidateFeatureMatrix(
            *XOut,
            XOut->size(),
            "existing output dataset");

        std::optional<FeaturePipelineSummary> expectedSummary;

        if (summaryOut != nullptr &&
            summaryOut->channelCount != 0)
        {
            expectedSummary = *summaryOut;
        }

        for (std::size_t pairIndex = 0;
             pairIndex < pairs.size();
             ++pairIndex)
        {
            const auto &pair = pairs[pairIndex];

            EdfReader psgReader;
            EdfReader hypnogramReader;
            psgReader.Open(pair.psgPath, true);
            hypnogramReader.Open(pair.hypPath, true);

            const auto stageAnnotations =
                hypnogramReader.ReadSleepStageAnnotations();

            const auto epochs =
                EpochBuilder::BuildSleepEpochsMneStyle(
                    stageAnnotations,
                    psgReader.Header().fileDurationSeconds,
                    EpochBuildConfig());

            if (epochs.empty())
            {
                throw std::runtime_error(
                    "No valid sleep epochs were generated for " +
                    splitName +
                    " pair #" +
                    std::to_string(pairIndex + 1) +
                    ".");
            }

            std::vector<std::vector<double>> baseFeatures;
            std::vector<int> labels;
            FeaturePipelineSummary partSummary;

            BuildBaseDataset(
                psgReader,
                epochs,
                true,
                &baseFeatures,
                &labels,
                &partSummary);

            if (baseFeatures.size() != labels.size())
            {
                throw std::runtime_error(
                    "Base feature and label row counts do not match in " +
                    splitName + " split.");
            }

            auto contextualFeatures =
                BuildTemporalContextFeatures(
                    baseFeatures,
                    contextConfig);

            const std::size_t temporalFeatureDimension =
                ValidateFeatureMatrix(
                    contextualFeatures,
                    labels.size(),
                    splitName + " temporal context");

            if (contextualFeatures.empty())
            {
                throw std::runtime_error(
                    "No features were generated after temporal context in " +
                    splitName + " split.");
            }

            if (contextualFeatures.size() != labels.size())
            {
                throw std::runtime_error(
                    "Temporal feature and label row counts do not match in " +
                    splitName + " split.");
            }

            partSummary.temporalFeatureDim =
                static_cast<int>(temporalFeatureDimension);

            if (!XOut->empty())
            {
                const std::size_t outputFeatureDimension =
                    ValidateFeatureMatrix(
                        *XOut,
                        XOut->size(),
                        "existing output dataset");

                if (outputFeatureDimension != temporalFeatureDimension)
                {
                    throw std::runtime_error(
                        "Feature dimension mismatch with existing output in " +
                        splitName + " split.");
                }
            }

            const std::string summaryLocation =
                splitName + " pair #" +
                std::to_string(pairIndex + 1);

            if (expectedSummary.has_value())
            {
                EnsureSummaryCompatible(
                    *expectedSummary,
                    partSummary,
                    summaryLocation);
            }
            else
            {
                expectedSummary = partSummary;
            }

            if (summaryOut != nullptr &&
                summaryOut->channelCount == 0)
            {
                *summaryOut = partSummary;
            }

            XOut->insert(
                XOut->end(),
                std::make_move_iterator(
                    contextualFeatures.begin()),
                std::make_move_iterator(
                    contextualFeatures.end()));

            yOut->insert(
                yOut->end(),
                std::make_move_iterator(labels.begin()),
                std::make_move_iterator(labels.end()));

            std::cout
                << "Loaded " << splitName
                << " pair #" << (pairIndex + 1)
                << ": X += " << baseFeatures.size()
                << " epochs, y += " << labels.size()
                << '\n';
        }
    }

    void AppendDatasetFromManifest(
        const DatasetManifest &manifest,
        const std::string &splitName,
        const TemporalContextConfig &contextConfig,
        std::vector<std::vector<double>> *XOut,
        std::vector<int> *yOut,
        FeaturePipelineSummary *summaryOut)
    {
        eeg_to_hypnogram::ValidateManifestForDatasetAssembly(manifest);

        AppendDatasetFromPairs(
            ConvertManifestPairs(manifest),
            splitName,
            contextConfig,
            XOut,
            yOut,
            summaryOut);
    }

#if defined(EEG_TO_HYPNOGRAM_DATASET_BUILDER_TESTING)
    namespace dataset_builder_testing
    {

        int StageToLabelId(const std::string &stage)
        {
            return StageToLabelIdImpl(stage);
        }

        std::vector<double> ResampleLinear(
            const std::vector<double> &samples,
            double sourceSampleRateHz,
            double targetSampleRateHz,
            int targetSampleCount)
        {
            return ResampleLinearImpl(
                samples,
                sourceSampleRateHz,
                targetSampleRateHz,
                targetSampleCount);
        }

    } // namespace dataset_builder_testing
#endif

} // namespace eeg_to_hypnogram
