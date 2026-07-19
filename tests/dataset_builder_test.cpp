#ifndef EEG_TO_HYPNOGRAM_DATASET_BUILDER_TESTING
#define EEG_TO_HYPNOGRAM_DATASET_BUILDER_TESTING 1
#endif

#include "eeg_to_hypnogram/dataset_builder.h"

#include "eeg_to_hypnogram/dataset_manifest.h"
#include "eeg_to_hypnogram/edf_reader.h"
#include "eeg_to_hypnogram/feature_extraction.h"
#include "eeg_to_hypnogram/dataset_builder.h"

#include "eeg_to_hypnogram/edf_reader.h"
#include "eeg_to_hypnogram/feature_extraction.h"

#include <edflib.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

    constexpr double kPi = 3.14159265358979323846;
    constexpr double kSampleRateHz = 100.0;
    constexpr int kSecondsPerEpoch = 30;
    constexpr int kDefaultChannelCount = 7;

    void Require(bool condition, const std::string &message)
    {
        if (!condition)
        {
            throw std::runtime_error(message);
        }
    }

    template <typename ExceptionType, typename Callable>
    void RequireThrows(Callable &&callable, const std::string &message)
    {
        bool threwExpected = false;

        try
        {
            callable();
        }
        catch (const ExceptionType &)
        {
            threwExpected = true;
        }

        Require(threwExpected, message);
    }

    bool NearlyEqual(
        double lhs,
        double rhs,
        double absoluteTolerance = 1e-11,
        double relativeTolerance = 1e-8)
    {
        const double difference = std::abs(lhs - rhs);
        if (difference <= absoluteTolerance)
        {
            return true;
        }

        return difference <=
               relativeTolerance *
                   std::max(std::abs(lhs), std::abs(rhs));
    }

    void RequireRowsNearlyEqual(
        const std::vector<double> &lhs,
        const std::vector<double> &rhs,
        const std::string &message,
        double absoluteTolerance = 1e-11,
        double relativeTolerance = 1e-8)
    {
        Require(lhs.size() == rhs.size(), message + " (size mismatch)");

        for (std::size_t index = 0; index < lhs.size(); ++index)
        {
            if (!NearlyEqual(
                    lhs[index],
                    rhs[index],
                    absoluteTolerance,
                    relativeTolerance))
            {
                throw std::runtime_error(
                    message + " at index " + std::to_string(index) +
                    ": lhs=" + std::to_string(lhs[index]) +
                    ", rhs=" + std::to_string(rhs[index]));
            }
        }
    }

    void RequireFeatureMatricesEqual(
        const std::vector<std::vector<double>> &lhs,
        const std::vector<std::vector<double>> &rhs,
        const std::string &message)
    {
        Require(lhs.size() == rhs.size(), message + " (row count mismatch)");

        for (std::size_t rowIndex = 0;
             rowIndex < lhs.size();
             ++rowIndex)
        {
            Require(
                lhs[rowIndex] == rhs[rowIndex],
                message + " (row " + std::to_string(rowIndex) + " changed)");
        }
    }

    void RequireSummaryEqual(
        const eeg_to_hypnogram::FeaturePipelineSummary &lhs,
        const eeg_to_hypnogram::FeaturePipelineSummary &rhs,
        const std::string &message)
    {
        Require(lhs.epochSeconds == rhs.epochSeconds, message + " (epochSeconds)");
        Require(lhs.targetSampleRateHz == rhs.targetSampleRateHz, message + " (targetSampleRateHz)");
        Require(lhs.channelCount == rhs.channelCount, message + " (channelCount)");
        Require(lhs.baseFeatureDim == rhs.baseFeatureDim, message + " (baseFeatureDim)");
        Require(lhs.temporalFeatureDim == rhs.temporalFeatureDim, message + " (temporalFeatureDim)");
        Require(lhs.channelLabels == rhs.channelLabels, message + " (channelLabels)");
        Require(lhs.channelSampleRatesHz == rhs.channelSampleRatesHz, message + " (channelSampleRatesHz)");
    }

    class TemporaryDirectory final
    {
    public:
        TemporaryDirectory()
        {
            const auto suffix =
                std::chrono::high_resolution_clock::now()
                    .time_since_epoch()
                    .count();

            path_ =
                std::filesystem::temp_directory_path() /
                ("eeg_dataset_builder_test_" +
                 std::to_string(suffix));

            std::filesystem::create_directories(path_);
        }

        ~TemporaryDirectory()
        {
            std::error_code error;
            std::filesystem::remove_all(path_, error);
        }

        const std::filesystem::path &Path() const noexcept
        {
            return path_;
        }

    private:
        std::filesystem::path path_;
    };

    void RequireEdfSuccess(int result, const std::string &operation)
    {
        if (result != 0)
        {
            throw std::runtime_error(
                operation + " failed with EDFlib code " +
                std::to_string(result) + ".");
        }
    }

    std::string AnnotationTextForStage(const std::string &stage)
    {
        if (stage == "W")
        {
            return "Sleep stage W";
        }
        if (stage == "N1")
        {
            return "Sleep stage 1";
        }
        if (stage == "N2")
        {
            return "Sleep stage 2";
        }
        if (stage == "N3")
        {
            return "Sleep stage 3";
        }
        if (stage == "REM")
        {
            return "Sleep stage R";
        }
        if (stage == "UNKNOWN")
        {
            return "Sleep stage ?";
        }
        if (stage == "MOVEMENT")
        {
            return "Movement time";
        }

        return stage;
    }

    double FrequencyForStage(const std::string &stage)
    {
        if (stage == "W")
        {
            return 2.0;
        }
        if (stage == "N1")
        {
            return 6.0;
        }
        if (stage == "N2")
        {
            return 10.0;
        }
        if (stage == "N3")
        {
            return 13.0;
        }
        if (stage == "REM")
        {
            return 20.0;
        }

        return 3.0;
    }

    using SampleGenerator =
        std::function<double(double, int, const std::string &)>;

    void ConfigureSignal(
        int handle,
        int signalIndex,
        int sampleRate,
        const std::string &label)
    {
        RequireEdfSuccess(
            edf_set_samplefrequency(
                handle,
                signalIndex,
                sampleRate),
            "edf_set_samplefrequency");

        RequireEdfSuccess(
            edf_set_physical_minimum(
                handle,
                signalIndex,
                -20.0),
            "edf_set_physical_minimum");

        RequireEdfSuccess(
            edf_set_physical_maximum(
                handle,
                signalIndex,
                20.0),
            "edf_set_physical_maximum");

        RequireEdfSuccess(
            edf_set_digital_minimum(
                handle,
                signalIndex,
                -32768),
            "edf_set_digital_minimum");

        RequireEdfSuccess(
            edf_set_digital_maximum(
                handle,
                signalIndex,
                32767),
            "edf_set_digital_maximum");

        RequireEdfSuccess(
            edf_set_label(
                handle,
                signalIndex,
                label.c_str()),
            "edf_set_label");

        RequireEdfSuccess(
            edf_set_physical_dimension(
                handle,
                signalIndex,
                "uV"),
            "edf_set_physical_dimension");
    }

    void WritePsg(
        const std::filesystem::path &path,
        const std::vector<std::string> &stages,
        int channelCount,
        const std::vector<std::string> &channelLabels,
        const SampleGenerator &sampleGenerator)
    {
        Require(channelCount > 0, "Synthetic PSG requires channels.");
        Require(
            channelLabels.size() ==
                static_cast<std::size_t>(channelCount),
            "Synthetic PSG labels must match channel count.");
        Require(!stages.empty(), "Synthetic PSG requires epochs.");

        const int handle =
            edfopen_file_writeonly(
                path.string().c_str(),
                EDFLIB_FILETYPE_EDFPLUS,
                channelCount);

        if (handle < 0)
        {
            throw std::runtime_error(
                "Failed to create synthetic PSG: " + path.string());
        }

        try
        {
            for (int channelIndex = 0;
                 channelIndex < channelCount;
                 ++channelIndex)
            {
                ConfigureSignal(
                    handle,
                    channelIndex,
                    static_cast<int>(kSampleRateHz),
                    channelLabels[static_cast<std::size_t>(channelIndex)]);
            }

            const int totalSeconds =
                static_cast<int>(stages.size()) * kSecondsPerEpoch;

            for (int second = 0; second < totalSeconds; ++second)
            {
                const std::size_t epochIndex =
                    static_cast<std::size_t>(second / kSecondsPerEpoch);

                for (int channelIndex = 0;
                     channelIndex < channelCount;
                     ++channelIndex)
                {
                    std::vector<double> record(
                        static_cast<std::size_t>(kSampleRateHz),
                        0.0);

                    for (int sampleIndex = 0;
                         sampleIndex < static_cast<int>(kSampleRateHz);
                         ++sampleIndex)
                    {
                        const double timeSeconds =
                            static_cast<double>(second) +
                            static_cast<double>(sampleIndex) /
                                kSampleRateHz;

                        record[static_cast<std::size_t>(sampleIndex)] =
                            sampleGenerator(
                                timeSeconds,
                                channelIndex,
                                stages[epochIndex]);
                    }

                    RequireEdfSuccess(
                        edfwrite_physical_samples(
                            handle,
                            record.data()),
                        "edfwrite_physical_samples");
                }
            }
        }
        catch (...)
        {
            edfclose_file(handle);
            throw;
        }

        RequireEdfSuccess(
            edfclose_file(handle),
            "edfclose_file PSG");
    }

    void WriteHypnogram(
        const std::filesystem::path &path,
        const std::vector<std::string> &stages)
    {
        const int handle =
            edfopen_file_writeonly(
                path.string().c_str(),
                EDFLIB_FILETYPE_EDFPLUS,
                1);

        if (handle < 0)
        {
            throw std::runtime_error(
                "Failed to create synthetic Hypnogram: " +
                path.string());
        }

        try
        {
            ConfigureSignal(
                handle,
                0,
                1,
                "Hypnogram");

            const int totalSeconds =
                static_cast<int>(stages.size()) *
                kSecondsPerEpoch;

            std::vector<double> record(1, 0.0);

            for (int second = 0;
                 second < totalSeconds;
                 ++second)
            {
                RequireEdfSuccess(
                    edfwrite_physical_samples(
                        handle,
                        record.data()),
                    "edfwrite_physical_samples hypnogram");
            }

            constexpr long long kMicrosecondsPerSecond =
                1000000LL;

            for (std::size_t epochIndex = 0;
                 epochIndex < stages.size();
                 ++epochIndex)
            {
                const long long onset =
                    static_cast<long long>(epochIndex) *
                    static_cast<long long>(kSecondsPerEpoch) *
                    kMicrosecondsPerSecond;

                const long long duration =
                    static_cast<long long>(kSecondsPerEpoch) *
                    kMicrosecondsPerSecond;

                const std::string text =
                    AnnotationTextForStage(
                        stages[epochIndex]);

                RequireEdfSuccess(
                    edfwrite_annotation_utf8_hr(
                        handle,
                        onset,
                        duration,
                        text.c_str()),
                    "edfwrite_annotation_utf8_hr");
            }
        }
        catch (...)
        {
            edfclose_file(handle);
            throw;
        }

        RequireEdfSuccess(
            edfclose_file(handle),
            "edfclose_file Hypnogram");
    }

    struct SyntheticPair
    {
        std::string psgPath;
        std::string hypnogramPath;
    };

    eeg_to_hypnogram::DatasetManifest MakeManifestFromSyntheticPairs(
        const std::vector<SyntheticPair> &pairs)
    {
        eeg_to_hypnogram::DatasetManifest manifest;
        manifest.pairs.reserve(pairs.size());

        for (std::size_t index = 0;
             index < pairs.size();
             ++index)
        {
            const std::string identifier =
                std::to_string(index + 1);

            manifest.pairs.push_back(
                {
                    "SYN" + identifier,
                    "SYN" + identifier + "1A",
                    "1",
                    pairs[index].psgPath,
                    pairs[index].hypnogramPath,
                });
        }

        return manifest;
    }

    SyntheticPair CreateSyntheticPair(
        const std::filesystem::path &directory,
        const std::string &name,
        const std::vector<std::string> &stages,
        int channelCount = kDefaultChannelCount,
        const std::string &firstChannelLabel = "CH0")
    {
        const auto psgPath = directory / (name + "-PSG.edf");
        const auto hypnogramPath =
            directory / (name + "-Hypnogram.edf");

        std::vector<std::string> labels;
        labels.reserve(static_cast<std::size_t>(channelCount));
        for (int channelIndex = 0;
             channelIndex < channelCount;
             ++channelIndex)
        {
            labels.push_back(
                channelIndex == 0
                    ? firstChannelLabel
                    : "CH" + std::to_string(channelIndex));
        }

        WritePsg(
            psgPath,
            stages,
            channelCount,
            labels,
            [](double timeSeconds,
               int channelIndex,
               const std::string &stage)
            {
                const double amplitude =
                    1.0 + 0.1 * static_cast<double>(channelIndex);
                return amplitude *
                       std::sin(
                           2.0 * kPi *
                           FrequencyForStage(stage) *
                           timeSeconds);
            });

        WriteHypnogram(hypnogramPath, stages);

        return {
            psgPath.string(),
            hypnogramPath.string(),
        };
    }

    std::string CreateGoldenUnlabeledPsg(
        const std::filesystem::path &directory)
    {
        const auto path = directory / "golden-unlabeled-PSG.edf";

        WritePsg(
            path,
            {"W"},
            1,
            {"GOLDEN"},
            [](double timeSeconds,
               int,
               const std::string &)
            {
                return 0.8 * std::sin(2.0 * kPi * 2.3 * timeSeconds) +
                       0.5 * std::sin(2.0 * kPi * 10.2 * timeSeconds) +
                       0.25 * std::sin(2.0 * kPi * 18.7 * timeSeconds);
            });

        return path.string();
    }

    std::vector<double> MakeGoldenWave()
    {
        std::vector<double> samples(3000, 0.0);

        for (std::size_t sampleIndex = 0;
             sampleIndex < samples.size();
             ++sampleIndex)
        {
            const double timeSeconds =
                static_cast<double>(sampleIndex) /
                kSampleRateHz;

            samples[sampleIndex] =
                0.8 * std::sin(2.0 * kPi * 2.3 * timeSeconds) +
                0.5 * std::sin(2.0 * kPi * 10.2 * timeSeconds) +
                0.25 * std::sin(2.0 * kPi * 18.7 * timeSeconds);
        }

        return samples;
    }

    eeg_to_hypnogram::TemporalContextConfig NoContext()
    {
        eeg_to_hypnogram::TemporalContextConfig config;
        config.leftContext = 0;
        config.rightContext = 0;
        return config;
    }

    void TestDefaultSummary()
    {
        const eeg_to_hypnogram::FeaturePipelineSummary summary;

        Require(
            NearlyEqual(summary.epochSeconds, 30.0),
            "Default epochSeconds must be 30.");
        Require(
            NearlyEqual(summary.targetSampleRateHz, 100.0),
            "Default targetSampleRateHz must be 100.");
        Require(summary.channelCount == 0, "Default channel count must be zero.");
        Require(summary.baseFeatureDim == 0, "Default base dimension must be zero.");
        Require(summary.temporalFeatureDim == 0, "Default temporal dimension must be zero.");
        Require(summary.channelLabels.empty(), "Default channel labels must be empty.");
        Require(summary.channelSampleRatesHz.empty(), "Default channel rates must be empty.");
    }

    void TestNullOutputPointers()
    {
        const eeg_to_hypnogram::TemporalContextConfig config;

        RequireThrows<std::invalid_argument>(
            [&]
            {
                eeg_to_hypnogram::BuildFeaturesFromPsgFile(
                    "unused.edf",
                    config,
                    nullptr,
                    nullptr);
            },
            "BuildFeaturesFromPsgFile must reject a null XOut.");

        std::vector<std::vector<double>> X;
        std::vector<int> y;

        RequireThrows<std::invalid_argument>(
            [&]
            {
                eeg_to_hypnogram::AppendDatasetFromPairs(
                    {},
                    "empty",
                    config,
                    nullptr,
                    &y,
                    nullptr);
            },
            "AppendDatasetFromPairs must reject a null XOut.");

        RequireThrows<std::invalid_argument>(
            [&]
            {
                eeg_to_hypnogram::AppendDatasetFromPairs(
                    {},
                    "empty",
                    config,
                    &X,
                    nullptr,
                    nullptr);
            },
            "AppendDatasetFromPairs must reject a null yOut.");
    }

    void TestEmptyPairs()
    {
        std::vector<std::vector<double>> X = {{1.0, 2.0}};
        std::vector<int> y = {9};
        eeg_to_hypnogram::FeaturePipelineSummary summary;

        eeg_to_hypnogram::AppendDatasetFromPairs(
            {},
            "empty",
            NoContext(),
            &X,
            &y,
            &summary);

        Require(X.size() == 1, "Empty pairs must not change X.");
        Require(y.size() == 1, "Empty pairs must not change y.");
        Require(summary.channelCount == 0, "Empty pairs must not initialize summary.");
    }

    void TestLabelMappingAndResampling()
    {
        using eeg_to_hypnogram::dataset_builder_testing::ResampleLinear;
        using eeg_to_hypnogram::dataset_builder_testing::StageToLabelId;

        Require(StageToLabelId("W") == 0, "W must map to 0.");
        Require(StageToLabelId("N1") == 1, "N1 must map to 1.");
        Require(StageToLabelId("N2") == 2, "N2 must map to 2.");
        Require(StageToLabelId("N3") == 3, "N3 must map to 3.");
        Require(StageToLabelId("REM") == 4, "REM must map to 4.");

        RequireThrows<std::invalid_argument>(
            []
            {
                (void)eeg_to_hypnogram::dataset_builder_testing::StageToLabelId(
                    "UNKNOWN");
            },
            "Unsupported stages must throw invalid_argument.");

        const auto resampled =
            ResampleLinear(
                {0.0, 10.0, 20.0},
                1.0,
                2.0,
                6);

        const std::vector<double> expected =
            {0.0, 5.0, 10.0, 15.0, 20.0, 20.0};

        RequireRowsNearlyEqual(
            resampled,
            expected,
            "Linear interpolation or right-edge replication changed.");

        const auto emptyResampled =
            ResampleLinear({}, 100.0, 100.0, 4);

        RequireRowsNearlyEqual(
            emptyResampled,
            {0.0, 0.0, 0.0, 0.0},
            "Empty input must resample to zeros.");
    }

    void TestSinglePairAndTemporalContext(
        const SyntheticPair &pair)
    {
        const eeg_to_hypnogram::DatasetFilePair filePair =
            {pair.psgPath, pair.hypnogramPath};

        std::vector<std::vector<double>> baseX;
        std::vector<int> baseY;
        eeg_to_hypnogram::FeaturePipelineSummary baseSummary;

        eeg_to_hypnogram::AppendDatasetFromPairs(
            {filePair},
            "synthetic-base",
            NoContext(),
            &baseX,
            &baseY,
            &baseSummary);

        const std::vector<int> expectedLabels = {0, 1, 2, 3, 4};

        Require(baseX.size() == expectedLabels.size(), "Single pair X row count changed.");
        Require(baseY == expectedLabels, "Stage labels must map to 0 through 4.");
        Require(baseX.size() == baseY.size(), "Single pair X/y row counts must match.");
        Require(baseSummary.channelCount == 7, "First seven channels must be selected.");
        Require(baseSummary.baseFeatureDim == 35, "Seven channels and five bands must produce 35 base features.");
        Require(baseSummary.temporalFeatureDim == 35, "No-context feature dimension must equal base dimension.");
        Require(baseSummary.channelLabels.size() == 7, "Summary must record seven channel labels.");
        Require(baseSummary.channelSampleRatesHz.size() == 7, "Summary must record seven original sample rates.");

        eeg_to_hypnogram::TemporalContextConfig defaultContext;
        std::vector<std::vector<double>> contextualX;
        std::vector<int> contextualY;
        eeg_to_hypnogram::FeaturePipelineSummary contextualSummary;

        eeg_to_hypnogram::AppendDatasetFromPairs(
            {filePair},
            "synthetic-context",
            defaultContext,
            &contextualX,
            &contextualY,
            &contextualSummary);

        Require(contextualX.size() == baseX.size(), "Temporal context must preserve row count.");
        Require(contextualY == baseY, "Temporal context must not change labels.");
        Require(contextualSummary.baseFeatureDim == 35, "Base dimension changed under context.");
        Require(contextualSummary.temporalFeatureDim == 175, "Default 2+2 context must produce 5 x 35 features.");

        for (std::size_t rowIndex = 0;
             rowIndex < contextualX.size();
             ++rowIndex)
        {
            Require(contextualX[rowIndex].size() == 175, "Contextual row dimension must be 175.");

            for (int offset = -2; offset <= 2; ++offset)
            {
                const int sourceIndex =
                    std::max(
                        0,
                        std::min(
                            static_cast<int>(baseX.size()) - 1,
                            static_cast<int>(rowIndex) + offset));

                const std::size_t blockIndex =
                    static_cast<std::size_t>(offset + 2);

                const auto blockBegin =
                    contextualX[rowIndex].begin() +
                    static_cast<std::ptrdiff_t>(blockIndex * 35);

                const std::vector<double> block(
                    blockBegin,
                    blockBegin + 35);

                RequireRowsNearlyEqual(
                    block,
                    baseX[static_cast<std::size_t>(sourceIndex)],
                    "Dataset Builder temporal context ordering changed.");
            }
        }
    }

    void TestMultiFileOrderAndNoCrossFileContext(
        const SyntheticPair &first,
        const SyntheticPair &second)
    {
        const eeg_to_hypnogram::TemporalContextConfig context;

        std::vector<std::vector<double>> firstX;
        std::vector<int> firstY;
        eeg_to_hypnogram::AppendDatasetFromPairs(
            {{first.psgPath, first.hypnogramPath}},
            "first-only",
            context,
            &firstX,
            &firstY,
            nullptr);

        std::vector<std::vector<double>> secondX;
        std::vector<int> secondY;
        eeg_to_hypnogram::AppendDatasetFromPairs(
            {{second.psgPath, second.hypnogramPath}},
            "second-only",
            context,
            &secondX,
            &secondY,
            nullptr);

        std::vector<std::vector<double>> combinedX;
        std::vector<int> combinedY;
        eeg_to_hypnogram::AppendDatasetFromPairs(
            {
                {first.psgPath, first.hypnogramPath},
                {second.psgPath, second.hypnogramPath},
            },
            "combined",
            context,
            &combinedX,
            &combinedY,
            nullptr);

        std::vector<int> expectedY = firstY;
        expectedY.insert(expectedY.end(), secondY.begin(), secondY.end());

        Require(combinedY == expectedY, "Multiple files must append in pairs input order.");
        Require(combinedX.size() == firstX.size() + secondX.size(), "Combined X row count changed.");

        for (std::size_t index = 0; index < firstX.size(); ++index)
        {
            RequireRowsNearlyEqual(
                combinedX[index],
                firstX[index],
                "First file contextual features changed when combined.");
        }

        for (std::size_t index = 0; index < secondX.size(); ++index)
        {
            RequireRowsNearlyEqual(
                combinedX[firstX.size() + index],
                secondX[index],
                "Temporal context crossed a PSG file boundary.");
        }
    }

    void TestManifestEntryMatchesPairEntryAndPreservesOrder(
        const SyntheticPair &first,
        const SyntheticPair &second)
    {
        const eeg_to_hypnogram::TemporalContextConfig context = NoContext();

        const std::vector<eeg_to_hypnogram::DatasetFilePair> filePairs = {
            {second.psgPath, second.hypnogramPath},
            {first.psgPath, first.hypnogramPath},
        };

        eeg_to_hypnogram::DatasetManifest manifest =
            MakeManifestFromSyntheticPairs({second, first});

        // Deliberately make the manifest order differ from key sort order.
        // Dataset Builder must consume manifest.pairs as-is.
        manifest.pairs[0].subjectId = "SYN002";
        manifest.pairs[0].recordingId = "SYN0021A";
        manifest.pairs[1].subjectId = "SYN001";
        manifest.pairs[1].recordingId = "SYN0011A";

        std::vector<std::vector<double>> pairX;
        std::vector<int> pairY;
        eeg_to_hypnogram::FeaturePipelineSummary pairSummary;

        eeg_to_hypnogram::AppendDatasetFromPairs(
            filePairs,
            "manifest-reference",
            context,
            &pairX,
            &pairY,
            &pairSummary);

        std::vector<std::vector<double>> manifestX;
        std::vector<int> manifestY;
        eeg_to_hypnogram::FeaturePipelineSummary manifestSummary;

        eeg_to_hypnogram::AppendDatasetFromManifest(
            manifest,
            "manifest-adapter",
            context,
            &manifestX,
            &manifestY,
            &manifestSummary);

        RequireFeatureMatricesEqual(
            manifestX,
            pairX,
            "Manifest adapter must preserve feature rows exactly.");
        Require(
            manifestY == pairY,
            "Manifest adapter must preserve labels and pair order.");
        RequireSummaryEqual(
            manifestSummary,
            pairSummary,
            "Manifest adapter summary changed.");
    }

    void TestManifestValidationAndIgnoredFiles(
        const SyntheticPair &pair)
    {
        const eeg_to_hypnogram::TemporalContextConfig context = NoContext();

        std::vector<std::vector<double>> X;
        std::vector<int> y;

        RequireThrows<std::invalid_argument>(
            [&]
            {
                eeg_to_hypnogram::DatasetManifest emptyManifest;
                eeg_to_hypnogram::AppendDatasetFromManifest(
                    emptyManifest,
                    "empty-manifest",
                    context,
                    &X,
                    &y,
                    nullptr);
            },
            "An empty manifest must be rejected.");

        struct DirtyManifestCase
        {
            std::string name;
            std::function<void(eeg_to_hypnogram::DatasetManifest *)> mutate;
        };

        const std::vector<DirtyManifestCase> dirtyCases = {
            {
                "unmatched PSG",
                [](eeg_to_hypnogram::DatasetManifest *manifest)
                {
                    manifest->unmatchedPsgFiles.push_back("unmatched-PSG.edf");
                },
            },
            {
                "unmatched Hypnogram",
                [](eeg_to_hypnogram::DatasetManifest *manifest)
                {
                    manifest->unmatchedHypnogramFiles.push_back("unmatched-Hypnogram.edf");
                },
            },
            {
                "duplicate PSG key",
                [](eeg_to_hypnogram::DatasetManifest *manifest)
                {
                    manifest->duplicatePsgKeys.push_back("SC4001E");
                },
            },
            {
                "duplicate Hypnogram key",
                [](eeg_to_hypnogram::DatasetManifest *manifest)
                {
                    manifest->duplicateHypnogramKeys.push_back("SC4001E");
                },
            },
            {
                "duplicate input path",
                [](eeg_to_hypnogram::DatasetManifest *manifest)
                {
                    manifest->duplicateInputPaths.push_back("SC4001E0-PSG.edf");
                },
            },
            {
                "unrecognized EDF",
                [](eeg_to_hypnogram::DatasetManifest *manifest)
                {
                    manifest->unrecognizedEdfFiles.push_back("unknown.edf");
                },
            },
        };

        for (const auto &dirtyCase : dirtyCases)
        {
            eeg_to_hypnogram::DatasetManifest manifest =
                MakeManifestFromSyntheticPairs({pair});
            dirtyCase.mutate(&manifest);
            X.clear();
            y.clear();

            RequireThrows<std::invalid_argument>(
                [&]
                {
                    eeg_to_hypnogram::AppendDatasetFromManifest(
                        manifest,
                        "dirty-manifest-" + dirtyCase.name,
                        context,
                        &X,
                        &y,
                        nullptr);
                },
                "Dirty manifest must be rejected: " + dirtyCase.name);
        }

        eeg_to_hypnogram::DatasetManifest ignoredAllowed =
            MakeManifestFromSyntheticPairs({pair});
        ignoredAllowed.ignoredFiles.push_back("README.txt");

        X.clear();
        y.clear();

        eeg_to_hypnogram::AppendDatasetFromManifest(
            ignoredAllowed,
            "ignored-files-allowed",
            context,
            &X,
            &y,
            nullptr);

        Require(!X.empty(), "ignoredFiles must not block dataset assembly.");
        Require(X.size() == y.size(), "ignoredFiles assembly X/y counts changed.");
    }

    void TestAppendToExistingOutput(const SyntheticPair &pair)
    {
        std::vector<std::vector<double>> expectedAddedX;
        std::vector<int> expectedAddedY;

        eeg_to_hypnogram::AppendDatasetFromPairs(
            {{pair.psgPath, pair.hypnogramPath}},
            "append-reference",
            NoContext(),
            &expectedAddedX,
            &expectedAddedY,
            nullptr);

        Require(
            !expectedAddedX.empty(),
            "Append test fixture must produce features.");

        Require(
            expectedAddedX.size() == expectedAddedY.size(),
            "Append reference X/y counts must match.");

        const std::size_t featureDimension =
            expectedAddedX.front().size();

        std::vector<std::vector<double>> X(
            1,
            std::vector<double>(featureDimension, -7.0));

        std::vector<int> y = {-1};

        eeg_to_hypnogram::AppendDatasetFromPairs(
            {{pair.psgPath, pair.hypnogramPath}},
            "append-existing",
            NoContext(),
            &X,
            &y,
            nullptr);

        Require(
            X.size() == expectedAddedX.size() + 1,
            "Existing X prefix plus appended epochs expected.");

        Require(
            y.size() == expectedAddedY.size() + 1,
            "Existing y prefix plus appended labels expected.");

        Require(
            y.front() == -1,
            "Existing y prefix must be preserved.");

        Require(
            std::all_of(
                X.front().begin(),
                X.front().end(),
                [](double value)
                {
                    return value == -7.0;
                }),
            "Existing X prefix must be preserved.");

        for (std::size_t index = 0;
             index < expectedAddedX.size();
             ++index)
        {
            RequireRowsNearlyEqual(
                X[index + 1],
                expectedAddedX[index],
                "Appended feature row changed.");

            Require(
                y[index + 1] == expectedAddedY[index],
                "Appended label changed.");
        }
    }

    void TestSummaryAndDimensionMismatch(
        const SyntheticPair &compatible,
        const SyntheticPair &labelMismatch,
        const SyntheticPair &dimensionMismatch)
    {
        std::vector<std::vector<double>> X;
        std::vector<int> y;

        RequireThrows<std::runtime_error>(
            [&]
            {
                eeg_to_hypnogram::AppendDatasetFromPairs(
                    {
                        {compatible.psgPath, compatible.hypnogramPath},
                        {labelMismatch.psgPath, labelMismatch.hypnogramPath},
                    },
                    "label-mismatch",
                    NoContext(),
                    &X,
                    &y,
                    nullptr);
            },
            "Different channel labels must be rejected across files.");

        X.clear();
        y.clear();

        RequireThrows<std::runtime_error>(
            [&]
            {
                eeg_to_hypnogram::AppendDatasetFromPairs(
                    {
                        {compatible.psgPath, compatible.hypnogramPath},
                        {dimensionMismatch.psgPath, dimensionMismatch.hypnogramPath},
                    },
                    "dimension-mismatch",
                    NoContext(),
                    &X,
                    &y,
                    nullptr);
            },
            "Different channel counts or feature dimensions must be rejected.");
    }

    void TestUnlabeledPsgOverwrite(
        const std::string &psgPath)
    {
        std::vector<std::vector<double>> X = {{999.0}};
        eeg_to_hypnogram::FeaturePipelineSummary summary;

        eeg_to_hypnogram::TemporalContextConfig context;

        eeg_to_hypnogram::BuildFeaturesFromPsgFile(
            psgPath,
            context,
            &X,
            &summary);

        Require(X.size() == 2, "A 60-second unlabeled PSG must produce two epochs.");
        Require(summary.channelCount == 7, "Unlabeled path must select the first seven channels.");
        Require(summary.baseFeatureDim == 35, "Unlabeled base dimension must be 35.");
        Require(summary.temporalFeatureDim == 175, "Unlabeled default temporal dimension must be 175.");
        Require(X.front().size() == 175, "Unlabeled contextual row dimension must be 175.");
        Require(X.front().front() != 999.0, "Unlabeled path must overwrite existing XOut content.");
    }

    void TestWelch12864GoldenAndBuilderOverride(
        const std::string &goldenPsgPath)
    {
        const auto rawGoldenWave = MakeGoldenWave();

        eeg_to_hypnogram::PsdConfig explicitConfig;
        explicitConfig.method = eeg_to_hypnogram::PsdMethod::Welch;
        explicitConfig.welch.nFft = 128;
        explicitConfig.welch.nOverlap = 64;

        const auto explicitFeatures =
            eeg_to_hypnogram::ComputeEegPowerBandFeatures(
                {{rawGoldenWave}},
                {100.0},
                eeg_to_hypnogram::DefaultEegBands(),
                0.5,
                30.0,
                explicitConfig);

        const std::vector<double> golden = {
            0.13437873257174415,
            0.0000002490812659237748,
            0.06561253876794795,
            0.000006788056954596113,
            0.0034537366514939207,
        };

        Require(explicitFeatures.size() == 1, "Expected one explicit Welch feature row.");
        RequireRowsNearlyEqual(
            explicitFeatures.front(),
            golden,
            "Synthetic Welch 128/64 golden values changed.",
            1e-12,
            1e-9);

        std::vector<double> readBackSamples;

        {
            eeg_to_hypnogram::EdfReader reader;
            reader.Open(goldenPsgPath, true);

            readBackSamples =
                reader.ReadPhysicalSamples(0, 0, 3000);

            // 离开作用域时 reader 析构并关闭 EDF，
            // 后续 BuildFeaturesFromPsgFile 才能重新打开同一个文件。
        }

        const auto readBack128 =
            eeg_to_hypnogram::ComputeEegPowerBandFeatures(
                {{readBackSamples}},
                {100.0},
                eeg_to_hypnogram::DefaultEegBands(),
                0.5,
                30.0,
                explicitConfig);

        std::vector<std::vector<double>> builderFeatures;

        eeg_to_hypnogram::BuildFeaturesFromPsgFile(
            goldenPsgPath,
            NoContext(),
            &builderFeatures,
            nullptr);

        Require(builderFeatures.size() == 1, "Golden PSG must produce one builder row.");
        RequireRowsNearlyEqual(
            builderFeatures.front(),
            readBack128.front(),
            "Dataset Builder must explicitly use Welch 128/64.",
            1e-12,
            1e-9);

        const auto defaultFeatures =
            eeg_to_hypnogram::ComputeEegPowerBandFeatures(
                {{readBackSamples}},
                {100.0});

        bool differsFromDefault = false;
        for (std::size_t index = 0;
             index < builderFeatures.front().size();
             ++index)
        {
            if (std::abs(
                    builderFeatures.front()[index] -
                    defaultFeatures.front()[index]) > 1e-5)
            {
                differsFromDefault = true;
                break;
            }
        }

        Require(
            differsFromDefault,
            "The golden waveform must distinguish Dataset Builder 128/64 from Feature Extraction default 256/128.");
    }

    void RunRealSleepEdfIntegrationIfConfigured()
    {
        const char *datasetDirectory =
            std::getenv("EEG_SLEEP_EDF_DATASET_DIR");

        if (datasetDirectory != nullptr &&
            *datasetDirectory != '\0')
        {
            eeg_to_hypnogram::DatasetManifestScanConfig scan;
            scan.recursive = true;

            const eeg_to_hypnogram::DatasetManifest manifest =
                eeg_to_hypnogram::BuildDatasetManifest(
                    datasetDirectory,
                    scan);

            Require(
                manifest.pairs.size() == 153,
                "Real Sleep-EDF manifest pair count must remain 153.");
            Require(
                manifest.unmatchedPsgFiles.empty(),
                "Real Sleep-EDF manifest must not have unmatched PSG files.");
            Require(
                manifest.unmatchedHypnogramFiles.empty(),
                "Real Sleep-EDF manifest must not have unmatched Hypnogram files.");
            Require(
                manifest.duplicatePsgKeys.empty(),
                "Real Sleep-EDF manifest must not have duplicate PSG keys.");
            Require(
                manifest.duplicateHypnogramKeys.empty(),
                "Real Sleep-EDF manifest must not have duplicate Hypnogram keys.");
            Require(
                manifest.duplicateInputPaths.empty(),
                "Real Sleep-EDF manifest must not have duplicate input paths.");
            Require(
                manifest.unrecognizedEdfFiles.empty(),
                "Real Sleep-EDF manifest must not have unrecognized EDF files.");

            const auto chosenPair = std::find_if(
                manifest.pairs.begin(),
                manifest.pairs.end(),
                [](const eeg_to_hypnogram::SleepEdfFilePair &pair)
                {
                    return pair.recordingId == "SC4001E";
                });

            Require(
                chosenPair != manifest.pairs.end(),
                "Real Sleep-EDF manifest must contain SC4001E.");

            eeg_to_hypnogram::DatasetManifest singletonManifest;
            singletonManifest.pairs.push_back(*chosenPair);

            std::vector<std::vector<double>> X;
            std::vector<int> y;
            eeg_to_hypnogram::FeaturePipelineSummary summary;

            eeg_to_hypnogram::AppendDatasetFromManifest(
                singletonManifest,
                "real-sleep-edf",
                eeg_to_hypnogram::TemporalContextConfig(),
                &X,
                &y,
                &summary);

            Require(!X.empty(), "Real Sleep-EDF must produce features.");
            Require(X.size() == y.size(), "Real Sleep-EDF X/y counts must match.");
            Require(summary.channelCount > 0 && summary.channelCount <= 7, "Real channel count must be in [1, 7].");
            Require(summary.baseFeatureDim == summary.channelCount * 5, "Real base dimension must equal channels x five bands.");
            Require(summary.temporalFeatureDim == summary.baseFeatureDim * 5, "Real default temporal dimension must be five base blocks.");

            for (int label : y)
            {
                Require(label >= 0 && label <= 4, "Real labels must be in [0, 4].");
            }

            const std::string psgName =
                std::filesystem::path(chosenPair->psgPath).filename().string();

            if (psgName == "SC4001E0-PSG.edf")
            {
                Require(X.size() == 841, "SC4001E0 must preserve the known 841-epoch result.");
            }

            std::cout
                << "Dataset Builder real integration test passed\n"
                << "epochs=" << X.size() << '\n'
                << "channels=" << summary.channelCount << '\n'
                << "base_feature_dim=" << summary.baseFeatureDim << '\n'
                << "temporal_feature_dim=" << summary.temporalFeatureDim << '\n';
            return;
        }

        const char *psgPath = std::getenv("EEG_TEST_EDF_FILE");
        const char *hypnogramPath =
            std::getenv("EEG_TEST_HYPNOGRAM_FILE");

        if (psgPath == nullptr || hypnogramPath == nullptr ||
            *psgPath == '\0' || *hypnogramPath == '\0')
        {
            std::cout
                << "Real Dataset Builder integration test skipped: "
                << "EEG_SLEEP_EDF_DATASET_DIR or EEG_TEST_EDF_FILE/EEG_TEST_HYPNOGRAM_FILE must be set.\n";
            return;
        }

        eeg_to_hypnogram::DatasetManifest singletonManifest;
        singletonManifest.pairs.push_back(
            {
                "REAL",
                "REAL1A",
                "1",
                psgPath,
                hypnogramPath,
            });

        std::vector<std::vector<double>> X;
        std::vector<int> y;
        eeg_to_hypnogram::FeaturePipelineSummary summary;

        eeg_to_hypnogram::AppendDatasetFromManifest(
            singletonManifest,
            "real-sleep-edf",
            eeg_to_hypnogram::TemporalContextConfig(),
            &X,
            &y,
            &summary);

        Require(!X.empty(), "Real Sleep-EDF must produce features.");
        Require(X.size() == y.size(), "Real Sleep-EDF X/y counts must match.");
        Require(summary.channelCount > 0 && summary.channelCount <= 7, "Real channel count must be in [1, 7].");
        Require(summary.baseFeatureDim == summary.channelCount * 5, "Real base dimension must equal channels x five bands.");
        Require(summary.temporalFeatureDim == summary.baseFeatureDim * 5, "Real default temporal dimension must be five base blocks.");

        for (int label : y)
        {
            Require(label >= 0 && label <= 4, "Real labels must be in [0, 4].");
        }

        const std::string psgName =
            std::filesystem::path(psgPath).filename().string();

        if (psgName == "SC4001E0-PSG.edf")
        {
            Require(X.size() == 841, "SC4001E0 must preserve the known 841-epoch result.");
        }

        std::cout
            << "Dataset Builder real integration test passed\n"
            << "epochs=" << X.size() << '\n'
            << "channels=" << summary.channelCount << '\n'
            << "base_feature_dim=" << summary.baseFeatureDim << '\n'
            << "temporal_feature_dim=" << summary.temporalFeatureDim << '\n';
    }

} // namespace

int main()
{
    try
    {
        TemporaryDirectory temporaryDirectory;

        const auto allStages =
            CreateSyntheticPair(
                temporaryDirectory.Path(),
                "all-stages",
                {"W", "N1", "N2", "N3", "REM"});

        const auto firstPair =
            CreateSyntheticPair(
                temporaryDirectory.Path(),
                "first-pair",
                {"N1", "N2", "N3", "REM", "N2"});

        const auto secondPair =
            CreateSyntheticPair(
                temporaryDirectory.Path(),
                "second-pair",
                {"REM", "N3", "N2", "N1", "REM"});

        const auto labelMismatch =
            CreateSyntheticPair(
                temporaryDirectory.Path(),
                "label-mismatch",
                {"N1", "N2", "N3", "REM", "N2"},
                7,
                "ALT0");

        const auto dimensionMismatch =
            CreateSyntheticPair(
                temporaryDirectory.Path(),
                "dimension-mismatch",
                {"N1", "N2", "N3", "REM", "N2"},
                6);

        const auto unlabeledPair =
            CreateSyntheticPair(
                temporaryDirectory.Path(),
                "unlabeled-source",
                {"N2", "REM"});

        const std::string goldenPsgPath =
            CreateGoldenUnlabeledPsg(
                temporaryDirectory.Path());

        TestDefaultSummary();
        TestNullOutputPointers();
        TestEmptyPairs();
        TestLabelMappingAndResampling();
        TestSinglePairAndTemporalContext(allStages);
        TestMultiFileOrderAndNoCrossFileContext(
            firstPair,
            secondPair);
        TestManifestEntryMatchesPairEntryAndPreservesOrder(
            firstPair,
            secondPair);
        TestManifestValidationAndIgnoredFiles(firstPair);
        TestAppendToExistingOutput(firstPair);
        TestSummaryAndDimensionMismatch(
            firstPair,
            labelMismatch,
            dimensionMismatch);
        TestUnlabeledPsgOverwrite(unlabeledPair.psgPath);
        TestWelch12864GoldenAndBuilderOverride(
            goldenPsgPath);
        RunRealSleepEdfIntegrationIfConfigured();

        std::cout << "All enabled Dataset Builder tests passed.\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr
            << "Dataset Builder test failed: "
            << error.what()
            << '\n';
        return 1;
    }
}
