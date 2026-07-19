#include "eeg_to_hypnogram/training_pipeline.h"

#include "eeg_to_hypnogram/dataset_builder.h"
#include "eeg_to_hypnogram/dataset_manifest.h"
#include "eeg_to_hypnogram/dataset_split.h"
#include "eeg_to_hypnogram/random_forest_baseline.h"

#include <edflib.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

    namespace fs = std::filesystem;

    constexpr double kPi = 3.14159265358979323846;
    constexpr double kSampleRateHz = 100.0;
    constexpr int kSecondsPerEpoch = 30;
    constexpr int kSyntheticChannelCount = 2;
    constexpr int kNumClasses = 5;

    int gPassed = 0;

    void Require(bool condition, const std::string &message)
    {
        if (!condition)
        {
            throw std::runtime_error(message);
        }
    }

    void RequireNear(
        double actual,
        double expected,
        double tolerance,
        const std::string &message)
    {
        if (std::abs(actual - expected) > tolerance)
        {
            throw std::runtime_error(
                message + ": actual=" + std::to_string(actual) +
                ", expected=" + std::to_string(expected));
        }
    }

    template <typename ExceptionType, typename Function>
    void RequireThrows(Function &&function, const std::string &message)
    {
        bool threwExpected = false;
        try
        {
            function();
        }
        catch (const ExceptionType &)
        {
            threwExpected = true;
        }
        Require(threwExpected, message);
    }

    void Run(const std::string &name, const std::function<void()> &test)
    {
        test();
        ++gPassed;
        std::cout << "[PASS] " << name << '\n';
    }

    class TemporaryDirectory final
    {
    public:
        TemporaryDirectory()
        {
            const auto timestamp =
                std::chrono::high_resolution_clock::now()
                    .time_since_epoch()
                    .count();
            path_ =
                fs::temp_directory_path() /
                ("eeg_training_pipeline_test_" +
                 std::to_string(timestamp));
            fs::create_directories(path_);
        }

        ~TemporaryDirectory()
        {
            std::error_code error;
            fs::remove_all(path_, error);
        }

        const fs::path &Path() const
        {
            return path_;
        }

    private:
        fs::path path_;
    };

    void RequireEdfSuccess(
        int result,
        const std::string &operation)
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
        const fs::path &path,
        const std::vector<std::string> &stages)
    {
        const int handle =
            edfopen_file_writeonly(
                path.string().c_str(),
                EDFLIB_FILETYPE_EDFPLUS,
                kSyntheticChannelCount);

        if (handle < 0)
        {
            throw std::runtime_error(
                "Failed to create synthetic PSG: " + path.string());
        }

        try
        {
            for (int channelIndex = 0;
                 channelIndex < kSyntheticChannelCount;
                 ++channelIndex)
            {
                ConfigureSignal(
                    handle,
                    channelIndex,
                    static_cast<int>(kSampleRateHz),
                    "CH" + std::to_string(channelIndex));
            }

            const int totalSeconds =
                static_cast<int>(stages.size()) *
                kSecondsPerEpoch;

            for (int second = 0; second < totalSeconds; ++second)
            {
                const std::size_t epochIndex =
                    static_cast<std::size_t>(
                        second / kSecondsPerEpoch);

                for (int channelIndex = 0;
                     channelIndex < kSyntheticChannelCount;
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
                        const double amplitude =
                            1.0 +
                            0.25 * static_cast<double>(channelIndex);
                        record[static_cast<std::size_t>(sampleIndex)] =
                            amplitude *
                            std::sin(
                                2.0 * kPi *
                                FrequencyForStage(stages[epochIndex]) *
                                timeSeconds);
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
        const fs::path &path,
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
            ConfigureSignal(handle, 0, 1, "Hypnogram");

            const int totalSeconds =
                static_cast<int>(stages.size()) *
                kSecondsPerEpoch;
            std::vector<double> record(1, 0.0);
            for (int second = 0; second < totalSeconds; ++second)
            {
                RequireEdfSuccess(
                    edfwrite_physical_samples(
                        handle,
                        record.data()),
                    "edfwrite_physical_samples hypnogram");
            }

            constexpr long long kMicrosecondsPerSecond = 1000000LL;
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
                    AnnotationTextForStage(stages[epochIndex]);

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

    eeg_to_hypnogram::SleepEdfFilePair CreateSyntheticPair(
        const fs::path &directory,
        int subjectIndex)
    {
        const std::string subject =
            "SC" + std::to_string(400 + subjectIndex);
        const std::string night =
            (subjectIndex % 2 == 0) ? "1" : "2";
        const std::string recording =
            subject + night + "E";

        const fs::path psgPath =
            directory / (recording + "0-PSG.edf");
        const fs::path hypnogramPath =
            directory / (recording + "C-Hypnogram.edf");

        WritePsg(psgPath, {"W", "N1", "N2", "N3", "REM"});
        WriteHypnogram(
            hypnogramPath,
            {"W", "N1", "N2", "N3", "REM"});

        return {
            subject,
            recording,
            night,
            psgPath.string(),
            hypnogramPath.string(),
        };
    }

    eeg_to_hypnogram::SleepEdfFilePair MakeFakePair(
        int subjectIndex)
    {
        const std::string subject =
            "SC" + std::to_string(400 + subjectIndex);
        const std::string recording = subject + "1E";
        return {
            subject,
            recording,
            "1",
            "/missing/" + recording + "0-PSG.edf",
            "/missing/" + recording + "C-Hypnogram.edf",
        };
    }

    eeg_to_hypnogram::DatasetManifest MakeFakeManifest(
        int subjectCount)
    {
        eeg_to_hypnogram::DatasetManifest manifest;
        for (int index = 0; index < subjectCount; ++index)
        {
            manifest.pairs.push_back(MakeFakePair(index));
        }
        return manifest;
    }

    eeg_to_hypnogram::DatasetManifest CreateSyntheticManifest(
        const fs::path &directory)
    {
        std::vector<eeg_to_hypnogram::SleepEdfFilePair> pairs;
        for (int index = 0; index < 6; ++index)
        {
            pairs.push_back(
                CreateSyntheticPair(directory, index));
        }

        eeg_to_hypnogram::DatasetManifest manifest;
        manifest.pairs = {
            pairs[2],
            pairs[0],
            pairs[5],
            pairs[1],
            pairs[4],
            pairs[3],
        };
        manifest.ignoredFiles.push_back(
            (directory / "README.txt").string());
        return manifest;
    }

    eeg_to_hypnogram::TrainingPipelineConfig FastConfig(
        const fs::path &directory,
        const std::string &name)
    {
        eeg_to_hypnogram::TrainingPipelineConfig config;
        config.splitConfig.testFraction = 0.33;
        config.splitConfig.randomSeed = 7;
        config.selectionSplitConfig.testFraction = 0.25;
        config.selectionSplitConfig.randomSeed = 11;
        config.experimentConfig.temporalContext.leftContext = 0;
        config.experimentConfig.temporalContext.rightContext = 0;
        config.experimentConfig.gridSearch.numTreesSet = {5};
        config.experimentConfig.gridSearch.maxDepthSet = {4};
        config.experimentConfig.gridSearch.minSamplesSplitSet = {2};
        config.experimentConfig.gridSearch.maxThresholdCandidates = 8;
        config.experimentConfig.gridSearch.maxFeaturesPerSplit = 2;
        config.experimentConfig.gridSearch.seed = 99;
        config.experimentConfig.gridSearch.rankingTopK = 1;
        config.modelOutputPath =
            (directory / (name + "-model.srf1")).string();
        config.reportOutputPath =
            (directory / (name + "-report.txt")).string();
        return config;
    }

    bool SameConfig(
        const eeg_to_hypnogram::RandomForestConfig &left,
        const eeg_to_hypnogram::RandomForestConfig &right)
    {
        return left.numTrees == right.numTrees &&
               left.maxDepth == right.maxDepth &&
               left.minSamplesSplit == right.minSamplesSplit &&
               left.maxFeaturesPerSplit == right.maxFeaturesPerSplit &&
               left.maxThresholdCandidates ==
                   right.maxThresholdCandidates &&
               left.seed == right.seed;
    }

    bool SameMetrics(
        const eeg_to_hypnogram::ClassificationMetrics &left,
        const eeg_to_hypnogram::ClassificationMetrics &right)
    {
        return left.accuracy == right.accuracy &&
               left.macroF1 == right.macroF1 &&
               left.weightedF1 == right.weightedF1;
    }

    std::set<std::string> SubjectSet(
        const std::vector<std::string> &subjects)
    {
        return std::set<std::string>(
            subjects.begin(),
            subjects.end());
    }

    std::set<std::string> SubjectsInPairs(
        const std::vector<eeg_to_hypnogram::SleepEdfFilePair> &pairs)
    {
        std::set<std::string> subjects;
        for (const auto &pair : pairs)
        {
            subjects.insert(pair.subjectId);
        }
        return subjects;
    }

    std::string PairSignature(
        const eeg_to_hypnogram::SleepEdfFilePair &pair)
    {
        return pair.subjectId + "\n" + pair.recordingId + "\n" +
               pair.nightId + "\n" + pair.psgPath + "\n" +
               pair.hypnogramPath;
    }

    bool SamePair(
        const eeg_to_hypnogram::SleepEdfFilePair &left,
        const eeg_to_hypnogram::SleepEdfFilePair &right)
    {
        return left.subjectId == right.subjectId &&
               left.recordingId == right.recordingId &&
               left.nightId == right.nightId &&
               left.psgPath == right.psgPath &&
               left.hypnogramPath == right.hypnogramPath;
    }

    bool SamePairs(
        const std::vector<eeg_to_hypnogram::SleepEdfFilePair> &left,
        const std::vector<eeg_to_hypnogram::SleepEdfFilePair> &right)
    {
        if (left.size() != right.size())
        {
            return false;
        }
        for (std::size_t index = 0; index < left.size(); ++index)
        {
            if (!SamePair(left[index], right[index]))
            {
                return false;
            }
        }
        return true;
    }

    std::vector<eeg_to_hypnogram::SleepEdfFilePair>
    FilterPairsBySubjects(
        const std::vector<eeg_to_hypnogram::SleepEdfFilePair> &pairs,
        const std::set<std::string> &subjects)
    {
        std::vector<eeg_to_hypnogram::SleepEdfFilePair> filtered;
        for (const auto &pair : pairs)
        {
            if (subjects.count(pair.subjectId) != 0)
            {
                filtered.push_back(pair);
            }
        }
        return filtered;
    }

    std::string ReadText(const fs::path &path)
    {
        std::ifstream input(path);
        Require(
            static_cast<bool>(input),
            "Failed to open report: " + path.string());
        return std::string(
            std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>());
    }

    struct BuiltDataset
    {
        std::vector<std::vector<double>> X;
        std::vector<int> y;
        eeg_to_hypnogram::FeaturePipelineSummary summary;
    };

    BuiltDataset BuildDataset(
        const eeg_to_hypnogram::DatasetManifest &manifest,
        const eeg_to_hypnogram::TemporalContextConfig &contextConfig,
        const std::string &name)
    {
        BuiltDataset dataset;
        eeg_to_hypnogram::AppendDatasetFromManifest(
            manifest,
            name,
            contextConfig,
            &dataset.X,
            &dataset.y,
            &dataset.summary);
        return dataset;
    }

    int ConfusionMatrixTotal(
        const std::vector<std::vector<int>> &confusionMatrix)
    {
        int total = 0;
        for (const auto &row : confusionMatrix)
        {
            for (int value : row)
            {
                total += value;
            }
        }
        return total;
    }

    void RequireMetricsValid(
        const eeg_to_hypnogram::ClassificationMetrics &metrics)
    {
        Require(metrics.accuracy >= 0.0 && metrics.accuracy <= 1.0, "Accuracy must be in [0, 1].");
        Require(metrics.macroF1 >= 0.0 && metrics.macroF1 <= 1.0, "Macro F1 must be in [0, 1].");
        Require(metrics.weightedF1 >= 0.0 && metrics.weightedF1 <= 1.0, "Weighted F1 must be in [0, 1].");
    }

    void RequireFiveClassConfusionMatrix(
        const std::vector<std::vector<int>> &confusionMatrix)
    {
        Require(confusionMatrix.size() == kNumClasses, "Confusion matrix must have five rows.");
        for (const auto &row : confusionMatrix)
        {
            Require(row.size() == kNumClasses, "Confusion matrix must have five columns.");
        }
    }

    void RequirePredictionsValid(
        const std::vector<int> &predictions)
    {
        for (int prediction : predictions)
        {
            Require(
                prediction >= 0 && prediction < kNumClasses,
                "Predictions must stay in [0, 4].");
        }
    }

    void RequireOuterSplitIntegrity(
        const eeg_to_hypnogram::DatasetManifest &manifest,
        const eeg_to_hypnogram::DatasetManifestSplit &split)
    {
        const std::set<std::string> trainSubjects =
            SubjectSet(split.trainSubjectIds);
        const std::set<std::string> testSubjects =
            SubjectSet(split.testSubjectIds);

        for (const auto &subject : trainSubjects)
        {
            Require(
                testSubjects.count(subject) == 0,
                "Train/test subjects must be disjoint.");
        }

        Require(
            split.train.pairs.size() + split.test.pairs.size() ==
                manifest.pairs.size(),
            "All pairs must be retained.");

        std::set<std::string> sourcePairs;
        std::set<std::string> splitPairs;
        for (const auto &pair : manifest.pairs)
        {
            sourcePairs.insert(PairSignature(pair));
        }
        for (const auto &pair : split.train.pairs)
        {
            splitPairs.insert(PairSignature(pair));
        }
        for (const auto &pair : split.test.pairs)
        {
            splitPairs.insert(PairSignature(pair));
        }
        Require(
            sourcePairs == splitPairs,
            "Each source pair must appear exactly once in split output.");

        Require(
            SamePairs(
                FilterPairsBySubjects(manifest.pairs, trainSubjects),
                split.train.pairs),
            "Train pair order must follow source manifest order.");
        Require(
            SamePairs(
                FilterPairsBySubjects(manifest.pairs, testSubjects),
                split.test.pairs),
            "Test pair order must follow source manifest order.");
    }

    void RequireSelectionDoesNotUseOuterTest(
        const eeg_to_hypnogram::TrainingPipelineResult &result)
    {
        const std::set<std::string> outerTrainSubjects =
            SubjectSet(result.manifestSplit.trainSubjectIds);
        const std::set<std::string> outerTestSubjects =
            SubjectSet(result.manifestSplit.testSubjectIds);

        const std::set<std::string> fittingSubjects =
            SubjectSet(
                result.experimentResult.selectionSplit.trainSubjectIds);
        const std::set<std::string> validationSubjects =
            SubjectSet(
                result.experimentResult.selectionSplit.testSubjectIds);

        for (const std::string &subject : fittingSubjects)
        {
            Require(
                outerTrainSubjects.count(subject) != 0,
                "Fitting subject must come from outer train subjects.");
            Require(
                outerTestSubjects.count(subject) == 0,
                "Outer test subject leaked into fitting split.");
        }
        for (const std::string &subject : validationSubjects)
        {
            Require(
                outerTrainSubjects.count(subject) != 0,
                "Validation subject must come from outer train subjects.");
            Require(
                outerTestSubjects.count(subject) == 0,
                "Outer test subject leaked into validation split.");
        }
    }

    void TestDefaultConfiguration()
    {
        const eeg_to_hypnogram::TrainingPipelineConfig config;
        Require(config.splitConfig.testFraction == 0.2, "Default outer test fraction changed.");
        Require(config.splitConfig.randomSeed == 42U, "Default outer split seed changed.");
        Require(config.selectionSplitConfig.testFraction == 0.2, "Default selection validation fraction changed.");
        Require(config.selectionSplitConfig.randomSeed == 42U, "Default selection split seed changed.");
        Require(config.experimentConfig.numClasses == 5, "Default class count changed.");
        Require(!config.modelOutputPath.empty(), "Default model path must be set.");
        Require(!config.reportOutputPath.empty(), "Default report path must be set.");
        Require(config.modelOutputPath != config.reportOutputPath, "Default output paths must differ.");

        const eeg_to_hypnogram::TrainingPipelineResult result;
        Require(result.trainSampleCount == 0, "Default result train sample count must be zero.");
        Require(result.testSampleCount == 0, "Default result test sample count must be zero.");
        Require(result.featureCount == 0, "Default result feature count must be zero.");
        Require(result.experimentResult.trials.empty(), "Default experiment result must have no trials.");
    }

    void TestInvalidManifests(const fs::path &directory)
    {
        const auto config =
            FastConfig(directory, "invalid-manifest");

        RequireThrows<std::invalid_argument>(
            [&]
            {
                (void)eeg_to_hypnogram::RunTrainingPipeline(
                    eeg_to_hypnogram::DatasetManifest{},
                    config);
            },
            "Empty manifest must be rejected.");

        eeg_to_hypnogram::DatasetManifest dirty =
            MakeFakeManifest(4);
        dirty.ignoredFiles.push_back("/dataset/README.txt");
        dirty.unmatchedPsgFiles.push_back("SC9991E0-PSG.edf");

        RequireThrows<std::invalid_argument>(
            [&]
            {
                (void)eeg_to_hypnogram::RunTrainingPipeline(
                    dirty,
                    config);
            },
            "Dirty manifest must be rejected before EDF loading.");
    }

    void TestSyntheticEndToEnd(
        const eeg_to_hypnogram::DatasetManifest &manifest,
        const fs::path &directory)
    {
        const auto config =
            FastConfig(directory, "synthetic-end-to-end");

        const auto result =
            eeg_to_hypnogram::RunTrainingPipeline(
                manifest,
                config);

        Require(
            result.manifestSplit.train.ignoredFiles.empty(),
            "Ignored files must not be copied to train split.");
        Require(
            result.manifestSplit.test.ignoredFiles.empty(),
            "Ignored files must not be copied to test split.");
        RequireOuterSplitIntegrity(manifest, result.manifestSplit);
        RequireSelectionDoesNotUseOuterTest(result);

        Require(result.trainSampleCount > 0, "Train dataset must be non-empty.");
        Require(result.testSampleCount > 0, "Test dataset must be non-empty.");
        Require(
            result.trainSampleCount ==
                result.manifestSplit.train.pairs.size() * 5,
            "Synthetic train sample count must match pair epochs.");
        Require(
            result.testSampleCount ==
                result.manifestSplit.test.pairs.size() * 5,
            "Synthetic test sample count must match pair epochs.");
        Require(
            result.featureCount ==
                static_cast<std::size_t>(
                    result.trainDatasetSummary.temporalFeatureDim),
            "Feature count must match train summary.");
        Require(
            result.featureCount ==
                static_cast<std::size_t>(
                    result.testDatasetSummary.temporalFeatureDim),
            "Feature count must match test summary.");
        Require(result.featureCount > 0, "Feature count must be positive.");
        Require(
            result.trainDatasetSummary.channelCount ==
                kSyntheticChannelCount,
            "Synthetic channel count changed.");
        Require(
            result.trainDatasetSummary.baseFeatureDim ==
                kSyntheticChannelCount * 5,
            "Synthetic base feature dimension changed.");
        Require(
            result.trainDatasetSummary.temporalFeatureDim ==
                result.trainDatasetSummary.baseFeatureDim,
            "No-context temporal dimension should equal base dimension.");

        Require(
            result.experimentResult.fittingSampleCount > 0,
            "Fitting split must be non-empty.");
        Require(
            result.experimentResult.validationSampleCount > 0,
            "Validation split must be non-empty.");
        Require(
            result.experimentResult.fittingSampleCount +
                    result.experimentResult.validationSampleCount ==
                result.trainSampleCount,
            "Selection fitting/validation samples must partition train samples.");
        Require(
            result.experimentResult.trials.size() == 1,
            "Fast grid search should execute exactly one trial.");
        Require(
            SameConfig(
                result.experimentResult.bestConfig,
                result.experimentResult.trials.front().config),
            "Best config must match the top-ranked trial.");

        RequireFiveClassConfusionMatrix(
            result.finalEvaluation.confusionMatrix);
        Require(
            ConfusionMatrixTotal(result.finalEvaluation.confusionMatrix) ==
                static_cast<int>(result.testSampleCount),
            "Final confusion matrix support must match test samples.");
        RequireMetricsValid(result.testMetrics);
        RequireNear(
            result.finalEvaluation.testAccuracy,
            result.testMetrics.accuracy,
            1e-12,
            "Final test accuracy must match derived metrics.");

        Require(
            fs::exists(config.modelOutputPath),
            "Model file must be written.");
        Require(
            fs::exists(config.reportOutputPath),
            "Report file must be written.");

        const BuiltDataset trainDataset =
            BuildDataset(
                result.manifestSplit.train,
                config.experimentConfig.temporalContext,
                "saved-model-train-check");
        const BuiltDataset testDataset =
            BuildDataset(
                result.manifestSplit.test,
                config.experimentConfig.temporalContext,
                "saved-model-test-check");

        Require(
            trainDataset.X.size() == result.trainSampleCount,
            "Rebuilt train dataset sample count changed.");
        Require(
            testDataset.X.size() == result.testSampleCount,
            "Rebuilt test dataset sample count changed.");
        Require(
            trainDataset.X.front().size() == result.featureCount,
            "Rebuilt train feature dimension changed.");
        Require(
            testDataset.X.front().size() == result.featureCount,
            "Rebuilt test feature dimension changed.");
        Require(
            trainDataset.X.size() == trainDataset.y.size(),
            "Rebuilt train X/y counts must match.");
        Require(
            testDataset.X.size() == testDataset.y.size(),
            "Rebuilt test X/y counts must match.");

        eeg_to_hypnogram::RandomForestModel expectedModel;
        expectedModel.Train(
            trainDataset.X,
            trainDataset.y,
            kNumClasses,
            result.experimentResult.bestConfig);
        const std::vector<int> expectedPredictions =
            expectedModel.PredictBatch(testDataset.X);

        eeg_to_hypnogram::RandomForestModel loadedModel;
        loadedModel.LoadBinary(config.modelOutputPath);
        const std::vector<int> loadedPredictions =
            loadedModel.PredictBatch(testDataset.X);

        Require(
            loadedPredictions.size() == testDataset.X.size(),
            "Loaded model must predict every test sample.");
        RequirePredictionsValid(loadedPredictions);
        Require(
            loadedPredictions == expectedPredictions,
            "Saved model predictions must match freshly trained final model.");

        const std::string report =
            ReadText(config.reportOutputPath);
        Require(
            report.find("dataset_total_pairs: 6") != std::string::npos,
            "Report must include total pair count.");
        Require(
            report.find("train_subjects:") != std::string::npos,
            "Report must include train subject count.");
        Require(
            report.find("test_subjects:") != std::string::npos,
            "Report must include test subject count.");
        Require(
            report.find("temporal_context_left: 0") != std::string::npos,
            "Report must include temporal context.");
        Require(
            report.find("best_random_forest_num_trees: 5") !=
                std::string::npos,
            "Report must include best random forest parameters.");
        Require(
            report.find("five_class_confusion_matrix:") !=
                std::string::npos,
            "Report must include confusion matrix.");
        Require(
            report.find("classification_report:") !=
                std::string::npos,
            "Report must include class metrics.");
        Require(
            report.find("model_output_path:") != std::string::npos,
            "Report must include model path.");
        Require(
            report.find("report_output_path:") != std::string::npos,
            "Report must include report path.");
    }

    void TestFixedSeedReproducibility(
        const eeg_to_hypnogram::DatasetManifest &manifest,
        const fs::path &directory)
    {
        const auto config =
            FastConfig(directory, "reproducibility");

        const auto first =
            eeg_to_hypnogram::RunTrainingPipeline(
                manifest,
                config);
        const auto second =
            eeg_to_hypnogram::RunTrainingPipeline(
                manifest,
                config);

        Require(
            first.manifestSplit.testSubjectIds ==
                second.manifestSplit.testSubjectIds,
            "Same outer split seed must produce the same test subjects.");
        Require(
            first.experimentResult.selectionSplit.testSubjectIds ==
                second.experimentResult.selectionSplit.testSubjectIds,
            "Same selection split seed must produce the same validation subjects.");
        Require(
            SameConfig(
                first.experimentResult.bestConfig,
                second.experimentResult.bestConfig),
            "Same seeds must produce the same best config.");
        Require(
            first.finalEvaluation.confusionMatrix ==
                second.finalEvaluation.confusionMatrix,
            "Same seeds must produce the same final confusion matrix.");
        Require(
            SameMetrics(first.testMetrics, second.testMetrics),
            "Same seeds must produce the same final metrics.");
    }

    void TestDifferentSplitSeeds(
        const eeg_to_hypnogram::DatasetManifest &manifest,
        const fs::path &directory)
    {
        auto firstConfig =
            FastConfig(directory, "different-seed-a");
        const std::set<std::string> baselineTestSubjects =
            SubjectSet(
                eeg_to_hypnogram::SplitDatasetManifestBySubject(
                    manifest,
                    firstConfig.splitConfig)
                    .testSubjectIds);

        bool foundDifferentSeed = false;
        std::uint32_t differentSeed = 0;
        for (std::uint32_t seed = 1; seed <= 64; ++seed)
        {
            if (seed == firstConfig.splitConfig.randomSeed)
            {
                continue;
            }

            auto candidateConfig = firstConfig.splitConfig;
            candidateConfig.randomSeed = seed;
            const std::set<std::string> candidateSubjects =
                SubjectSet(
                    eeg_to_hypnogram::SplitDatasetManifestBySubject(
                        manifest,
                        candidateConfig)
                        .testSubjectIds);
            if (candidateSubjects != baselineTestSubjects)
            {
                foundDifferentSeed = true;
                differentSeed = seed;
                break;
            }
        }
        Require(
            foundDifferentSeed,
            "A different split seed should change outer test subjects.");

        auto secondConfig =
            FastConfig(directory, "different-seed-b");
        secondConfig.splitConfig.randomSeed = differentSeed;

        const auto first =
            eeg_to_hypnogram::RunTrainingPipeline(
                manifest,
                firstConfig);
        const auto second =
            eeg_to_hypnogram::RunTrainingPipeline(
                manifest,
                secondConfig);

        Require(
            SubjectSet(first.manifestSplit.testSubjectIds) !=
                SubjectSet(second.manifestSplit.testSubjectIds),
            "Training pipeline must honor different outer split seeds.");
    }

    void RunRealSleepEdfSmokeIfConfigured(const fs::path &directory)
    {
        const char *datasetDirectory =
            std::getenv("EEG_SLEEP_EDF_DATASET_DIR");

        if (datasetDirectory == nullptr || *datasetDirectory == '\0')
        {
            std::cout
                << "[SKIP] optional real training pipeline smoke "
                << "(EEG_SLEEP_EDF_DATASET_DIR not set)\n";
            return;
        }

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
            SubjectsInPairs(manifest.pairs).size() == 78,
            "Real Sleep-EDF subject count must remain 78.");

        auto config =
            FastConfig(directory, "real-sleep-edf-smoke");
        config.splitConfig.testFraction = 0.2;
        config.splitConfig.randomSeed = 42;
        config.selectionSplitConfig.testFraction = 0.2;
        config.selectionSplitConfig.randomSeed = 123;
        config.experimentConfig.gridSearch.numTreesSet = {3};
        config.experimentConfig.gridSearch.maxDepthSet = {3};
        config.experimentConfig.gridSearch.minSamplesSplitSet = {2};
        config.experimentConfig.gridSearch.maxThresholdCandidates = 4;
        config.experimentConfig.gridSearch.maxFeaturesPerSplit = 0;
        config.experimentConfig.gridSearch.seed = 321;
        config.experimentConfig.gridSearch.rankingTopK = 1;

        const auto result =
            eeg_to_hypnogram::RunTrainingPipeline(
                manifest,
                config);

        Require(
            result.manifestSplit.trainSubjectIds.size() == 62,
            "Real train subject count must remain 62.");
        Require(
            result.manifestSplit.testSubjectIds.size() == 16,
            "Real test subject count must remain 16.");
        Require(
            result.manifestSplit.train.pairs.size() == 121,
            "Real train pair count must remain 121.");
        Require(
            result.manifestSplit.test.pairs.size() == 32,
            "Real test pair count must remain 32.");
        Require(result.trainSampleCount > 0, "Real XTrain must be non-empty.");
        Require(result.testSampleCount > 0, "Real XTest must be non-empty.");
        Require(result.featureCount > 0, "Real feature count must be positive.");
        Require(
            static_cast<std::size_t>(
                result.trainDatasetSummary.temporalFeatureDim) ==
                result.featureCount,
            "Real train feature summary must match feature count.");
        Require(
            static_cast<std::size_t>(
                result.testDatasetSummary.temporalFeatureDim) ==
                result.featureCount,
            "Real test feature summary must match feature count.");
        Require(
            ConfusionMatrixTotal(result.finalEvaluation.confusionMatrix) ==
                static_cast<int>(result.testSampleCount),
            "Real predictions must cover every test sample.");
        RequireMetricsValid(result.testMetrics);
        RequireFiveClassConfusionMatrix(
            result.finalEvaluation.confusionMatrix);
        Require(fs::exists(config.modelOutputPath), "Real model file must exist.");
        Require(fs::exists(config.reportOutputPath), "Real report file must exist.");
    }

} // namespace

int main()
{
    try
    {
        TemporaryDirectory temporaryDirectory;
        const eeg_to_hypnogram::DatasetManifest syntheticManifest =
            CreateSyntheticManifest(temporaryDirectory.Path());

        Run("default configuration", TestDefaultConfiguration);
        Run(
            "invalid manifests",
            [&]
            {
                TestInvalidManifests(temporaryDirectory.Path());
            });
        Run(
            "synthetic end-to-end training pipeline",
            [&]
            {
                TestSyntheticEndToEnd(
                    syntheticManifest,
                    temporaryDirectory.Path());
            });
        Run(
            "fixed seed reproducibility",
            [&]
            {
                TestFixedSeedReproducibility(
                    syntheticManifest,
                    temporaryDirectory.Path());
            });
        Run(
            "different split seeds",
            [&]
            {
                TestDifferentSplitSeeds(
                    syntheticManifest,
                    temporaryDirectory.Path());
            });
        Run(
            "optional real Sleep-EDF smoke",
            [&]
            {
                RunRealSleepEdfSmokeIfConfigured(
                    temporaryDirectory.Path());
            });

        std::cout << "All enabled Training Pipeline tests passed.\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
