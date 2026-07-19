#include "eeg_to_hypnogram/training_pipeline.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

    namespace fs = std::filesystem;

    constexpr int kSleepStageClassCount = 5;
    constexpr std::array<const char *, kSleepStageClassCount> kClassNames = {
        "W",
        "N1",
        "N2",
        "N3",
        "REM",
    };

    struct ClassMetrics
    {
        double precision = 0.0;
        double recall = 0.0;
        double f1 = 0.0;
        int support = 0;
    };

    std::set<std::string> CollectSubjects(
        const eeg_to_hypnogram::DatasetManifest &manifest)
    {
        std::set<std::string> subjects;
        for (const auto &pair : manifest.pairs)
        {
            subjects.insert(pair.subjectId);
        }
        return subjects;
    }

    std::set<std::string> CollectSubjects(
        const std::vector<std::string> &subjectIds)
    {
        return std::set<std::string>(
            subjectIds.begin(),
            subjectIds.end());
    }

    void EnsureOutputPath(
        const std::string &path,
        const std::string &name)
    {
        if (path.empty())
        {
            throw std::invalid_argument(
                name + " must not be empty.");
        }

        const fs::path outputPath(path);
        const fs::path parentPath = outputPath.parent_path();
        if (parentPath.empty())
        {
            return;
        }

        std::error_code error;
        fs::create_directories(parentPath, error);
        if (error)
        {
            throw std::runtime_error(
                "Failed to create parent directory for " + name + ": " +
                parentPath.string() + ": " + error.message());
        }
    }

    void ValidateConfig(
        const eeg_to_hypnogram::TrainingPipelineConfig &config)
    {
        if (config.experimentConfig.numClasses !=
            kSleepStageClassCount)
        {
            throw std::invalid_argument(
                "Training pipeline requires numClasses=5 for sleep stages.");
        }

        if (config.experimentConfig.temporalContext.leftContext < 0 ||
            config.experimentConfig.temporalContext.rightContext < 0)
        {
            throw std::invalid_argument(
                "Temporal context sizes must be >= 0.");
        }

        if (config.modelOutputPath == config.reportOutputPath)
        {
            throw std::invalid_argument(
                "modelOutputPath and reportOutputPath must be different.");
        }

        EnsureOutputPath(config.modelOutputPath, "modelOutputPath");
        EnsureOutputPath(config.reportOutputPath, "reportOutputPath");
    }

    void ValidateDisjointSubjects(
        const eeg_to_hypnogram::DatasetManifestSplit &split,
        const std::string &where)
    {
        const std::set<std::string> trainSubjects =
            CollectSubjects(split.trainSubjectIds);
        const std::set<std::string> testSubjects =
            CollectSubjects(split.testSubjectIds);

        if (trainSubjects.size() != split.trainSubjectIds.size())
        {
            throw std::logic_error(
                where + " train subject list contains duplicates.");
        }
        if (testSubjects.size() != split.testSubjectIds.size())
        {
            throw std::logic_error(
                where + " test subject list contains duplicates.");
        }

        for (const std::string &subject : trainSubjects)
        {
            if (testSubjects.count(subject) != 0)
            {
                throw std::logic_error(
                    where + " split contains overlapping subject: " +
                    subject);
            }
        }
    }

    void ValidateSelectionSplitUsesOnlyOuterTrain(
        const eeg_to_hypnogram::DatasetManifestSplit &outerSplit,
        const eeg_to_hypnogram::DatasetManifestSplit &selectionSplit)
    {
        const std::set<std::string> outerTrainSubjects =
            CollectSubjects(outerSplit.trainSubjectIds);
        const std::set<std::string> outerTestSubjects =
            CollectSubjects(outerSplit.testSubjectIds);

        for (const auto &subject : selectionSplit.trainSubjectIds)
        {
            if (outerTrainSubjects.count(subject) == 0 ||
                outerTestSubjects.count(subject) != 0)
            {
                throw std::logic_error(
                    "Selection fitting split used a subject outside the outer train split: " +
                    subject);
            }
        }

        for (const auto &subject : selectionSplit.testSubjectIds)
        {
            if (outerTrainSubjects.count(subject) == 0 ||
                outerTestSubjects.count(subject) != 0)
            {
                throw std::logic_error(
                    "Selection validation split used a subject outside the outer train split: " +
                    subject);
            }
        }
    }

    std::size_t ValidateDataset(
        const std::vector<std::vector<double>> &X,
        const std::vector<int> &y,
        int numClasses,
        const std::string &splitName,
        bool requireAtLeastTwoLabels)
    {
        if (X.empty() || y.empty() || X.size() != y.size())
        {
            throw std::invalid_argument(
                splitName + " X/y is empty or size mismatch.");
        }

        const std::size_t featureDim = X.front().size();
        if (featureDim == 0)
        {
            throw std::invalid_argument(
                splitName + " feature dimension must be > 0.");
        }

        for (const auto &row : X)
        {
            if (row.size() != featureDim)
            {
                throw std::invalid_argument(
                    "Inconsistent feature dimensions in " + splitName + ".");
            }
        }

        std::set<int> observedLabels;
        for (int label : y)
        {
            if (label < 0 || label >= numClasses)
            {
                throw std::invalid_argument(
                    splitName + " contains a label outside [0, 4].");
            }
            observedLabels.insert(label);
        }

        if (requireAtLeastTwoLabels && observedLabels.size() < 2)
        {
            throw std::invalid_argument(
                splitName + " must contain at least two trainable classes.");
        }

        return featureDim;
    }

    void ValidateSummaryMatchesFeatureDim(
        const eeg_to_hypnogram::FeaturePipelineSummary &summary,
        std::size_t featureDim,
        const std::string &where)
    {
        if (summary.temporalFeatureDim <= 0 ||
            static_cast<std::size_t>(summary.temporalFeatureDim) !=
                featureDim)
        {
            throw std::runtime_error(
                where + " feature summary does not match the feature matrix.");
        }
    }

    void ValidateCompatibleSummaries(
        const eeg_to_hypnogram::FeaturePipelineSummary &left,
        const eeg_to_hypnogram::FeaturePipelineSummary &right,
        const std::string &where)
    {
        if (left.epochSeconds != right.epochSeconds ||
            left.targetSampleRateHz != right.targetSampleRateHz ||
            left.channelCount != right.channelCount ||
            left.baseFeatureDim != right.baseFeatureDim ||
            left.temporalFeatureDim != right.temporalFeatureDim ||
            left.channelLabels != right.channelLabels ||
            left.channelSampleRatesHz != right.channelSampleRatesHz)
        {
            throw std::runtime_error(
                where + " feature pipeline summaries are incompatible.");
        }
    }

    double Accuracy(
        const std::vector<int> &actual,
        const std::vector<int> &predicted)
    {
        if (actual.size() != predicted.size())
        {
            throw std::logic_error(
                "Prediction count does not match label count.");
        }
        if (actual.empty())
        {
            return 0.0;
        }

        std::size_t correct = 0;
        for (std::size_t index = 0; index < actual.size(); ++index)
        {
            if (actual[index] == predicted[index])
            {
                ++correct;
            }
        }
        return static_cast<double>(correct) /
               static_cast<double>(actual.size());
    }

    std::vector<std::vector<int>> BuildConfusionMatrix(
        const std::vector<int> &actual,
        const std::vector<int> &predicted,
        int numClasses)
    {
        if (actual.size() != predicted.size())
        {
            throw std::logic_error(
                "Prediction count does not match label count.");
        }

        std::vector<std::vector<int>> confusionMatrix(
            static_cast<std::size_t>(numClasses),
            std::vector<int>(static_cast<std::size_t>(numClasses), 0));

        for (std::size_t index = 0; index < actual.size(); ++index)
        {
            const int expected = actual[index];
            const int predictedLabel = predicted[index];

            if (expected < 0 || expected >= numClasses ||
                predictedLabel < 0 || predictedLabel >= numClasses)
            {
                throw std::runtime_error(
                    "Final model produced or received a label outside [0, 4].");
            }

            ++confusionMatrix[static_cast<std::size_t>(expected)]
                             [static_cast<std::size_t>(predictedLabel)];
        }

        return confusionMatrix;
    }

    std::vector<ClassMetrics> ComputePerClassMetrics(
        const std::vector<std::vector<int>> &confusionMatrix)
    {
        (void)eeg_to_hypnogram::ComputeClassificationMetrics(
            confusionMatrix);

        std::vector<ClassMetrics> metrics;
        metrics.reserve(confusionMatrix.size());

        for (std::size_t classIndex = 0;
             classIndex < confusionMatrix.size();
             ++classIndex)
        {
            const int truePositive =
                confusionMatrix[classIndex][classIndex];

            int rowSum = 0;
            int columnSum = 0;
            for (std::size_t other = 0;
                 other < confusionMatrix.size();
                 ++other)
            {
                rowSum += confusionMatrix[classIndex][other];
                columnSum += confusionMatrix[other][classIndex];
            }

            ClassMetrics classMetrics;
            classMetrics.support = rowSum;
            classMetrics.precision =
                columnSum > 0
                    ? static_cast<double>(truePositive) /
                          static_cast<double>(columnSum)
                    : 0.0;
            classMetrics.recall =
                rowSum > 0
                    ? static_cast<double>(truePositive) /
                          static_cast<double>(rowSum)
                    : 0.0;
            classMetrics.f1 =
                classMetrics.precision + classMetrics.recall > 0.0
                    ? (2.0 * classMetrics.precision *
                       classMetrics.recall) /
                          (classMetrics.precision +
                           classMetrics.recall)
                    : 0.0;
            metrics.push_back(classMetrics);
        }

        return metrics;
    }

    std::string BuildReportText(
        const eeg_to_hypnogram::DatasetManifest &manifest,
        const eeg_to_hypnogram::TrainingPipelineConfig &config,
        const eeg_to_hypnogram::TrainingPipelineResult &result)
    {
        const std::set<std::string> allSubjects =
            CollectSubjects(manifest);
        const auto classMetrics =
            ComputePerClassMetrics(
                result.finalEvaluation.confusionMatrix);

        std::ostringstream report;
        report << std::fixed << std::setprecision(6);

        report << "Training Pipeline Report\n";
        report << "dataset_total_pairs: " << manifest.pairs.size()
               << '\n';
        report << "total_subjects: " << allSubjects.size() << '\n';
        report << "train_subjects: "
               << result.manifestSplit.trainSubjectIds.size() << '\n';
        report << "test_subjects: "
               << result.manifestSplit.testSubjectIds.size() << '\n';
        report << "train_pairs: "
               << result.manifestSplit.train.pairs.size() << '\n';
        report << "test_pairs: "
               << result.manifestSplit.test.pairs.size() << '\n';
        report << "split_seed: "
               << config.splitConfig.randomSeed << '\n';
        report << "test_fraction: "
               << config.splitConfig.testFraction << '\n';
        report << "selection_seed: "
               << config.selectionSplitConfig.randomSeed << '\n';
        report << "selection_validation_fraction: "
               << config.selectionSplitConfig.testFraction << '\n';
        report << "selection_fitting_samples: "
               << result.experimentResult.fittingSampleCount << '\n';
        report << "selection_validation_samples: "
               << result.experimentResult.validationSampleCount << '\n';
        report << "train_samples: "
               << result.trainSampleCount << '\n';
        report << "test_samples: "
               << result.testSampleCount << '\n';
        report << "feature_count: "
               << result.featureCount << '\n';
        report << "temporal_context_left: "
               << config.experimentConfig.temporalContext.leftContext
               << '\n';
        report << "temporal_context_right: "
               << config.experimentConfig.temporalContext.rightContext
               << '\n';

        const auto &bestConfig =
            result.experimentResult.bestConfig;
        report << "best_random_forest_num_trees: "
               << bestConfig.numTrees << '\n';
        report << "best_random_forest_max_depth: "
               << bestConfig.maxDepth << '\n';
        report << "best_random_forest_min_samples_split: "
               << bestConfig.minSamplesSplit << '\n';
        report << "best_random_forest_max_features_per_split: "
               << bestConfig.maxFeaturesPerSplit << '\n';
        report << "best_random_forest_max_threshold_candidates: "
               << bestConfig.maxThresholdCandidates << '\n';
        report << "best_random_forest_seed: "
               << bestConfig.seed << '\n';

        report << "final_train_accuracy: "
               << result.finalEvaluation.trainAccuracy << '\n';
        report << "final_test_accuracy: "
               << result.testMetrics.accuracy << '\n';
        report << "final_test_macro_f1: "
               << result.testMetrics.macroF1 << '\n';
        report << "final_test_weighted_f1: "
               << result.testMetrics.weightedF1 << '\n';

        report << "five_class_confusion_matrix:\n";
        for (const auto &row : result.finalEvaluation.confusionMatrix)
        {
            report << '[';
            for (std::size_t column = 0; column < row.size(); ++column)
            {
                if (column != 0)
                {
                    report << ' ';
                }
                report << row[column];
            }
            report << "]\n";
        }

        report << "classification_report:\n";
        report << "class\tprecision\trecall\tf1\tsupport\n";
        for (std::size_t classIndex = 0;
             classIndex < classMetrics.size();
             ++classIndex)
        {
            const std::string className =
                classIndex < kClassNames.size()
                    ? kClassNames[classIndex]
                    : std::to_string(classIndex);
            const auto &metrics = classMetrics[classIndex];
            report << className << '\t'
                   << metrics.precision << '\t'
                   << metrics.recall << '\t'
                   << metrics.f1 << '\t'
                   << metrics.support << '\n';
        }

        report << "model_output_path: "
               << config.modelOutputPath << '\n';
        report << "report_output_path: "
               << config.reportOutputPath << '\n';

        return report.str();
    }

    void WriteReportFile(
        const eeg_to_hypnogram::DatasetManifest &manifest,
        const eeg_to_hypnogram::TrainingPipelineConfig &config,
        const eeg_to_hypnogram::TrainingPipelineResult &result)
    {
        std::ofstream output(config.reportOutputPath);
        if (!output)
        {
            throw std::runtime_error(
                "Cannot open training report for writing: " +
                config.reportOutputPath);
        }

        output << BuildReportText(manifest, config, result);
        if (!output)
        {
            throw std::runtime_error(
                "Failed while writing training report: " +
                config.reportOutputPath);
        }
    }

} // namespace

namespace eeg_to_hypnogram
{

    TrainingPipelineResult RunTrainingPipeline(
        const DatasetManifest &manifest,
        const TrainingPipelineConfig &config)
    {
        ValidateConfig(config);
        ValidateManifestForDatasetAssembly(manifest);

        TrainingPipelineResult result;
        result.manifestSplit =
            SplitDatasetManifestBySubject(
                manifest,
                config.splitConfig);

        ValidateManifestForDatasetAssembly(result.manifestSplit.train);
        ValidateManifestForDatasetAssembly(result.manifestSplit.test);
        ValidateDisjointSubjects(
            result.manifestSplit,
            "Outer train/test");

        std::vector<std::vector<double>> XTrain;
        std::vector<int> yTrain;
        AppendDatasetFromManifest(
            result.manifestSplit.train,
            "train",
            config.experimentConfig.temporalContext,
            &XTrain,
            &yTrain,
            &result.trainDatasetSummary);

        std::vector<std::vector<double>> XTest;
        std::vector<int> yTest;
        AppendDatasetFromManifest(
            result.manifestSplit.test,
            "test",
            config.experimentConfig.temporalContext,
            &XTest,
            &yTest,
            &result.testDatasetSummary);

        const std::size_t trainFeatureDim =
            ValidateDataset(
                XTrain,
                yTrain,
                config.experimentConfig.numClasses,
                "train",
                true);
        const std::size_t testFeatureDim =
            ValidateDataset(
                XTest,
                yTest,
                config.experimentConfig.numClasses,
                "test",
                false);

        if (trainFeatureDim != testFeatureDim)
        {
            throw std::runtime_error(
                "Train/test feature dimension mismatch.");
        }

        ValidateSummaryMatchesFeatureDim(
            result.trainDatasetSummary,
            trainFeatureDim,
            "Train");
        ValidateSummaryMatchesFeatureDim(
            result.testDatasetSummary,
            testFeatureDim,
            "Test");
        ValidateCompatibleSummaries(
            result.trainDatasetSummary,
            result.testDatasetSummary,
            "Train/test");

        result.trainSampleCount = XTrain.size();
        result.testSampleCount = XTest.size();
        result.featureCount = trainFeatureDim;

        result.experimentResult.selectionSplit =
            SplitDatasetManifestBySubject(
                result.manifestSplit.train,
                config.selectionSplitConfig);
        ValidateDisjointSubjects(
            result.experimentResult.selectionSplit,
            "Selection fitting/validation");
        ValidateSelectionSplitUsesOnlyOuterTrain(
            result.manifestSplit,
            result.experimentResult.selectionSplit);

        std::vector<std::vector<double>> XFitting;
        std::vector<int> yFitting;
        FeaturePipelineSummary fittingSummary;
        AppendDatasetFromManifest(
            result.experimentResult.selectionSplit.train,
            "fitting",
            config.experimentConfig.temporalContext,
            &XFitting,
            &yFitting,
            &fittingSummary);

        std::vector<std::vector<double>> XValidation;
        std::vector<int> yValidation;
        FeaturePipelineSummary validationSummary;
        AppendDatasetFromManifest(
            result.experimentResult.selectionSplit.test,
            "validation",
            config.experimentConfig.temporalContext,
            &XValidation,
            &yValidation,
            &validationSummary);

        const std::size_t fittingFeatureDim =
            ValidateDataset(
                XFitting,
                yFitting,
                config.experimentConfig.numClasses,
                "fitting",
                true);
        const std::size_t validationFeatureDim =
            ValidateDataset(
                XValidation,
                yValidation,
                config.experimentConfig.numClasses,
                "validation",
                false);

        if (fittingFeatureDim != trainFeatureDim ||
            validationFeatureDim != trainFeatureDim)
        {
            throw std::runtime_error(
                "Selection feature dimensions must match the outer train dataset.");
        }

        ValidateSummaryMatchesFeatureDim(
            fittingSummary,
            fittingFeatureDim,
            "Fitting");
        ValidateSummaryMatchesFeatureDim(
            validationSummary,
            validationFeatureDim,
            "Validation");
        ValidateCompatibleSummaries(
            result.trainDatasetSummary,
            fittingSummary,
            "Train/fitting");
        ValidateCompatibleSummaries(
            result.trainDatasetSummary,
            validationSummary,
            "Train/validation");

        result.experimentResult.fittingSampleCount = XFitting.size();
        result.experimentResult.validationSampleCount =
            XValidation.size();

        result.experimentResult.trials =
            RunSmallGridSearch(
                XFitting,
                yFitting,
                XValidation,
                yValidation,
                config.experimentConfig.numClasses,
                config.experimentConfig.gridSearch);

        if (result.experimentResult.trials.empty())
        {
            throw std::runtime_error(
                "Grid search returned no trials.");
        }
        result.experimentResult.bestConfig =
            result.experimentResult.trials.front().config;

        RandomForestModel finalModel;
        finalModel.Train(
            XTrain,
            yTrain,
            config.experimentConfig.numClasses,
            result.experimentResult.bestConfig);

        const std::vector<int> trainPredictions =
            finalModel.PredictBatch(XTrain);
        const std::vector<int> testPredictions =
            finalModel.PredictBatch(XTest);

        result.finalEvaluation.trainAccuracy =
            Accuracy(yTrain, trainPredictions);
        result.finalEvaluation.testAccuracy =
            Accuracy(yTest, testPredictions);
        result.finalEvaluation.confusionMatrix =
            BuildConfusionMatrix(
                yTest,
                testPredictions,
                config.experimentConfig.numClasses);

        result.testMetrics =
            ComputeClassificationMetrics(
                result.finalEvaluation.confusionMatrix);

        finalModel.SaveBinary(config.modelOutputPath);

        RandomForestModel loadedModel;
        loadedModel.LoadBinary(config.modelOutputPath);
        const std::vector<int> loadedPredictions =
            loadedModel.PredictBatch(XTest);
        if (loadedPredictions != testPredictions)
        {
            throw std::runtime_error(
                "Saved model predictions changed after reload.");
        }

        WriteReportFile(manifest, config, result);

        return result;
    }

} // namespace eeg_to_hypnogram
