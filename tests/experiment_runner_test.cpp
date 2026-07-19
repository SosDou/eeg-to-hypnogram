#define EEG_TO_HYPNOGRAM_EXPERIMENT_RUNNER_TESTING

#include "eeg_to_hypnogram/experiment_runner.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

    using eeg_to_hypnogram::ClassificationMetrics;
    using eeg_to_hypnogram::GridSearchConfig;
    using eeg_to_hypnogram::GridSearchTrial;
    using eeg_to_hypnogram::RandomForestConfig;
    using eeg_to_hypnogram::RandomForestResult;

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

    class CoutCapture final
    {
    public:
        CoutCapture()
            : oldBuffer_(std::cout.rdbuf(stream_.rdbuf()))
        {
        }

        ~CoutCapture()
        {
            std::cout.rdbuf(oldBuffer_);
        }

        std::string Text() const
        {
            return stream_.str();
        }

    private:
        std::ostringstream stream_;
        std::streambuf *oldBuffer_;
    };

    std::pair<std::vector<std::vector<double>>, std::vector<int>>
    MakeFiveClassData(int copiesPerClass)
    {
        std::vector<std::vector<double>> features;
        std::vector<int> labels;

        for (int classIndex = 0; classIndex < 5; ++classIndex)
        {
            for (int copy = 0; copy < copiesPerClass; ++copy)
            {
                const double jitter =
                    0.01 * static_cast<double>(copy);
                features.push_back(
                    {
                        10.0 * static_cast<double>(classIndex) + jitter,
                        static_cast<double>(classIndex),
                        static_cast<double>((classIndex + copy) % 3),
                    });
                labels.push_back(classIndex);
            }
        }

        return {features, labels};
    }

    bool SameConfig(
        const RandomForestConfig &left,
        const RandomForestConfig &right)
    {
        return left.numTrees == right.numTrees &&
               left.maxDepth == right.maxDepth &&
               left.minSamplesSplit == right.minSamplesSplit &&
               left.maxFeaturesPerSplit == right.maxFeaturesPerSplit &&
               left.maxThresholdCandidates == right.maxThresholdCandidates &&
               left.seed == right.seed;
    }

    bool SameMetrics(
        const ClassificationMetrics &left,
        const ClassificationMetrics &right)
    {
        return left.accuracy == right.accuracy &&
               left.macroF1 == right.macroF1 &&
               left.weightedF1 == right.weightedF1;
    }

    void TestDefaultConfiguration()
    {
        const eeg_to_hypnogram::TemporalContextConfig context;
        Require(context.leftContext == 2, "Default left context changed.");
        Require(context.rightContext == 2, "Default right context changed.");

        const GridSearchConfig grid;
        Require(grid.numTreesSet == std::vector<int>({60, 120}), "Default tree grid changed.");
        Require(grid.maxDepthSet == std::vector<int>({10, 14}), "Default depth grid changed.");
        Require(grid.minSamplesSplitSet == std::vector<int>({2, 6}), "Default split grid changed.");
        Require(grid.maxThresholdCandidates == 24, "Default threshold candidates changed.");
        Require(grid.maxFeaturesPerSplit == 0, "Default max features changed.");
        Require(grid.seed == 42U, "Default grid seed changed.");
        Require(grid.rankingTopK == 5, "Default rankingTopK changed.");

        const eeg_to_hypnogram::ExperimentConfig experiment;
        Require(experiment.numClasses == 5, "Default experiment class count changed.");
    }

    void TestTemporalContextRemainsAvailable()
    {
        eeg_to_hypnogram::TemporalContextConfig config;
        config.leftContext = 1;
        config.rightContext = 1;

        const std::vector<std::vector<double>> input =
            {{1.0}, {2.0}, {3.0}};
        const auto output =
            eeg_to_hypnogram::BuildTemporalContextFeatures(input, config);

        const std::vector<std::vector<double>> expected =
            {
                {1.0, 1.0, 2.0},
                {1.0, 2.0, 3.0},
                {2.0, 3.0, 3.0},
            };
        Require(output == expected, "Experiment header must expose the migrated temporal context API.");
    }

    void TestClassificationMetrics()
    {
        const std::vector<std::vector<int>> confusionMatrix =
            {
                {3, 1, 0},
                {1, 2, 1},
                {0, 0, 0},
            };

        const auto metrics =
            eeg_to_hypnogram::experiment_runner_testing::
                ComputeClassificationMetrics(confusionMatrix);

        // class 0: P=3/4, R=3/4, F1=3/4
        // class 1: P=2/3, R=1/2, F1=4/7
        // class 2: P=0,   R=0,   F1=0
        const double expectedMacroF1 =
            (0.75 + (4.0 / 7.0) + 0.0) / 3.0;
        const double expectedWeightedF1 =
            (0.75 * 4.0 + (4.0 / 7.0) * 4.0) / 8.0;

        RequireNear(metrics.accuracy, 5.0 / 8.0, 1e-12, "Accuracy formula changed.");
        RequireNear(metrics.macroF1, expectedMacroF1, 1e-12, "Macro F1 formula changed.");
        RequireNear(metrics.weightedF1, expectedWeightedF1, 1e-12, "Weighted F1 formula changed.");
    }

    void TestEmptyAndInvalidMetricsInput()
    {
        const auto empty =
            eeg_to_hypnogram::experiment_runner_testing::
                ComputeClassificationMetrics({});
        Require(empty.accuracy == 0.0, "Empty accuracy must be zero.");
        Require(empty.macroF1 == 0.0, "Empty macro F1 must be zero.");
        Require(empty.weightedF1 == 0.0, "Empty weighted F1 must be zero.");

        RequireThrows<std::invalid_argument>(
            []
            {
                (void)eeg_to_hypnogram::experiment_runner_testing::
                    ComputeClassificationMetrics({{1, 0}, {0}});
            },
            "Non-square confusion matrix must be rejected.");

        RequireThrows<std::invalid_argument>(
            []
            {
                (void)eeg_to_hypnogram::experiment_runner_testing::
                    ComputeClassificationMetrics({{1, -1}, {0, 1}});
            },
            "Negative confusion matrix counts must be rejected.");
    }

    void TestGridExpansionOrder()
    {
        GridSearchConfig config;
        config.numTreesSet = {3, 5};
        config.maxDepthSet = {2, 4};
        config.minSamplesSplitSet = {6, 8};
        config.maxThresholdCandidates = 11;
        config.maxFeaturesPerSplit = 2;
        config.seed = 99;

        const auto grid =
            eeg_to_hypnogram::experiment_runner_testing::
                BuildSmallGrid(config);

        Require(grid.size() == 8, "2 x 2 x 2 grid must contain eight trials.");

        const std::vector<std::vector<int>> expected =
            {
                {3, 2, 6},
                {3, 2, 8},
                {3, 4, 6},
                {3, 4, 8},
                {5, 2, 6},
                {5, 2, 8},
                {5, 4, 6},
                {5, 4, 8},
            };

        for (std::size_t index = 0; index < grid.size(); ++index)
        {
            Require(grid[index].numTrees == expected[index][0], "numTrees grid order changed.");
            Require(grid[index].maxDepth == expected[index][1], "maxDepth grid order changed.");
            Require(grid[index].minSamplesSplit == expected[index][2], "minSamplesSplit grid order changed.");
            Require(grid[index].maxThresholdCandidates == 11, "Threshold candidate propagation changed.");
            Require(grid[index].maxFeaturesPerSplit == 2, "Feature subset propagation changed.");
            Require(grid[index].seed == 99U, "Seed propagation changed.");
        }
    }

    void TestInvalidGridConfiguration()
    {
        GridSearchConfig config;
        config.numTreesSet.clear();
        RequireThrows<std::invalid_argument>(
            [&]
            {
                (void)eeg_to_hypnogram::experiment_runner_testing::
                    BuildSmallGrid(config);
            },
            "Empty grid range must be rejected.");

        config = GridSearchConfig();
        config.numTreesSet = {0};
        RequireThrows<std::invalid_argument>(
            [&]
            {
                (void)eeg_to_hypnogram::experiment_runner_testing::
                    BuildSmallGrid(config);
            },
            "Non-positive tree candidates must be rejected.");

        config = GridSearchConfig();
        config.maxDepthSet = {-1};
        RequireThrows<std::invalid_argument>(
            [&]
            {
                (void)eeg_to_hypnogram::experiment_runner_testing::
                    BuildSmallGrid(config);
            },
            "Negative depth candidates must be rejected.");

        config = GridSearchConfig();
        config.minSamplesSplitSet = {0};
        RequireThrows<std::invalid_argument>(
            [&]
            {
                (void)eeg_to_hypnogram::experiment_runner_testing::
                    BuildSmallGrid(config);
            },
            "Non-positive split candidates must be rejected.");
    }

    void TestRankingRules()
    {
        GridSearchTrial left;
        GridSearchTrial right;

        left.testMetrics.macroF1 = 0.8;
        right.testMetrics.macroF1 = 0.7;
        Require(
            eeg_to_hypnogram::experiment_runner_testing::
                IsBetterTrial(left, right),
            "Higher macro F1 must rank first.");

        left.testMetrics.macroF1 = right.testMetrics.macroF1 = 0.8;
        left.testMetrics.accuracy = 0.9;
        right.testMetrics.accuracy = 0.85;
        Require(
            eeg_to_hypnogram::experiment_runner_testing::
                IsBetterTrial(left, right),
            "Higher accuracy must break macro F1 ties.");

        left.testMetrics.accuracy = right.testMetrics.accuracy = 0.9;
        left.result.trainAccuracy = 0.95;
        right.result.trainAccuracy = 0.98;
        Require(
            eeg_to_hypnogram::experiment_runner_testing::
                IsBetterTrial(left, right),
            "Lower train accuracy must break the remaining tie.");

        right.result.trainAccuracy = 0.95;
        Require(
            !eeg_to_hypnogram::experiment_runner_testing::
                    IsBetterTrial(left, right) &&
                !eeg_to_hypnogram::experiment_runner_testing::
                    IsBetterTrial(right, left),
            "Complete metric ties must have no extra configuration rule.");
    }

    void TestGridSearchExecutionAndReproducibility()
    {
        const auto train = MakeFiveClassData(6);
        const auto test = MakeFiveClassData(3);

        const auto originalTrainX = train.first;
        const auto originalTrainY = train.second;
        const auto originalTestX = test.first;
        const auto originalTestY = test.second;

        GridSearchConfig config;
        config.numTreesSet = {5, 9};
        config.maxDepthSet = {3};
        config.minSamplesSplitSet = {2, 5};
        config.maxThresholdCandidates = 8;
        config.maxFeaturesPerSplit = 2;
        config.seed = 123;
        config.rankingTopK = 2;

        std::vector<GridSearchTrial> first;
        std::vector<GridSearchTrial> second;
        {
            CoutCapture capture;
            first = eeg_to_hypnogram::RunSmallGridSearch(
                train.first,
                train.second,
                test.first,
                test.second,
                5,
                config);
        }
        {
            CoutCapture capture;
            second = eeg_to_hypnogram::RunSmallGridSearch(
                train.first,
                train.second,
                test.first,
                test.second,
                5,
                config);
        }

        Require(first.size() == 4, "Grid search must execute all four combinations.");
        Require(second.size() == first.size(), "Repeated grid search size changed.");

        for (std::size_t index = 0; index < first.size(); ++index)
        {
            Require(SameConfig(first[index].config, second[index].config), "Fixed-seed ranking configuration changed.");
            Require(first[index].result.trainAccuracy == second[index].result.trainAccuracy, "Fixed-seed training accuracy changed.");
            Require(first[index].result.testAccuracy == second[index].result.testAccuracy, "Fixed-seed test accuracy changed.");
            Require(first[index].result.confusionMatrix == second[index].result.confusionMatrix, "Fixed-seed confusion matrix changed.");
            Require(SameMetrics(first[index].testMetrics, second[index].testMetrics), "Fixed-seed derived metrics changed.");

            Require(first[index].result.confusionMatrix.size() == 5, "Five-class confusion matrix row count changed.");
            for (const auto &row : first[index].result.confusionMatrix)
            {
                Require(row.size() == 5, "Five-class confusion matrix column count changed.");
            }
        }

        for (std::size_t index = 1; index < first.size(); ++index)
        {
            Require(
                !eeg_to_hypnogram::experiment_runner_testing::
                    IsBetterTrial(first[index], first[index - 1]),
                "Returned trials must be sorted by the old ranking comparator.");
        }

        Require(train.first == originalTrainX, "Grid search modified XTrain.");
        Require(train.second == originalTrainY, "Grid search modified yTrain.");
        Require(test.first == originalTestX, "Grid search modified XTest.");
        Require(test.second == originalTestY, "Grid search modified yTest.");
    }

    void TestGridSearchInputValidation()
    {
        const auto data = MakeFiveClassData(2);
        GridSearchConfig config;
        config.numTreesSet = {3};
        config.maxDepthSet = {2};
        config.minSamplesSplitSet = {2};
        config.rankingTopK = 0;

        RequireThrows<std::invalid_argument>(
            [&]
            {
                CoutCapture capture;
                (void)eeg_to_hypnogram::RunSmallGridSearch(
                    {}, {}, data.first, data.second, 5, config);
            },
            "Empty training split must be rejected.");

        RequireThrows<std::invalid_argument>(
            [&]
            {
                CoutCapture capture;
                (void)eeg_to_hypnogram::RunSmallGridSearch(
                    data.first, data.second, {}, {}, 5, config);
            },
            "Empty test split must be rejected.");

        auto badLabels = data.second;
        badLabels.front() = 5;
        RequireThrows<std::invalid_argument>(
            [&]
            {
                CoutCapture capture;
                (void)eeg_to_hypnogram::RunSmallGridSearch(
                    data.first,
                    data.second,
                    data.first,
                    badLabels,
                    5,
                    config);
            },
            "Out-of-range test labels must be rejected.");

        auto badFeatures = data.first;
        badFeatures.front().push_back(9.0);
        RequireThrows<std::invalid_argument>(
            [&]
            {
                CoutCapture capture;
                (void)eeg_to_hypnogram::RunSmallGridSearch(
                    data.first,
                    data.second,
                    badFeatures,
                    data.second,
                    5,
                    config);
            },
            "Inconsistent test feature dimensions must be rejected.");
    }

    void TestReportOutputAndValidation()
    {
        RandomForestResult result;
        result.confusionMatrix =
            {
                {2, 0, 0, 0, 0},
                {0, 1, 1, 0, 0},
                {0, 0, 2, 0, 0},
                {0, 0, 0, 2, 0},
                {0, 0, 0, 0, 2},
            };

        std::string text;
        {
            CoutCapture capture;
            eeg_to_hypnogram::PrintPythonStyleReport(result);
            text = capture.Text();
        }

        Require(text.find("准确率:") != std::string::npos, "Report must print accuracy.");
        Require(text.find("混淆矩阵") != std::string::npos, "Report must print confusion matrix.");
        Require(text.find("W\t") != std::string::npos, "Report must print W label.");
        Require(text.find("N1\t") != std::string::npos, "Report must print N1 label.");
        Require(text.find("N2\t") != std::string::npos, "Report must print N2 label.");
        Require(text.find("N3\t") != std::string::npos, "Report must print N3 label.");
        Require(text.find("REM\t") != std::string::npos, "Report must print REM label.");
        Require(text.find("macro avg") != std::string::npos, "Report must print macro average.");
        Require(text.find("weighted avg") != std::string::npos, "Report must print weighted average.");

        RequireThrows<std::invalid_argument>(
            []
            {
                RandomForestResult invalid;
                invalid.confusionMatrix = {{1, 0}, {0}};
                CoutCapture capture;
                eeg_to_hypnogram::PrintPythonStyleReport(invalid);
            },
            "Report must reject a non-square confusion matrix.");
    }

} // namespace

int main()
{
    try
    {
        Run("default configuration", TestDefaultConfiguration);
        Run("migrated temporal context API", TestTemporalContextRemainsAvailable);
        Run("classification metrics", TestClassificationMetrics);
        Run("empty and invalid metrics input", TestEmptyAndInvalidMetricsInput);
        Run("grid expansion order", TestGridExpansionOrder);
        Run("invalid grid configuration", TestInvalidGridConfiguration);
        Run("ranking rules", TestRankingRules);
        Run("grid execution and reproducibility", TestGridSearchExecutionAndReproducibility);
        Run("grid input validation", TestGridSearchInputValidation);
        Run("report output and validation", TestReportOutputAndValidation);

        std::cout << "Experiment Runner tests passed\n";
        std::cout << "cases=" << gPassed << '\n';
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Experiment Runner test failed: "
                  << error.what() << '\n';
        return 1;
    }
}
