#include "eeg_to_hypnogram/experiment_runner.h"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

    std::string LabelIdToName(int label)
    {
        switch (label)
        {
        case 0:
            return "W";
        case 1:
            return "N1";
        case 2:
            return "N2";
        case 3:
            return "N3";
        case 4:
            return "REM";
        default:
            return "UNKNOWN";
        }
    }

    void ValidateConfusionMatrix(
        const std::vector<std::vector<int>> &confusionMatrix,
        bool allowEmpty)
    {
        if (confusionMatrix.empty())
        {
            if (allowEmpty)
            {
                return;
            }
            throw std::invalid_argument("Confusion matrix must not be empty.");
        }

        const std::size_t numClasses = confusionMatrix.size();
        for (const auto &row : confusionMatrix)
        {
            if (row.size() != numClasses)
            {
                throw std::invalid_argument("Confusion matrix must be square.");
            }
            for (int value : row)
            {
                if (value < 0)
                {
                    throw std::invalid_argument("Confusion matrix counts must be >= 0.");
                }
            }
        }
    }

    void ValidateGridSearchConfig(
        const eeg_to_hypnogram::GridSearchConfig &config)
    {
        if (config.numTreesSet.empty() ||
            config.maxDepthSet.empty() ||
            config.minSamplesSplitSet.empty())
        {
            throw std::invalid_argument("Grid search ranges must not be empty.");
        }

        for (int numTrees : config.numTreesSet)
        {
            if (numTrees <= 0)
            {
                throw std::invalid_argument("Every numTrees candidate must be > 0.");
            }
        }
        for (int maxDepth : config.maxDepthSet)
        {
            if (maxDepth < 0)
            {
                throw std::invalid_argument("Every maxDepth candidate must be >= 0.");
            }
        }
        for (int minSamplesSplit : config.minSamplesSplitSet)
        {
            if (minSamplesSplit <= 0)
            {
                throw std::invalid_argument("Every minSamplesSplit candidate must be > 0.");
            }
        }
        if (config.maxThresholdCandidates <= 0)
        {
            throw std::invalid_argument("maxThresholdCandidates must be > 0.");
        }
        if (config.maxFeaturesPerSplit < 0)
        {
            throw std::invalid_argument("maxFeaturesPerSplit must be >= 0.");
        }
    }

    eeg_to_hypnogram::ClassificationMetrics ComputeClassificationMetricsImpl(
        const std::vector<std::vector<int>> &confusionMatrix)
    {
        ValidateConfusionMatrix(confusionMatrix, true);

        const int numClasses = static_cast<int>(confusionMatrix.size());
        if (numClasses <= 0)
        {
            return {};
        }

        int total = 0;
        int correct = 0;
        double macroF1 = 0.0;
        double weightedF1 = 0.0;

        for (int classIndex = 0; classIndex < numClasses; ++classIndex)
        {
            const int truePositive =
                confusionMatrix[static_cast<std::size_t>(classIndex)]
                               [static_cast<std::size_t>(classIndex)];

            int rowSum = 0;
            int columnSum = 0;
            for (int other = 0; other < numClasses; ++other)
            {
                rowSum +=
                    confusionMatrix[static_cast<std::size_t>(classIndex)]
                                   [static_cast<std::size_t>(other)];
                columnSum +=
                    confusionMatrix[static_cast<std::size_t>(other)]
                                   [static_cast<std::size_t>(classIndex)];
            }

            const double precision =
                columnSum > 0
                    ? static_cast<double>(truePositive) /
                          static_cast<double>(columnSum)
                    : 0.0;
            const double recall =
                rowSum > 0
                    ? static_cast<double>(truePositive) /
                          static_cast<double>(rowSum)
                    : 0.0;
            const double f1 =
                precision + recall > 0.0
                    ? (2.0 * precision * recall) /
                          (precision + recall)
                    : 0.0;

            macroF1 += f1;
            weightedF1 += f1 * static_cast<double>(rowSum);
            total += rowSum;
            correct += truePositive;
        }

        eeg_to_hypnogram::ClassificationMetrics metrics;
        metrics.accuracy =
            total > 0
                ? static_cast<double>(correct) /
                      static_cast<double>(total)
                : 0.0;
        metrics.macroF1 =
            macroF1 / static_cast<double>(numClasses);
        metrics.weightedF1 =
            total > 0
                ? weightedF1 / static_cast<double>(total)
                : 0.0;
        return metrics;
    }

    std::vector<eeg_to_hypnogram::RandomForestConfig> BuildSmallGridImpl(
        const eeg_to_hypnogram::GridSearchConfig &config)
    {
        ValidateGridSearchConfig(config);

        std::vector<eeg_to_hypnogram::RandomForestConfig> grid;
        grid.reserve(
            config.numTreesSet.size() *
            config.maxDepthSet.size() *
            config.minSamplesSplitSet.size());

        // 旧循环顺序：numTrees -> maxDepth -> minSamplesSplit。
        for (int numTrees : config.numTreesSet)
        {
            for (int maxDepth : config.maxDepthSet)
            {
                for (int minSamplesSplit : config.minSamplesSplitSet)
                {
                    eeg_to_hypnogram::RandomForestConfig forestConfig;
                    forestConfig.numTrees = numTrees;
                    forestConfig.maxDepth = maxDepth;
                    forestConfig.minSamplesSplit = minSamplesSplit;
                    forestConfig.maxThresholdCandidates =
                        config.maxThresholdCandidates;
                    forestConfig.maxFeaturesPerSplit =
                        config.maxFeaturesPerSplit;
                    forestConfig.seed = config.seed;
                    grid.push_back(forestConfig);
                }
            }
        }

        return grid;
    }

    bool IsBetterTrialImpl(
        const eeg_to_hypnogram::GridSearchTrial &left,
        const eeg_to_hypnogram::GridSearchTrial &right)
    {
        if (left.testMetrics.macroF1 != right.testMetrics.macroF1)
        {
            return left.testMetrics.macroF1 > right.testMetrics.macroF1;
        }
        if (left.testMetrics.accuracy != right.testMetrics.accuracy)
        {
            return left.testMetrics.accuracy > right.testMetrics.accuracy;
        }
        return left.result.trainAccuracy < right.result.trainAccuracy;
    }

} // namespace

namespace eeg_to_hypnogram
{

    ClassificationMetrics ComputeClassificationMetrics(
        const std::vector<std::vector<int>> &confusionMatrix)
    {
        return ComputeClassificationMetricsImpl(confusionMatrix);
    }

    void PrintPythonStyleReport(const RandomForestResult &rf)
    {
        const auto &confusionMatrix = rf.confusionMatrix;
        ValidateConfusionMatrix(confusionMatrix, false);

        const int numClasses = static_cast<int>(confusionMatrix.size());
        int total = 0;
        int correct = 0;

        for (int row = 0; row < numClasses; ++row)
        {
            for (int column = 0; column < numClasses; ++column)
            {
                total +=
                    confusionMatrix[static_cast<std::size_t>(row)]
                                   [static_cast<std::size_t>(column)];
            }
            correct +=
                confusionMatrix[static_cast<std::size_t>(row)]
                               [static_cast<std::size_t>(row)];
        }

        const double accuracy =
            total > 0
                ? static_cast<double>(correct) /
                      static_cast<double>(total)
                : 0.0;

        std::cout << "准确率: " << accuracy << std::endl;

        std::cout << "混淆矩阵" << std::endl;
        std::cout << "[";
        for (int row = 0; row < numClasses; ++row)
        {
            std::cout << "[";
            for (int column = 0; column < numClasses; ++column)
            {
                std::cout
                    << confusionMatrix[static_cast<std::size_t>(row)]
                                      [static_cast<std::size_t>(column)]
                    << (column + 1 == numClasses ? "" : " ");
            }
            std::cout << "]"
                      << (row + 1 == numClasses ? "" : "\n ");
        }
        std::cout << "]" << std::endl;

        std::cout << "\nclassification report" << std::endl;
        std::cout
            << "class      precision    recall  f1-score   support"
            << std::endl;

        double macroPrecision = 0.0;
        double macroRecall = 0.0;
        double macroF1 = 0.0;
        double weightedPrecision = 0.0;
        double weightedRecall = 0.0;
        double weightedF1 = 0.0;

        for (int classIndex = 0; classIndex < numClasses; ++classIndex)
        {
            const int truePositive =
                confusionMatrix[static_cast<std::size_t>(classIndex)]
                               [static_cast<std::size_t>(classIndex)];

            int rowSum = 0;
            int columnSum = 0;
            for (int other = 0; other < numClasses; ++other)
            {
                rowSum +=
                    confusionMatrix[static_cast<std::size_t>(classIndex)]
                                   [static_cast<std::size_t>(other)];
                columnSum +=
                    confusionMatrix[static_cast<std::size_t>(other)]
                                   [static_cast<std::size_t>(classIndex)];
            }

            const double precision =
                columnSum > 0
                    ? static_cast<double>(truePositive) /
                          static_cast<double>(columnSum)
                    : 0.0;
            const double recall =
                rowSum > 0
                    ? static_cast<double>(truePositive) /
                          static_cast<double>(rowSum)
                    : 0.0;
            const double f1 =
                precision + recall > 0.0
                    ? (2.0 * precision * recall) /
                          (precision + recall)
                    : 0.0;

            std::cout
                << LabelIdToName(classIndex) << "\t"
                << precision << "\t"
                << recall << "\t"
                << f1 << "\t"
                << rowSum << std::endl;

            macroPrecision += precision;
            macroRecall += recall;
            macroF1 += f1;
            weightedPrecision +=
                precision * static_cast<double>(rowSum);
            weightedRecall +=
                recall * static_cast<double>(rowSum);
            weightedF1 +=
                f1 * static_cast<double>(rowSum);
        }

        const double classDenominator = static_cast<double>(numClasses);
        const double macroAveragePrecision =
            macroPrecision / classDenominator;
        const double macroAverageRecall =
            macroRecall / classDenominator;
        const double macroAverageF1 =
            macroF1 / classDenominator;

        // 与旧实现一致：空样本时加权指标分母使用 1。
        const int sampleDenominator = std::max(1, total);
        const double weightedAveragePrecision =
            weightedPrecision / static_cast<double>(sampleDenominator);
        const double weightedAverageRecall =
            weightedRecall / static_cast<double>(sampleDenominator);
        const double weightedAverageF1 =
            weightedF1 / static_cast<double>(sampleDenominator);

        std::cout
            << "\naccuracy\t\t\t" << accuracy
            << "\t" << total << std::endl;
        std::cout
            << "macro avg\t" << macroAveragePrecision
            << "\t" << macroAverageRecall
            << "\t" << macroAverageF1
            << "\t" << total << std::endl;
        std::cout
            << "weighted avg\t" << weightedAveragePrecision
            << "\t" << weightedAverageRecall
            << "\t" << weightedAverageF1
            << "\t" << total << std::endl;
    }

    std::vector<GridSearchTrial> RunSmallGridSearch(
        const std::vector<std::vector<double>> &XTrain,
        const std::vector<int> &yTrain,
        const std::vector<std::vector<double>> &XTest,
        const std::vector<int> &yTest,
        int numClasses,
        const GridSearchConfig &config)
    {
        const auto grid = BuildSmallGridImpl(config);

        std::vector<GridSearchTrial> trials;
        trials.reserve(grid.size());

        std::cout << "\n===== Small Grid Search =====" << std::endl;
        std::cout << "Total trials: " << grid.size() << std::endl;

        for (std::size_t trialIndex = 0;
             trialIndex < grid.size();
             ++trialIndex)
        {
            const auto &forestConfig = grid[trialIndex];

            std::cout
                << "[Trial " << (trialIndex + 1)
                << "/" << grid.size() << "] "
                << "numTrees=" << forestConfig.numTrees << ", "
                << "maxDepth=" << forestConfig.maxDepth << ", "
                << "minSamplesSplit="
                << forestConfig.minSamplesSplit << std::endl;

            const auto result =
                TrainEvaluateRandomForestFixedSplit(
                    XTrain,
                    yTrain,
                    XTest,
                    yTest,
                    numClasses,
                    forestConfig);

            const auto metrics =
                ComputeClassificationMetrics(
                    result.confusionMatrix);

            std::cout
                << "  -> test_acc=" << metrics.accuracy
                << ", macro_f1=" << metrics.macroF1
                << ", weighted_f1=" << metrics.weightedF1
                << std::endl;

            trials.push_back(
                GridSearchTrial{
                    forestConfig,
                    result,
                    metrics,
                });
        }

        // 旧排序规则：macro F1 降序 -> accuracy 降序 -> trainAccuracy 升序。
        // 三项完全相同时没有额外配置平局规则，继续使用 std::sort 的旧行为。
        std::sort(
            trials.begin(),
            trials.end(),
            IsBetterTrialImpl);

        // 保留旧输出标题；rankingTopK 只控制实际打印数量。
        std::cout << "\n===== Grid Ranking (Top 5) =====" << std::endl;

        const std::size_t topK =
            std::min<std::size_t>(
                static_cast<std::size_t>(
                    std::max(0, config.rankingTopK)),
                trials.size());

        for (std::size_t rank = 0; rank < topK; ++rank)
        {
            const auto &trial = trials[rank];
            std::cout
                << "#" << (rank + 1)
                << " numTrees=" << trial.config.numTrees
                << ", maxDepth=" << trial.config.maxDepth
                << ", minSamplesSplit="
                << trial.config.minSamplesSplit
                << " | test_acc=" << trial.testMetrics.accuracy
                << ", macro_f1=" << trial.testMetrics.macroF1
                << ", weighted_f1="
                << trial.testMetrics.weightedF1
                << std::endl;
        }

        return trials;
    }

    namespace experiment_runner_testing
    {

        ClassificationMetrics ComputeClassificationMetrics(
            const std::vector<std::vector<int>> &confusionMatrix)
        {
            return ComputeClassificationMetricsImpl(confusionMatrix);
        }

        std::vector<RandomForestConfig> BuildSmallGrid(
            const GridSearchConfig &config)
        {
            return BuildSmallGridImpl(config);
        }

        bool IsBetterTrial(
            const GridSearchTrial &left,
            const GridSearchTrial &right)
        {
            return IsBetterTrialImpl(left, right);
        }

    } // namespace experiment_runner_testing

} // namespace eeg_to_hypnogram
