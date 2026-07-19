#pragma once

#include "eeg_to_hypnogram/random_forest_baseline.h"
#include "eeg_to_hypnogram/temporal_context.h"

#include <vector>

namespace eeg_to_hypnogram
{

    // 随机森林小规模网格搜索配置。
    struct GridSearchConfig
    {
        // 候选树数量集合；遍历顺序与容器顺序一致。
        std::vector<int> numTreesSet = {60, 120};

        // 候选最大深度集合；遍历顺序与容器顺序一致。
        std::vector<int> maxDepthSet = {10, 14};

        // 候选最小分裂样本数集合；遍历顺序与容器顺序一致。
        std::vector<int> minSamplesSplitSet = {2, 6};

        // 每个特征用于寻找分裂阈值的最大候选数量。
        int maxThresholdCandidates = 24;

        // 每次分裂时可考虑的最大特征数；0 表示随机森林默认策略。
        int maxFeaturesPerSplit = 0;

        // 所有网格试验共同使用的随机种子。
        unsigned int seed = 42;

        // 控制终端排名输出的最大结果数量，不改变返回结果数量。
        int rankingTopK = 5;
    };

    // 旧项目实验配置容器。
    // 当前 Experiment Runner 公开执行接口仍直接接收已构建好的 X/y。
    struct ExperimentConfig
    {
        // 分类类别数；睡眠分期通常为 5 类。
        int numClasses = 5;

        // 时序上下文配置。具体特征拼接由 temporal_context 模块实现。
        TemporalContextConfig temporalContext;

        // 随机森林网格搜索配置。
        GridSearchConfig gridSearch;
    };

    // 从测试集混淆矩阵派生的汇总指标。
    struct ClassificationMetrics
    {
        double accuracy = 0.0;
        double macroF1 = 0.0;
        double weightedF1 = 0.0;
    };

    // 单个随机森林参数组合的训练、评估和派生指标。
    struct GridSearchTrial
    {
        RandomForestConfig config;
        RandomForestResult result;
        ClassificationMetrics testMetrics;
    };

    // 从方形混淆矩阵计算准确率、macro F1 和 weighted F1。
    // 行是真实类别，列是预测类别。
    ClassificationMetrics ComputeClassificationMetrics(
        const std::vector<std::vector<int>> &confusionMatrix);

    // 对已明确划分的训练集和测试集执行旧项目兼容的小规模网格搜索。
    // 返回结果按 macro F1 降序、accuracy 降序、trainAccuracy 升序排列。
    std::vector<GridSearchTrial> RunSmallGridSearch(
        const std::vector<std::vector<double>> &XTrain,
        const std::vector<int> &yTrain,
        const std::vector<std::vector<double>> &XTest,
        const std::vector<int> &yTest,
        int numClasses,
        const GridSearchConfig &config);

    // 以旧项目接近 Python/sklearn 的文本格式打印分类报告。
    void PrintPythonStyleReport(const RandomForestResult &rf);

#if defined(EEG_TO_HYPNOGRAM_EXPERIMENT_RUNNER_TESTING)
    namespace experiment_runner_testing
    {

        // 仅供单元测试锁定旧指标公式。
        ClassificationMetrics ComputeClassificationMetrics(
            const std::vector<std::vector<int>> &confusionMatrix);

        // 仅供单元测试锁定旧网格展开顺序。
        std::vector<RandomForestConfig> BuildSmallGrid(
            const GridSearchConfig &config);

        // 仅供单元测试锁定旧排名比较规则。
        bool IsBetterTrial(
            const GridSearchTrial &left,
            const GridSearchTrial &right);

    } // namespace experiment_runner_testing
#endif

} // namespace eeg_to_hypnogram
