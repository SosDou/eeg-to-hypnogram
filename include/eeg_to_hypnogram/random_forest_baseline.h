#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace eeg_to_hypnogram
{

    // 随机森林训练超参数。
    struct RandomForestConfig
    {
        // 森林中的决策树数量。
        int numTrees = 120;

        // 单棵树的最大深度；根节点深度为 0。
        int maxDepth = 12;

        // 一个节点继续分裂所需的最小样本数。
        int minSamplesSplit = 4;

        // 每次分裂可用的最大特征数；0 表示 sqrt(num_features)。
        int maxFeaturesPerSplit = 0;

        // 每个特征用于尝试分裂阈值的最大候选数。
        int maxThresholdCandidates = 24;

        // 随机种子，用于可复现训练。
        unsigned int seed = 42;
    };

    // 随机森林训练和评估结果。
    struct RandomForestResult
    {
        // 训练集准确率。
        double trainAccuracy = 0.0;

        // 测试集准确率。
        double testAccuracy = 0.0;

        // 混淆矩阵：行是真实类别，列是预测类别。
        std::vector<std::vector<int>> confusionMatrix;
    };

    // 以数组形式保存的单个决策树节点。
    struct RandomForestNodeData
    {
        // 是否为叶子节点。
        bool isLeaf = true;

        // 叶子节点预测类别；非叶子节点中保留当前节点多数类。
        int predictedClass = 0;

        // 分裂使用的特征下标，仅非叶子节点有效。
        int featureIndex = -1;

        // 分裂阈值，仅非叶子节点有效。
        double threshold = 0.0;

        // 左子节点在树数组中的索引。
        int left = -1;

        // 右子节点在树数组中的索引。
        int right = -1;
    };

    class RandomForestModel
    {
    public:
        // 使用给定 X/y 训练随机森林。重复调用会替换旧模型。
        void Train(
            const std::vector<std::vector<double>> &X,
            const std::vector<int> &y,
            int numClasses,
            const RandomForestConfig &config = RandomForestConfig());

        // 对单个样本进行预测。
        int PredictOne(const std::vector<double> &x) const;

        // 对一批样本进行预测。
        std::vector<int> PredictBatch(const std::vector<std::vector<double>> &X) const;

        // 以旧项目兼容的 SRF1 二进制格式保存模型。
        void SaveBinary(const std::string &modelPath) const;

        // 从旧项目兼容的 SRF1 二进制格式加载模型。
        void LoadBinary(const std::string &modelPath);

        int NumClasses() const { return numClasses_; }
        int FeatureDim() const { return featureDim_; }
        int NumTrees() const { return static_cast<int>(trees_.size()); }
        const RandomForestConfig &Config() const { return config_; }

    private:
        int numClasses_ = 0;
        int featureDim_ = 0;
        RandomForestConfig config_{};
        std::vector<std::vector<RandomForestNodeData>> trees_;
    };

    // 旧项目兼容的行级随机划分评估辅助函数。
    // 完整数据集发现、受试者级划分和 train/validation/test 编排不属于本模块。
    RandomForestResult TrainEvaluateRandomForest(
        const std::vector<std::vector<double>> &X,
        const std::vector<int> &y,
        int numClasses,
        double trainRatio,
        const RandomForestConfig &config = RandomForestConfig());

    // 使用给定的固定训练集与测试集训练并评估随机森林。
    RandomForestResult TrainEvaluateRandomForestFixedSplit(
        const std::vector<std::vector<double>> &XTrain,
        const std::vector<int> &yTrain,
        const std::vector<std::vector<double>> &XTest,
        const std::vector<int> &yTest,
        int numClasses,
        const RandomForestConfig &config = RandomForestConfig());

#ifdef EEG_TO_HYPNOGRAM_RANDOM_FOREST_TESTING
    // 仅供单元测试验证旧算法内部规则，不属于正式公开 API。
    namespace random_forest_testing
    {
        double GiniImpurity(const std::vector<int> &labels, int numClasses);
        int MajorityClass(const std::vector<int> &labels, int numClasses);
        std::vector<int> GenerateBootstrapIndices(std::size_t sampleCount, unsigned int seed);
        int ResolveFeaturesToTry(int featureDim, int maxFeaturesPerSplit);
    } // namespace random_forest_testing
#endif

} // namespace eeg_to_hypnogram
