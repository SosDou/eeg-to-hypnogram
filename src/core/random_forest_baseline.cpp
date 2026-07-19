#include "eeg_to_hypnogram/random_forest_baseline.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace
{

    struct Node
    {
        bool isLeaf = true;
        int predictedClass = 0;
        int featureIndex = -1;
        double threshold = 0.0;
        int left = -1;
        int right = -1;
    };

    void ValidateConfig(const eeg_to_hypnogram::RandomForestConfig &config)
    {
        if (config.numTrees <= 0)
        {
            throw std::invalid_argument("config.numTrees must be > 0.");
        }
        if (config.maxDepth < 0)
        {
            throw std::invalid_argument("config.maxDepth must be >= 0.");
        }
        if (config.minSamplesSplit <= 0)
        {
            throw std::invalid_argument("config.minSamplesSplit must be > 0.");
        }
        if (config.maxFeaturesPerSplit < 0)
        {
            throw std::invalid_argument("config.maxFeaturesPerSplit must be >= 0.");
        }
        if (config.maxThresholdCandidates <= 0)
        {
            throw std::invalid_argument("config.maxThresholdCandidates must be > 0.");
        }
    }

    int ValidateFeatureMatrix(const std::vector<std::vector<double>> &X,
                              const char *matrixName,
                              bool requireNonEmpty)
    {
        if (requireNonEmpty && X.empty())
        {
            throw std::invalid_argument(std::string(matrixName) + " must not be empty.");
        }
        if (X.empty())
        {
            return 0;
        }

        const int featureDim = static_cast<int>(X.front().size());
        if (featureDim <= 0)
        {
            throw std::invalid_argument(std::string(matrixName) + " feature dimension must be > 0.");
        }

        for (const auto &row : X)
        {
            if (static_cast<int>(row.size()) != featureDim)
            {
                throw std::invalid_argument(std::string("Inconsistent feature dimensions in ") + matrixName + ".");
            }
        }
        return featureDim;
    }

    void ValidateLabels(const std::vector<int> &y,
                        int numClasses,
                        const char *labelName)
    {
        for (int label : y)
        {
            if (label < 0 || label >= numClasses)
            {
                throw std::invalid_argument(std::string(labelName) + " contains a label outside [0, numClasses).");
            }
        }
    }

    double GiniImpurity(const std::vector<int> &indices,
                        const std::vector<int> &y,
                        int numClasses)
    {
        if (indices.empty())
        {
            return 0.0;
        }

        std::vector<int> counts(static_cast<std::size_t>(numClasses), 0);
        for (int idx : indices)
        {
            ++counts[static_cast<std::size_t>(y[static_cast<std::size_t>(idx)])];
        }

        double gini = 1.0;
        const double n = static_cast<double>(indices.size());
        for (int count : counts)
        {
            const double p = static_cast<double>(count) / n;
            gini -= p * p;
        }
        return gini;
    }

    int MajorityClass(const std::vector<int> &indices,
                      const std::vector<int> &y,
                      int numClasses)
    {
        std::vector<int> counts(static_cast<std::size_t>(numClasses), 0);
        for (int idx : indices)
        {
            ++counts[static_cast<std::size_t>(y[static_cast<std::size_t>(idx)])];
        }

        int bestClass = 0;
        int bestCount = -1;
        for (int c = 0; c < numClasses; ++c)
        {
            if (counts[static_cast<std::size_t>(c)] > bestCount)
            {
                bestCount = counts[static_cast<std::size_t>(c)];
                bestClass = c;
            }
        }
        return bestClass;
    }

    bool IsPure(const std::vector<int> &indices,
                const std::vector<int> &y)
    {
        if (indices.empty())
        {
            return true;
        }

        const int first = y[static_cast<std::size_t>(indices.front())];
        for (int idx : indices)
        {
            if (y[static_cast<std::size_t>(idx)] != first)
            {
                return false;
            }
        }
        return true;
    }

    int ResolveFeaturesToTry(int featureDim, int maxFeaturesPerSplit)
    {
        int featuresToTry = maxFeaturesPerSplit;
        if (featuresToTry <= 0)
        {
            featuresToTry = std::max(1, static_cast<int>(std::sqrt(static_cast<double>(featureDim))));
        }
        return std::min(featuresToTry, featureDim);
    }

    std::vector<int> GenerateBootstrapIndices(std::size_t sampleCount,
                                              std::mt19937 *rng)
    {
        if (sampleCount == 0)
        {
            return {};
        }

        std::uniform_int_distribution<int> pick(0, static_cast<int>(sampleCount) - 1);
        std::vector<int> bootstrap;
        bootstrap.reserve(sampleCount);
        for (std::size_t i = 0; i < sampleCount; ++i)
        {
            bootstrap.push_back(pick(*rng));
        }
        return bootstrap;
    }

    class DecisionTree
    {
    public:
        DecisionTree(int numClasses,
                     int maxDepth,
                     int minSamplesSplit,
                     int maxFeaturesPerSplit,
                     int maxThresholdCandidates,
                     std::mt19937 *rng)
            : numClasses_(numClasses),
              maxDepth_(maxDepth),
              minSamplesSplit_(minSamplesSplit),
              maxFeaturesPerSplit_(maxFeaturesPerSplit),
              maxThresholdCandidates_(maxThresholdCandidates),
              rng_(rng)
        {
        }

        void Fit(const std::vector<std::vector<double>> &X,
                 const std::vector<int> &y,
                 const std::vector<int> &bootstrapIndices)
        {
            nodes_.clear();
            BuildNode(X, y, bootstrapIndices, 0);
        }

        const std::vector<Node> &Nodes() const
        {
            return nodes_;
        }

    private:
        int BuildNode(const std::vector<std::vector<double>> &X,
                      const std::vector<int> &y,
                      const std::vector<int> &indices,
                      int depth)
        {
            Node node;
            node.predictedClass = MajorityClass(indices, y, numClasses_);

            const bool shouldStop =
                depth >= maxDepth_ ||
                static_cast<int>(indices.size()) < minSamplesSplit_ ||
                IsPure(indices, y);

            const int currentIdx = static_cast<int>(nodes_.size());
            nodes_.push_back(node);

            if (shouldStop)
            {
                return currentIdx;
            }

            const int numFeatures = static_cast<int>(X.front().size());
            const int featuresToTry = ResolveFeaturesToTry(numFeatures, maxFeaturesPerSplit_);

            std::vector<int> featureIds(static_cast<std::size_t>(numFeatures));
            std::iota(featureIds.begin(), featureIds.end(), 0);
            std::shuffle(featureIds.begin(), featureIds.end(), *rng_);
            featureIds.resize(static_cast<std::size_t>(featuresToTry));

            double bestScore = std::numeric_limits<double>::infinity();
            int bestFeature = -1;
            double bestThreshold = 0.0;
            std::vector<int> bestLeft;
            std::vector<int> bestRight;

            for (int feature : featureIds)
            {
                std::vector<double> candidates;
                candidates.reserve(static_cast<std::size_t>(
                    std::min(maxThresholdCandidates_, static_cast<int>(indices.size()))));

                if (static_cast<int>(indices.size()) <= maxThresholdCandidates_)
                {
                    for (int idx : indices)
                    {
                        candidates.push_back(
                            X[static_cast<std::size_t>(idx)][static_cast<std::size_t>(feature)]);
                    }
                }
                else
                {
                    std::uniform_int_distribution<int> pick(
                        0,
                        static_cast<int>(indices.size()) - 1);
                    for (int k = 0; k < maxThresholdCandidates_; ++k)
                    {
                        const int sampled = indices[static_cast<std::size_t>(pick(*rng_))];
                        candidates.push_back(
                            X[static_cast<std::size_t>(sampled)][static_cast<std::size_t>(feature)]);
                    }
                }

                std::sort(candidates.begin(), candidates.end());
                candidates.erase(
                    std::unique(candidates.begin(), candidates.end()),
                    candidates.end());

                for (double threshold : candidates)
                {
                    std::vector<int> left;
                    std::vector<int> right;
                    left.reserve(indices.size());
                    right.reserve(indices.size());

                    for (int idx : indices)
                    {
                        if (X[static_cast<std::size_t>(idx)][static_cast<std::size_t>(feature)] <= threshold)
                        {
                            left.push_back(idx);
                        }
                        else
                        {
                            right.push_back(idx);
                        }
                    }

                    if (left.empty() || right.empty())
                    {
                        continue;
                    }

                    const double n = static_cast<double>(indices.size());
                    const double score =
                        (static_cast<double>(left.size()) / n) * GiniImpurity(left, y, numClasses_) +
                        (static_cast<double>(right.size()) / n) * GiniImpurity(right, y, numClasses_);

                    // 旧行为使用严格小于；Gini 相同时保留更早遇到的候选。
                    if (score < bestScore)
                    {
                        bestScore = score;
                        bestFeature = feature;
                        bestThreshold = threshold;
                        bestLeft = std::move(left);
                        bestRight = std::move(right);
                    }
                }
            }

            if (bestFeature < 0)
            {
                return currentIdx;
            }

            nodes_[static_cast<std::size_t>(currentIdx)].isLeaf = false;
            nodes_[static_cast<std::size_t>(currentIdx)].featureIndex = bestFeature;
            nodes_[static_cast<std::size_t>(currentIdx)].threshold = bestThreshold;

            const int leftIdx = BuildNode(X, y, bestLeft, depth + 1);
            const int rightIdx = BuildNode(X, y, bestRight, depth + 1);

            nodes_[static_cast<std::size_t>(currentIdx)].left = leftIdx;
            nodes_[static_cast<std::size_t>(currentIdx)].right = rightIdx;
            return currentIdx;
        }

        int numClasses_;
        int maxDepth_;
        int minSamplesSplit_;
        int maxFeaturesPerSplit_;
        int maxThresholdCandidates_;
        std::mt19937 *rng_;
        std::vector<Node> nodes_;
    };

    template <typename T>
    void WritePod(std::ofstream *ofs, const T &value)
    {
        ofs->write(
            reinterpret_cast<const char *>(&value),
            static_cast<std::streamsize>(sizeof(T)));
        if (!(*ofs))
        {
            throw std::runtime_error("Failed while writing model binary.");
        }
    }

    template <typename T>
    void ReadPod(std::ifstream *ifs, T *value)
    {
        ifs->read(
            reinterpret_cast<char *>(value),
            static_cast<std::streamsize>(sizeof(T)));
        if (!(*ifs))
        {
            throw std::runtime_error("Failed while reading model binary.");
        }
    }

    double Accuracy(const std::vector<int> &yTrue,
                    const std::vector<int> &yPred)
    {
        if (yTrue.size() != yPred.size())
        {
            throw std::invalid_argument("Accuracy input size mismatch.");
        }

        int correct = 0;
        for (std::size_t i = 0; i < yTrue.size(); ++i)
        {
            if (yTrue[i] == yPred[i])
            {
                ++correct;
            }
        }

        return yTrue.empty()
                   ? 0.0
                   : static_cast<double>(correct) / static_cast<double>(yTrue.size());
    }

    void ValidateLoadedTree(const std::vector<eeg_to_hypnogram::RandomForestNodeData> &tree,
                            int numClasses,
                            int featureDim)
    {
        if (tree.empty())
        {
            throw std::runtime_error("Invalid empty tree in model file.");
        }

        for (const auto &node : tree)
        {
            if (node.predictedClass < 0 || node.predictedClass >= numClasses)
            {
                throw std::runtime_error("Invalid predicted class in model file.");
            }

            if (!node.isLeaf)
            {
                if (node.featureIndex < 0 || node.featureIndex >= featureDim)
                {
                    throw std::runtime_error("Invalid feature index in model file.");
                }
                if (node.left < 0 || node.left >= static_cast<int>(tree.size()) ||
                    node.right < 0 || node.right >= static_cast<int>(tree.size()))
                {
                    throw std::runtime_error("Invalid child index in model file.");
                }
            }
        }
    }

} // namespace

namespace eeg_to_hypnogram
{

    void RandomForestModel::Train(
        const std::vector<std::vector<double>> &X,
        const std::vector<int> &y,
        int numClasses,
        const RandomForestConfig &config)
    {
        if (X.empty() || y.empty() || X.size() != y.size())
        {
            throw std::invalid_argument("X/y is empty or size mismatch.");
        }
        if (numClasses <= 1)
        {
            throw std::invalid_argument("numClasses must be > 1.");
        }

        ValidateConfig(config);
        const int featureDim = ValidateFeatureMatrix(X, "X", true);
        ValidateLabels(y, numClasses, "y");

        // 旧行为：整个森林共用一个 mt19937；bootstrap、特征 shuffle 和阈值采样
        // 按实际执行顺序持续消耗同一随机数序列。
        std::mt19937 rng(config.seed);

        std::vector<std::vector<RandomForestNodeData>> trees;
        trees.reserve(static_cast<std::size_t>(config.numTrees));

        for (int treeIndex = 0; treeIndex < config.numTrees; ++treeIndex)
        {
            const std::vector<int> bootstrap = GenerateBootstrapIndices(X.size(), &rng);

            DecisionTree tree(
                numClasses,
                config.maxDepth,
                config.minSamplesSplit,
                config.maxFeaturesPerSplit,
                config.maxThresholdCandidates,
                &rng);
            tree.Fit(X, y, bootstrap);

            std::vector<RandomForestNodeData> outputTree;
            const auto &nodes = tree.Nodes();
            outputTree.reserve(nodes.size());
            for (const auto &node : nodes)
            {
                outputTree.push_back(RandomForestNodeData{
                    node.isLeaf,
                    node.predictedClass,
                    node.featureIndex,
                    node.threshold,
                    node.left,
                    node.right});
            }
            trees.push_back(std::move(outputTree));
        }

        // 成功构建完整森林后一次性替换旧模型，重复训练不会追加旧树。
        trees_ = std::move(trees);
        numClasses_ = numClasses;
        featureDim_ = featureDim;
        config_ = config;
    }

    int RandomForestModel::PredictOne(const std::vector<double> &x) const
    {
        if (trees_.empty())
        {
            throw std::runtime_error("Model is empty. Train or load a model first.");
        }
        if (static_cast<int>(x.size()) != featureDim_)
        {
            throw std::invalid_argument("Feature dimension mismatch in PredictOne.");
        }

        std::vector<int> votes(static_cast<std::size_t>(numClasses_), 0);

        for (const auto &tree : trees_)
        {
            if (tree.empty())
            {
                throw std::runtime_error("Corrupted model contains an empty tree.");
            }

            int nodeIndex = 0;
            while (true)
            {
                if (nodeIndex < 0 || nodeIndex >= static_cast<int>(tree.size()))
                {
                    throw std::runtime_error("Corrupted model tree structure while predicting.");
                }

                const auto &node = tree[static_cast<std::size_t>(nodeIndex)];
                if (node.isLeaf)
                {
                    if (node.predictedClass < 0 || node.predictedClass >= numClasses_)
                    {
                        throw std::runtime_error("Corrupted model contains an invalid predicted class.");
                    }
                    ++votes[static_cast<std::size_t>(node.predictedClass)];
                    break;
                }

                if (node.featureIndex < 0 || node.featureIndex >= featureDim_)
                {
                    throw std::runtime_error("Corrupted model contains an invalid feature index.");
                }

                nodeIndex =
                    (x[static_cast<std::size_t>(node.featureIndex)] <= node.threshold)
                        ? node.left
                        : node.right;
            }
        }

        int bestClass = 0;
        int bestVotes = -1;
        for (int c = 0; c < numClasses_; ++c)
        {
            // 旧行为使用严格大于；票数相同保留编号更小的类别。
            if (votes[static_cast<std::size_t>(c)] > bestVotes)
            {
                bestVotes = votes[static_cast<std::size_t>(c)];
                bestClass = c;
            }
        }
        return bestClass;
    }

    std::vector<int> RandomForestModel::PredictBatch(
        const std::vector<std::vector<double>> &X) const
    {
        if (trees_.empty())
        {
            throw std::runtime_error("Model is empty. Train or load a model first.");
        }

        std::vector<int> predictions;
        predictions.reserve(X.size());
        for (const auto &row : X)
        {
            predictions.push_back(PredictOne(row));
        }
        return predictions;
    }

    void RandomForestModel::SaveBinary(const std::string &modelPath) const
    {
        if (trees_.empty())
        {
            throw std::runtime_error("Model is empty. Nothing to save.");
        }

        std::ofstream ofs(modelPath, std::ios::binary);
        if (!ofs)
        {
            throw std::runtime_error("Cannot open model file for writing: " + modelPath);
        }

        const char magic[4] = {'S', 'R', 'F', '1'};
        ofs.write(magic, 4);
        if (!ofs)
        {
            throw std::runtime_error("Failed while writing model binary magic.");
        }

        const std::uint32_t version = 1;
        WritePod(&ofs, version);

        WritePod(&ofs, numClasses_);
        WritePod(&ofs, featureDim_);
        WritePod(&ofs, config_.numTrees);
        WritePod(&ofs, config_.maxDepth);
        WritePod(&ofs, config_.minSamplesSplit);
        WritePod(&ofs, config_.maxFeaturesPerSplit);
        WritePod(&ofs, config_.maxThresholdCandidates);
        WritePod(&ofs, config_.seed);

        const std::int32_t numTrees = static_cast<std::int32_t>(trees_.size());
        WritePod(&ofs, numTrees);

        for (const auto &tree : trees_)
        {
            const std::int32_t nodeCount = static_cast<std::int32_t>(tree.size());
            WritePod(&ofs, nodeCount);
            for (const auto &node : tree)
            {
                const std::uint8_t isLeaf = node.isLeaf ? 1U : 0U;
                WritePod(&ofs, isLeaf);
                WritePod(&ofs, node.predictedClass);
                WritePod(&ofs, node.featureIndex);
                WritePod(&ofs, node.threshold);
                WritePod(&ofs, node.left);
                WritePod(&ofs, node.right);
            }
        }
    }

    void RandomForestModel::LoadBinary(const std::string &modelPath)
    {
        std::ifstream ifs(modelPath, std::ios::binary);
        if (!ifs)
        {
            throw std::runtime_error("Cannot open model file for reading: " + modelPath);
        }

        char magic[4] = {0, 0, 0, 0};
        ifs.read(magic, 4);
        if (!ifs || magic[0] != 'S' || magic[1] != 'R' || magic[2] != 'F' || magic[3] != '1')
        {
            throw std::runtime_error("Invalid model binary magic.");
        }

        std::uint32_t version = 0;
        ReadPod(&ifs, &version);
        if (version != 1)
        {
            throw std::runtime_error("Unsupported model binary version.");
        }

        int loadedNumClasses = 0;
        int loadedFeatureDim = 0;
        RandomForestConfig loadedConfig;

        ReadPod(&ifs, &loadedNumClasses);
        ReadPod(&ifs, &loadedFeatureDim);
        ReadPod(&ifs, &loadedConfig.numTrees);
        ReadPod(&ifs, &loadedConfig.maxDepth);
        ReadPod(&ifs, &loadedConfig.minSamplesSplit);
        ReadPod(&ifs, &loadedConfig.maxFeaturesPerSplit);
        ReadPod(&ifs, &loadedConfig.maxThresholdCandidates);
        ReadPod(&ifs, &loadedConfig.seed);

        if (loadedNumClasses <= 1)
        {
            throw std::runtime_error("Invalid class count in model file.");
        }
        if (loadedFeatureDim <= 0)
        {
            throw std::runtime_error("Invalid feature dimension in model file.");
        }
        try
        {
            ValidateConfig(loadedConfig);
        }
        catch (const std::invalid_argument &error)
        {
            throw std::runtime_error(std::string("Invalid config in model file: ") + error.what());
        }

        std::int32_t numTrees = 0;
        ReadPod(&ifs, &numTrees);
        if (numTrees <= 0)
        {
            throw std::runtime_error("Invalid tree count in model file.");
        }
        if (numTrees != loadedConfig.numTrees)
        {
            throw std::runtime_error("Tree count does not match stored config.");
        }

        std::vector<std::vector<RandomForestNodeData>> loadedTrees;
        loadedTrees.reserve(static_cast<std::size_t>(numTrees));

        for (int treeIndex = 0; treeIndex < numTrees; ++treeIndex)
        {
            std::int32_t nodeCount = 0;
            ReadPod(&ifs, &nodeCount);
            if (nodeCount <= 0)
            {
                throw std::runtime_error("Invalid node count in model file.");
            }

            std::vector<RandomForestNodeData> tree;
            tree.reserve(static_cast<std::size_t>(nodeCount));

            for (int nodeIndex = 0; nodeIndex < nodeCount; ++nodeIndex)
            {
                std::uint8_t isLeaf = 0;
                RandomForestNodeData node;
                ReadPod(&ifs, &isLeaf);
                ReadPod(&ifs, &node.predictedClass);
                ReadPod(&ifs, &node.featureIndex);
                ReadPod(&ifs, &node.threshold);
                ReadPod(&ifs, &node.left);
                ReadPod(&ifs, &node.right);
                node.isLeaf = (isLeaf != 0);
                tree.push_back(node);
            }

            ValidateLoadedTree(tree, loadedNumClasses, loadedFeatureDim);
            loadedTrees.push_back(std::move(tree));
        }

        // 成功读取并验证完整模型后再替换当前状态。
        numClasses_ = loadedNumClasses;
        featureDim_ = loadedFeatureDim;
        config_ = loadedConfig;
        trees_ = std::move(loadedTrees);
    }

    RandomForestResult TrainEvaluateRandomForestFixedSplit(
        const std::vector<std::vector<double>> &XTrain,
        const std::vector<int> &yTrain,
        const std::vector<std::vector<double>> &XTest,
        const std::vector<int> &yTest,
        int numClasses,
        const RandomForestConfig &config)
    {
        if (XTrain.empty() || yTrain.empty() || XTrain.size() != yTrain.size())
        {
            throw std::invalid_argument("XTrain/yTrain is empty or size mismatch.");
        }
        if (XTest.empty() || yTest.empty() || XTest.size() != yTest.size())
        {
            throw std::invalid_argument("XTest/yTest is empty or size mismatch.");
        }
        if (numClasses <= 1)
        {
            throw std::invalid_argument("numClasses must be > 1.");
        }

        ValidateConfig(config);
        const int trainFeatureDim = ValidateFeatureMatrix(XTrain, "XTrain", true);
        const int testFeatureDim = ValidateFeatureMatrix(XTest, "XTest", true);
        if (trainFeatureDim != testFeatureDim)
        {
            throw std::invalid_argument("Train/test feature dimension mismatch.");
        }
        ValidateLabels(yTrain, numClasses, "yTrain");
        ValidateLabels(yTest, numClasses, "yTest");

        RandomForestModel model;
        model.Train(XTrain, yTrain, numClasses, config);

        const std::vector<int> predTrain = model.PredictBatch(XTrain);
        const std::vector<int> predTest = model.PredictBatch(XTest);

        RandomForestResult result;
        result.trainAccuracy = Accuracy(yTrain, predTrain);
        result.testAccuracy = Accuracy(yTest, predTest);
        result.confusionMatrix.assign(
            static_cast<std::size_t>(numClasses),
            std::vector<int>(static_cast<std::size_t>(numClasses), 0));

        for (std::size_t i = 0; i < yTest.size(); ++i)
        {
            const int actual = yTest[i];
            const int predicted = predTest[i];
            ++result.confusionMatrix[static_cast<std::size_t>(actual)]
                                    [static_cast<std::size_t>(predicted)];
        }

        return result;
    }

    RandomForestResult TrainEvaluateRandomForest(
        const std::vector<std::vector<double>> &X,
        const std::vector<int> &y,
        int numClasses,
        double trainRatio,
        const RandomForestConfig &config)
    {
        if (X.empty() || y.empty() || X.size() != y.size())
        {
            throw std::invalid_argument("X/y is empty or size mismatch.");
        }
        if (numClasses <= 1)
        {
            throw std::invalid_argument("numClasses must be > 1.");
        }
        if (trainRatio <= 0.0 || trainRatio >= 1.0)
        {
            throw std::invalid_argument("trainRatio must be in (0, 1).");
        }

        ValidateConfig(config);
        ValidateFeatureMatrix(X, "X", true);
        ValidateLabels(y, numClasses, "y");

        std::mt19937 rng(config.seed);
        std::vector<int> indices(X.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        const int trainSize = static_cast<int>(
            std::round(trainRatio * static_cast<double>(indices.size())));
        if (trainSize <= 0 || trainSize >= static_cast<int>(indices.size()))
        {
            throw std::invalid_argument("Invalid train split size computed from trainRatio.");
        }

        std::vector<std::vector<double>> XTrain;
        std::vector<int> yTrain;
        std::vector<std::vector<double>> XTest;
        std::vector<int> yTest;

        XTrain.reserve(static_cast<std::size_t>(trainSize));
        yTrain.reserve(static_cast<std::size_t>(trainSize));
        XTest.reserve(indices.size() - static_cast<std::size_t>(trainSize));
        yTest.reserve(indices.size() - static_cast<std::size_t>(trainSize));

        for (int i = 0; i < static_cast<int>(indices.size()); ++i)
        {
            const int index = indices[static_cast<std::size_t>(i)];
            if (i < trainSize)
            {
                XTrain.push_back(X[static_cast<std::size_t>(index)]);
                yTrain.push_back(y[static_cast<std::size_t>(index)]);
            }
            else
            {
                XTest.push_back(X[static_cast<std::size_t>(index)]);
                yTest.push_back(y[static_cast<std::size_t>(index)]);
            }
        }

        return TrainEvaluateRandomForestFixedSplit(
            XTrain,
            yTrain,
            XTest,
            yTest,
            numClasses,
            config);
    }

    namespace random_forest_testing
    {
        double GiniImpurity(const std::vector<int> &labels,
                            int numClasses)
        {
            if (numClasses <= 0)
            {
                throw std::invalid_argument("numClasses must be > 0.");
            }
            ValidateLabels(labels, numClasses, "labels");
            std::vector<int> indices(labels.size());
            std::iota(indices.begin(), indices.end(), 0);
            return ::GiniImpurity(indices, labels, numClasses);
        }

        int MajorityClass(const std::vector<int> &labels,
                          int numClasses)
        {
            if (labels.empty())
            {
                throw std::invalid_argument("labels must not be empty.");
            }
            if (numClasses <= 0)
            {
                throw std::invalid_argument("numClasses must be > 0.");
            }
            ValidateLabels(labels, numClasses, "labels");
            std::vector<int> indices(labels.size());
            std::iota(indices.begin(), indices.end(), 0);
            return ::MajorityClass(indices, labels, numClasses);
        }

        std::vector<int> GenerateBootstrapIndices(std::size_t sampleCount,
                                                  unsigned int seed)
        {
            std::mt19937 rng(seed);
            return ::GenerateBootstrapIndices(sampleCount, &rng);
        }

        int ResolveFeaturesToTry(int featureDim,
                                 int maxFeaturesPerSplit)
        {
            if (featureDim <= 0)
            {
                throw std::invalid_argument("featureDim must be > 0.");
            }
            if (maxFeaturesPerSplit < 0)
            {
                throw std::invalid_argument("maxFeaturesPerSplit must be >= 0.");
            }
            return ::ResolveFeaturesToTry(featureDim, maxFeaturesPerSplit);
        }
    } // namespace random_forest_testing

} // namespace eeg_to_hypnogram
