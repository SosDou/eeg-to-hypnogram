#define EEG_TO_HYPNOGRAM_RANDOM_FOREST_TESTING
#include "eeg_to_hypnogram/random_forest_baseline.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    using eeg_to_hypnogram::RandomForestConfig;
    using eeg_to_hypnogram::RandomForestModel;
    using eeg_to_hypnogram::RandomForestNodeData;
    using eeg_to_hypnogram::RandomForestResult;

    int gPassed = 0;

    void Require(bool condition, const std::string &message)
    {
        if (!condition)
        {
            throw std::runtime_error(message);
        }
    }

    void RequireNear(double actual,
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

    template <typename Exception, typename Function>
    void RequireThrows(Function &&function, const std::string &message)
    {
        bool threw = false;
        try
        {
            function();
        }
        catch (const Exception &)
        {
            threw = true;
        }
        Require(threw, message);
    }

    void Run(const std::string &name, const std::function<void()> &test)
    {
        test();
        ++gPassed;
        std::cout << "[PASS] " << name << '\n';
    }

    std::filesystem::path TempPath(const std::string &name)
    {
        return std::filesystem::temp_directory_path() /
               ("eeg_to_hypnogram_" + name);
    }

    std::vector<std::uint8_t> ReadAllBytes(const std::filesystem::path &path)
    {
        std::ifstream input(path, std::ios::binary);
        Require(static_cast<bool>(input), "failed to open binary file: " + path.string());
        return std::vector<std::uint8_t>(
            std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>());
    }

    template <typename T>
    T ReadPod(std::ifstream &input)
    {
        T value{};
        input.read(reinterpret_cast<char *>(&value), static_cast<std::streamsize>(sizeof(T)));
        Require(static_cast<bool>(input), "failed to parse model binary");
        return value;
    }

    struct ParsedModel
    {
        int numClasses = 0;
        int featureDim = 0;
        RandomForestConfig config;
        std::vector<std::vector<RandomForestNodeData>> trees;
    };

    ParsedModel ParseModel(const std::filesystem::path &path)
    {
        std::ifstream input(path, std::ios::binary);
        Require(static_cast<bool>(input), "failed to open model binary");

        char magic[4]{};
        input.read(magic, 4);
        Require(
            input && magic[0] == 'S' && magic[1] == 'R' &&
                magic[2] == 'F' && magic[3] == '1',
            "unexpected model magic");

        const std::uint32_t version = ReadPod<std::uint32_t>(input);
        Require(version == 1U, "unexpected model version");

        ParsedModel model;
        model.numClasses = ReadPod<int>(input);
        model.featureDim = ReadPod<int>(input);
        model.config.numTrees = ReadPod<int>(input);
        model.config.maxDepth = ReadPod<int>(input);
        model.config.minSamplesSplit = ReadPod<int>(input);
        model.config.maxFeaturesPerSplit = ReadPod<int>(input);
        model.config.maxThresholdCandidates = ReadPod<int>(input);
        model.config.seed = ReadPod<unsigned int>(input);

        const std::int32_t treeCount = ReadPod<std::int32_t>(input);
        model.trees.reserve(static_cast<std::size_t>(treeCount));

        for (int treeIndex = 0; treeIndex < treeCount; ++treeIndex)
        {
            const std::int32_t nodeCount = ReadPod<std::int32_t>(input);
            std::vector<RandomForestNodeData> tree;
            tree.reserve(static_cast<std::size_t>(nodeCount));

            for (int nodeIndex = 0; nodeIndex < nodeCount; ++nodeIndex)
            {
                RandomForestNodeData node;
                node.isLeaf = ReadPod<std::uint8_t>(input) != 0U;
                node.predictedClass = ReadPod<int>(input);
                node.featureIndex = ReadPod<int>(input);
                node.threshold = ReadPod<double>(input);
                node.left = ReadPod<int>(input);
                node.right = ReadPod<int>(input);
                tree.push_back(node);
            }
            model.trees.push_back(std::move(tree));
        }

        return model;
    }

    template <typename T>
    void WritePod(std::ofstream &output, const T &value)
    {
        output.write(
            reinterpret_cast<const char *>(&value),
            static_cast<std::streamsize>(sizeof(T)));
        Require(static_cast<bool>(output), "failed to write test model binary");
    }

    void WriteVotingTieModel(const std::filesystem::path &path)
    {
        std::ofstream output(path, std::ios::binary);
        Require(static_cast<bool>(output), "failed to create voting tie model");

        const char magic[4] = {'S', 'R', 'F', '1'};
        output.write(magic, 4);
        const std::uint32_t version = 1;
        WritePod(output, version);

        const int numClasses = 5;
        const int featureDim = 1;
        RandomForestConfig config;
        config.numTrees = 2;
        WritePod(output, numClasses);
        WritePod(output, featureDim);
        WritePod(output, config.numTrees);
        WritePod(output, config.maxDepth);
        WritePod(output, config.minSamplesSplit);
        WritePod(output, config.maxFeaturesPerSplit);
        WritePod(output, config.maxThresholdCandidates);
        WritePod(output, config.seed);

        const std::int32_t treeCount = 2;
        WritePod(output, treeCount);

        for (int predictedClass : {1, 3})
        {
            const std::int32_t nodeCount = 1;
            WritePod(output, nodeCount);
            const std::uint8_t isLeaf = 1U;
            const int featureIndex = -1;
            const double threshold = 0.0;
            const int left = -1;
            const int right = -1;
            WritePod(output, isLeaf);
            WritePod(output, predictedClass);
            WritePod(output, featureIndex);
            WritePod(output, threshold);
            WritePod(output, left);
            WritePod(output, right);
        }
    }

    int TreeDepth(const std::vector<RandomForestNodeData> &tree,
                  int nodeIndex = 0)
    {
        const auto &node = tree.at(static_cast<std::size_t>(nodeIndex));
        if (node.isLeaf)
        {
            return 0;
        }
        return 1 + std::max(
                       TreeDepth(tree, node.left),
                       TreeDepth(tree, node.right));
    }

    std::pair<std::vector<std::vector<double>>, std::vector<int>>
    MakeBinaryData()
    {
        std::vector<std::vector<double>> X;
        std::vector<int> y;
        for (int i = 0; i < 40; ++i)
        {
            X.push_back({0.01 * i, 0.1 * (i % 3)});
            y.push_back(0);
        }
        for (int i = 0; i < 40; ++i)
        {
            X.push_back({10.0 + 0.01 * i, 1.0 + 0.1 * (i % 3)});
            y.push_back(1);
        }
        return {X, y};
    }

    std::pair<std::vector<std::vector<double>>, std::vector<int>>
    MakeFiveClassData(int samplesPerClass)
    {
        std::vector<std::vector<double>> X;
        std::vector<int> y;
        for (int label = 0; label < 5; ++label)
        {
            for (int i = 0; i < samplesPerClass; ++i)
            {
                X.push_back({10.0 * label + 0.01 * i,
                             static_cast<double>(label),
                             static_cast<double>((label + i) % 3)});
                y.push_back(label);
            }
        }
        return {X, y};
    }

    RandomForestConfig FastConfig()
    {
        RandomForestConfig config;
        config.numTrees = 31;
        config.maxDepth = 8;
        config.minSamplesSplit = 2;
        config.maxFeaturesPerSplit = 3;
        config.maxThresholdCandidates = 128;
        config.seed = 42;
        return config;
    }

} // namespace

int main()
{
    try
    {
        Run("default configuration", []
            {
            const RandomForestConfig config;
            Require(config.numTrees == 120, "default numTrees changed");
            Require(config.maxDepth == 12, "default maxDepth changed");
            Require(config.minSamplesSplit == 4, "default minSamplesSplit changed");
            Require(config.maxFeaturesPerSplit == 0, "default maxFeaturesPerSplit changed");
            Require(config.maxThresholdCandidates == 24, "default maxThresholdCandidates changed");
            Require(config.seed == 42U, "default seed changed"); });

        Run("gini and majority tie rules", []
            {
            using namespace eeg_to_hypnogram::random_forest_testing;
            RequireNear(GiniImpurity({0, 0, 1, 1}, 2), 0.5, 1e-12, "gini mismatch");
            Require(MajorityClass({3, 1, 3, 1}, 5) == 1,
                    "majority tie must select smaller class id"); });

        Run("empty training set", []
            {
            RandomForestModel model;
            RequireThrows<std::invalid_argument>(
                [&] { model.Train({}, {}, 5); },
                "empty training set must throw"); });

        Run("X/y size mismatch", []
            {
            RandomForestModel model;
            RequireThrows<std::invalid_argument>(
                [&] { model.Train({{1.0}}, {}, 5); },
                "X/y size mismatch must throw"); });

        Run("inconsistent feature dimensions", []
            {
            RandomForestModel model;
            RequireThrows<std::invalid_argument>(
                [&] { model.Train({{1.0}, {2.0, 3.0}}, {0, 1}, 5); },
                "inconsistent feature dimensions must throw"); });

        Run("invalid labels", []
            {
            RandomForestModel model;
            RequireThrows<std::invalid_argument>(
                [&] { model.Train({{1.0}, {2.0}}, {0, -1}, 5); },
                "negative label must throw");
            RequireThrows<std::invalid_argument>(
                [&] { model.Train({{1.0}, {2.0}}, {0, 5}, 5); },
                "label >= numClasses must throw"); });

        Run("invalid configuration", []
            {
            const auto [X, y] = MakeBinaryData();
            RandomForestModel model;
            RandomForestConfig config = FastConfig();

            config.numTrees = 0;
            RequireThrows<std::invalid_argument>([&] { model.Train(X, y, 2, config); }, "zero trees must throw");
            config = FastConfig();
            config.maxDepth = -1;
            RequireThrows<std::invalid_argument>([&] { model.Train(X, y, 2, config); }, "negative max depth must throw");
            config = FastConfig();
            config.minSamplesSplit = 0;
            RequireThrows<std::invalid_argument>([&] { model.Train(X, y, 2, config); }, "zero min split must throw");
            config = FastConfig();
            config.maxFeaturesPerSplit = -1;
            RequireThrows<std::invalid_argument>([&] { model.Train(X, y, 2, config); }, "negative feature subset must throw");
            config = FastConfig();
            config.maxThresholdCandidates = 0;
            RequireThrows<std::invalid_argument>([&] { model.Train(X, y, 2, config); }, "zero threshold candidates must throw"); });

        Run("single-class training set", []
            {
            std::vector<std::vector<double>> X;
            std::vector<int> y;
            for (int i = 0; i < 30; ++i)
            {
                X.push_back({static_cast<double>(i), static_cast<double>(i % 2)});
                y.push_back(3);
            }

            RandomForestConfig config = FastConfig();
            config.numTrees = 7;
            RandomForestModel model;
            model.Train(X, y, 5, config);
            Require(model.PredictOne({123.0, -8.0}) == 3,
                    "single-class model must predict the only observed class"); });

        Run("simple separable binary data", []
            {
            const auto [X, y] = MakeBinaryData();
            RandomForestModel model;
            model.Train(X, y, 2, FastConfig());
            Require(model.PredictOne({0.2, 0.0}) == 0, "binary class 0 prediction failed");
            Require(model.PredictOne({10.2, 1.0}) == 1, "binary class 1 prediction failed"); });

        Run("separable five-class data", []
            {
            const auto [X, y] = MakeFiveClassData(30);
            RandomForestModel model;
            model.Train(X, y, 5, FastConfig());
            for (int label = 0; label < 5; ++label)
            {
                const int prediction = model.PredictOne({
                    10.0 * label + 0.15,
                    static_cast<double>(label),
                    static_cast<double>(label % 3)});
                Require(prediction == label, "five-class prediction mismatch");
            } });

        Run("prediction count and label range", []
            {
            const auto [X, y] = MakeFiveClassData(20);
            RandomForestModel model;
            model.Train(X, y, 5, FastConfig());
            const auto predictions = model.PredictBatch(X);
            Require(predictions.size() == X.size(), "prediction row count mismatch");
            for (int prediction : predictions)
            {
                Require(prediction >= 0 && prediction <= 4,
                        "prediction outside five-class range");
            } });

        Run("fixed seed reproducibility", []
            {
            const auto [X, y] = MakeFiveClassData(12);
            RandomForestConfig config = FastConfig();
            config.numTrees = 9;
            config.maxThresholdCandidates = 4;

            RandomForestModel first;
            RandomForestModel second;
            first.Train(X, y, 5, config);
            second.Train(X, y, 5, config);

            const auto firstPath = TempPath("same_seed_first.srf");
            const auto secondPath = TempPath("same_seed_second.srf");
            first.SaveBinary(firstPath.string());
            second.SaveBinary(secondPath.string());
            Require(ReadAllBytes(firstPath) == ReadAllBytes(secondPath),
                    "same seed must produce byte-identical models");
            std::filesystem::remove(firstPath);
            std::filesystem::remove(secondPath); });

        Run("different seeds change model", []
            {
            const auto [X, y] = MakeFiveClassData(12);
            RandomForestConfig firstConfig = FastConfig();
            firstConfig.numTrees = 9;
            firstConfig.maxFeaturesPerSplit = 1;
            firstConfig.maxThresholdCandidates = 3;
            RandomForestConfig secondConfig = firstConfig;
            secondConfig.seed = firstConfig.seed + 1U;

            RandomForestModel first;
            RandomForestModel second;
            first.Train(X, y, 5, firstConfig);
            second.Train(X, y, 5, secondConfig);

            const auto firstPath = TempPath("different_seed_first.srf");
            const auto secondPath = TempPath("different_seed_second.srf");
            first.SaveBinary(firstPath.string());
            second.SaveBinary(secondPath.string());
            Require(ReadAllBytes(firstPath) != ReadAllBytes(secondPath),
                    "different seeds should change sampled forest structure");
            std::filesystem::remove(firstPath);
            std::filesystem::remove(secondPath); });

        Run("bootstrap sample count and replacement", []
            {
            using eeg_to_hypnogram::random_forest_testing::GenerateBootstrapIndices;
            const auto bootstrap = GenerateBootstrapIndices(32, 42U);
            Require(bootstrap.size() == 32U,
                    "bootstrap sample count must equal training sample count");
            for (int index : bootstrap)
            {
                Require(index >= 0 && index < 32, "bootstrap index out of range");
            }
            const std::set<int> unique(bootstrap.begin(), bootstrap.end());
            Require(unique.size() < bootstrap.size(),
                    "bootstrap must sample with replacement"); });

        Run("feature subset count", []
            {
            using eeg_to_hypnogram::random_forest_testing::ResolveFeaturesToTry;
            Require(ResolveFeaturesToTry(175, 0) == 13,
                    "default feature subset must use floor(sqrt(featureDim))");
            Require(ResolveFeaturesToTry(10, 3) == 3,
                    "explicit feature subset count changed");
            Require(ResolveFeaturesToTry(10, 100) == 10,
                    "feature subset must clamp to feature dimension"); });

        Run("maximum depth limit", []
            {
            const auto [X, y] = MakeBinaryData();
            RandomForestConfig config = FastConfig();
            config.numTrees = 3;
            config.maxDepth = 1;

            RandomForestModel model;
            model.Train(X, y, 2, config);
            const auto path = TempPath("max_depth.srf");
            model.SaveBinary(path.string());
            const ParsedModel parsed = ParseModel(path);
            for (const auto &tree : parsed.trees)
            {
                Require(TreeDepth(tree) <= 1, "tree exceeded configured maximum depth");
            }
            std::filesystem::remove(path); });

        Run("minimum split sample limit", []
            {
            const auto [X, y] = MakeBinaryData();
            RandomForestConfig config = FastConfig();
            config.numTrees = 3;
            config.minSamplesSplit = static_cast<int>(X.size()) + 1;

            RandomForestModel model;
            model.Train(X, y, 2, config);
            const auto path = TempPath("min_split.srf");
            model.SaveBinary(path.string());
            const ParsedModel parsed = ParseModel(path);
            for (const auto &tree : parsed.trees)
            {
                Require(tree.size() == 1U && tree.front().isLeaf,
                        "root must remain a leaf below minSamplesSplit");
            }
            std::filesystem::remove(path); });

        Run("forest voting tie rule", []
            {
            const auto path = TempPath("voting_tie.srf");
            WriteVotingTieModel(path);
            RandomForestModel model;
            model.LoadBinary(path.string());
            Require(model.PredictOne({0.0}) == 1,
                    "vote tie must select smaller class id");
            std::filesystem::remove(path); });

        Run("prediction feature mismatch", []
            {
            const auto [X, y] = MakeBinaryData();
            RandomForestModel model;
            model.Train(X, y, 2, FastConfig());
            RequireThrows<std::invalid_argument>(
                [&] { model.PredictOne({1.0}); },
                "prediction dimension mismatch must throw"); });

        Run("accuracy and five-class confusion matrix", []
            {
            const auto [trainX, trainY] = MakeFiveClassData(30);
            std::vector<std::vector<double>> testX;
            std::vector<int> testY;
            for (int label = 0; label < 5; ++label)
            {
                for (int i = 0; i < 3; ++i)
                {
                    testX.push_back({
                        10.0 * label + 0.05 * i,
                        static_cast<double>(label),
                        static_cast<double>((label + i) % 3)});
                    testY.push_back(label);
                }
            }

            const RandomForestResult result =
                eeg_to_hypnogram::TrainEvaluateRandomForestFixedSplit(
                    trainX, trainY, testX, testY, 5, FastConfig());

            RequireNear(result.trainAccuracy, 1.0, 1e-12, "train accuracy mismatch");
            RequireNear(result.testAccuracy, 1.0, 1e-12, "test accuracy mismatch");
            Require(result.confusionMatrix.size() == 5U,
                    "confusion matrix row count mismatch");
            for (int row = 0; row < 5; ++row)
            {
                Require(result.confusionMatrix[static_cast<std::size_t>(row)].size() == 5U,
                        "confusion matrix column count mismatch");
                for (int column = 0; column < 5; ++column)
                {
                    const int expected = row == column ? 3 : 0;
                    Require(
                        result.confusionMatrix[static_cast<std::size_t>(row)]
                                              [static_cast<std::size_t>(column)] == expected,
                        "confusion matrix content mismatch");
                }
            } });

        Run("retraining resets forest", []
            {
            const auto [binaryX, binaryY] = MakeBinaryData();
            const auto [fiveX, fiveY] = MakeFiveClassData(12);

            RandomForestConfig firstConfig = FastConfig();
            firstConfig.numTrees = 7;
            RandomForestConfig secondConfig = FastConfig();
            secondConfig.numTrees = 3;

            RandomForestModel model;
            model.Train(binaryX, binaryY, 2, firstConfig);
            Require(model.NumTrees() == 7, "first training tree count mismatch");
            model.Train(fiveX, fiveY, 5, secondConfig);
            Require(model.NumTrees() == 3, "retraining must replace old trees");
            Require(model.NumClasses() == 5, "retraining must replace class count");
            Require(model.FeatureDim() == 3, "retraining must replace feature dimension"); });

        Run("prediction before training", []
            {
            RandomForestModel model;
            RequireThrows<std::runtime_error>(
                [&] { model.PredictOne({1.0}); },
                "PredictOne before training must throw");
            RequireThrows<std::runtime_error>(
                [&] { model.PredictBatch({}); },
                "PredictBatch before training must throw even for an empty batch"); });

        Run("training does not modify input", []
            {
            auto [X, y] = MakeFiveClassData(10);
            const auto originalX = X;
            const auto originalY = y;
            RandomForestModel model;
            model.Train(X, y, 5, FastConfig());
            Require(X == originalX, "Train modified X");
            Require(y == originalY, "Train modified y"); });

        Run("binary save/load compatibility", []
            {
            const auto [X, y] = MakeFiveClassData(15);
            RandomForestConfig config = FastConfig();
            config.numTrees = 11;
            config.seed = 1234U;

            RandomForestModel trained;
            trained.Train(X, y, 5, config);
            const auto before = trained.PredictBatch(X);

            const auto path = TempPath("round_trip.srf");
            trained.SaveBinary(path.string());

            RandomForestModel loaded;
            loaded.LoadBinary(path.string());
            const auto after = loaded.PredictBatch(X);

            Require(before == after, "predictions changed after binary round trip");
            Require(loaded.NumClasses() == 5, "loaded class count mismatch");
            Require(loaded.FeatureDim() == 3, "loaded feature dimension mismatch");
            Require(loaded.NumTrees() == 11, "loaded tree count mismatch");
            Require(loaded.Config().seed == 1234U, "loaded config mismatch");
            std::filesystem::remove(path); });

        Run("fixed-split validation", []
            {
            const auto [X, y] = MakeBinaryData();
            RequireThrows<std::invalid_argument>(
                [&] {
                    eeg_to_hypnogram::TrainEvaluateRandomForestFixedSplit(
                        X, y, {{1.0}}, {0}, 2, FastConfig());
                },
                "train/test feature dimension mismatch must throw");

            auto invalidTestY = y;
            invalidTestY.front() = 2;
            RequireThrows<std::invalid_argument>(
                [&] {
                    eeg_to_hypnogram::TrainEvaluateRandomForestFixedSplit(
                        X, y, X, invalidTestY, 2, FastConfig());
                },
                "invalid evaluation label must throw"); });

        std::cout << "Random Forest Baseline tests passed\n";
        std::cout << "cases=" << gPassed << '\n';
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Random Forest Baseline test failed: " << error.what() << '\n';
        return 1;
    }
}
