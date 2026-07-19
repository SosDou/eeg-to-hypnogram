#pragma once

#include "eeg_to_hypnogram/dataset_builder.h"
#include "eeg_to_hypnogram/dataset_manifest.h"
#include "eeg_to_hypnogram/dataset_split.h"
#include "eeg_to_hypnogram/experiment_runner.h"
#include "eeg_to_hypnogram/random_forest_baseline.h"

#include <cstddef>
#include <string>
#include <vector>

namespace eeg_to_hypnogram
{

    struct TrainingPipelineConfig
    {
        // 外层受试者级 train/test 划分。
        DatasetSplitConfig splitConfig;

        // 仅在外层训练受试者内部执行的内层受试者级 fitting/validation 划分。
        // 其中 selectionSplit.test 是传给 ExperimentRunner 排名的验证集。
        DatasetSplitConfig selectionSplitConfig;

        // 现有实验配置：类别数、时序上下文和随机森林网格搜索。
        ExperimentConfig experimentConfig;

        // 最终 SRF1 模型输出路径。
        std::string modelOutputPath =
            "models/eeg_to_hypnogram_random_forest.srf1";

        // 人类可读的训练报告输出路径。
        std::string reportOutputPath =
            "models/training_report.txt";
    };

    struct ExperimentResult
    {
        // 用于超参数选择的内层划分。test 成员是验证数据，不是独立的外层测试集。
        DatasetManifestSplit selectionSplit;

        std::size_t fittingSampleCount = 0;
        std::size_t validationSampleCount = 0;

        // 按 ExperimentRunner 的排名规则从优到劣排序。
        std::vector<GridSearchTrial> trials;

        RandomForestConfig bestConfig;
    };

    struct TrainingPipelineResult
    {
        DatasetManifestSplit manifestSplit;

        FeaturePipelineSummary trainDatasetSummary;
        FeaturePipelineSummary testDatasetSummary;

        std::size_t trainSampleCount = 0;
        std::size_t testSampleCount = 0;
        std::size_t featureCount = 0;

        ExperimentResult experimentResult;

        // 在全部外层训练样本上训练、并在独立外层测试集上评估的最终模型。
        RandomForestResult finalEvaluation;
        ClassificationMetrics testMetrics;
    };

    TrainingPipelineResult RunTrainingPipeline(
        const DatasetManifest &manifest,
        const TrainingPipelineConfig &config);

} // namespace eeg_to_hypnogram
