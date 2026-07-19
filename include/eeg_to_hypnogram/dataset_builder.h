#pragma once

#include "eeg_to_hypnogram/dataset_manifest.h"
#include "eeg_to_hypnogram/temporal_context.h"

#include <string>
#include <vector>

namespace eeg_to_hypnogram
{

    struct DatasetFilePair
    {
        // 多导睡眠监测信号（PSG）文件路径。
        std::string psgPath;

        // 睡眠分期标注（Hypnogram）文件路径。
        std::string hypPath;
    };

    struct FeaturePipelineSummary
    {
        // 每个 epoch 的固定时长（秒）。
        double epochSeconds = 30.0;

        // 特征提取前统一重采样到的目标采样率（Hz）。
        double targetSampleRateHz = 100.0;

        // 实际参与特征计算的通道数量。
        int channelCount = 0;

        // 单个 epoch 的基础特征维度，不包含时间上下文。
        int baseFeatureDim = 0;

        // 拼接时间上下文后的最终特征维度。
        int temporalFeatureDim = 0;

        // 参与计算的原始 EDF 通道标签，顺序与特征通道顺序一致。
        std::vector<std::string> channelLabels;

        // 各通道在原始 EDF 中的采样率，顺序与 channelLabels 一致。
        std::vector<double> channelSampleRatesHz;
    };

    // 从单个 PSG 文件构建无标签特征。
    // 结果覆盖写入 XOut，不追加到 XOut 原有内容。
    void BuildFeaturesFromPsgFile(
        const std::string &psgPath,
        const TemporalContextConfig &contextConfig,
        std::vector<std::vector<double>> *XOut,
        FeaturePipelineSummary *summaryOut);

    // 按 pairs 输入顺序追加构建有标签数据集。
    // 每个 PSG 文件独立构建时间上下文，不跨文件拼接。
    void AppendDatasetFromPairs(
        const std::vector<DatasetFilePair> &pairs,
        const std::string &splitName,
        const TemporalContextConfig &contextConfig,
        std::vector<std::vector<double>> *XOut,
        std::vector<int> *yOut,
        FeaturePipelineSummary *summaryOut = nullptr);

    // 从 DatasetManifest 中的完整 PSG/Hypnogram pairs 构建有标签数据集。
    // 这个 Manifest 入口只负责清单校验和路径适配，实际构建仍复用
    // 由 AppendDatasetFromPairs 负责。
    void AppendDatasetFromManifest(
        const DatasetManifest &manifest,
        const std::string &splitName,
        const TemporalContextConfig &contextConfig,
        std::vector<std::vector<double>> *XOut,
        std::vector<int> *yOut,
        FeaturePipelineSummary *summaryOut = nullptr);

#if defined(EEG_TO_HYPNOGRAM_DATASET_BUILDER_TESTING)
    namespace dataset_builder_testing
    {

        // 仅在 BUILD_TESTING 构建中可见，用于锁定旧标签映射。
        int StageToLabelId(const std::string &stage);

        // 仅在 BUILD_TESTING 构建中可见，用于锁定旧线性重采样行为。
        std::vector<double> ResampleLinear(
            const std::vector<double> &samples,
            double sourceSampleRateHz,
            double targetSampleRateHz,
            int targetSampleCount);

    } // namespace dataset_builder_testing
#endif

} // namespace eeg_to_hypnogram
