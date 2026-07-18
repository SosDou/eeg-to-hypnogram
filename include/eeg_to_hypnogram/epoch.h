#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "eeg_to_hypnogram/sleep_stage.h"

namespace eeg_to_hypnogram
{

    // Epoch 构建配置。
    struct EpochBuildConfig
    {
        // 每个 epoch 的持续时间，默认 30 秒。
        double epochSeconds = 30.0;

        // MNE 风格裁剪中，在开始锚点之前保留的时间。
        double cropBeforeSeconds = 30.0 * 60.0;

        // MNE 风格裁剪中，在结束锚点之后保留的时间。
        double cropAfterSeconds = 30.0 * 60.0;

        // 是否丢弃 UNKNOWN 和 MOVEMENT 阶段。
        bool dropUnknownAndMovement = true;
    };

    // 一个已经分配睡眠阶段标签的 epoch 时间窗。
    //
    // 本结构只描述时间位置和标签，不包含实际 EEG 样本。
    // 后续 DatasetBuilder 应根据 startTicks 或 startSeconds 提取对应信号。
    struct SleepEpoch
    {
        // Epoch 索引。
        //
        // BuildSleepEpochs：
        //   表示相对于录制起点的时间槽位。
        //
        // BuildSleepEpochsMneStyle：
        //   表示过滤、排序后的连续结果索引。
        //
        // 因此下游不得直接使用 index 计算 EEG 样本位置，
        // 应使用 startTicks 或 startSeconds。
        std::int64_t index = 0;

        // Epoch 相对于录制起点的开始时间。
        std::int64_t startTicks = 0;
        double startSeconds = 0.0;

        // Epoch 的实际持续时间。
        //
        // 基础模式下，最后一个 epoch 可能不足 epochSeconds。
        // MNE 风格模式下，只生成完整 epoch。
        std::int64_t durationTicks = 0;
        double durationSeconds = 0.0;

        // W / N1 / N2 / N3 / REM / UNKNOWN / MOVEMENT。
        std::string stage;

        // annotation 原始文本。
        std::string rawText;
    };

    class EpochBuilder
    {
    public:
        // 按每条 annotation 的起点和持续时间直接切分。
        //
        // 一个 90 秒 annotation 在默认配置下会被拆成三个 30 秒 epoch。
        // 如果 annotation 最后一段不足一个完整 epoch，该部分仍会保留。
        //
        // 当 annotation 未提供有效 duration 时，回退为一个 epoch。
        static std::vector<SleepEpoch> BuildSleepEpochs(
            const std::vector<SleepStageAnnotation> &annotations,
            double epochSeconds = 30.0);

        // Sleep-EDF / 旧项目兼容的 MNE 风格构建方式：
        //
        // 1. 使用第二条 annotation 作为开始锚点；
        // 2. 使用倒数第二条 annotation 作为结束锚点；
        // 3. 在锚点前后按配置扩展裁剪范围；
        // 4. 将 annotation 与裁剪区间求交集；
        // 5. 只生成完整 epoch；
        // 6. 可过滤 UNKNOWN 和 MOVEMENT；
        // 7. 按开始时间排序并连续编号。
        static std::vector<SleepEpoch> BuildSleepEpochsMneStyle(
            const std::vector<SleepStageAnnotation> &annotations,
            double recordingDurationSeconds,
            const EpochBuildConfig &config = EpochBuildConfig());
    };

} // namespace eeg_to_hypnogram