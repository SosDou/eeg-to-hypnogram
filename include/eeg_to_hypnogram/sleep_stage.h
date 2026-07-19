#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace eeg_to_hypnogram
{
    // 项目内部统一睡眠阶段。
    enum class SleepStage
    {
        Wake,
        N1,
        N2,
        N3,
        Rem,
        Unknown,
        Movement
    };

    // 解析 Sleep-EDF annotation 文本或项目内部规范字符串。
    //
    // 无法识别的文本返回 SleepStage::Unknown。
    [[nodiscard]] SleepStage ParseSleepStage(
        std::string_view text);

    // 仅在文本明确表示睡眠阶段时返回 true。
    //
    // 普通 EDF annotation 不应被当成 UNKNOWN 睡眠阶段。
    [[nodiscard]] bool TryParseSleepStage(
        std::string_view text,
        SleepStage &stage);

    // 转换为项目当前对外使用的规范字符串：
    // W / N1 / N2 / N3 / REM / UNKNOWN / MOVEMENT。
    [[nodiscard]] std::string SleepStageToString(
        SleepStage stage);

    // 只有 W / N1 / N2 / N3 / REM 参与五分类训练。
    [[nodiscard]] bool IsTrainableSleepStage(
        SleepStage stage);

    // 五分类标签：
    // Wake -> 0, N1 -> 1, N2 -> 2, N3 -> 3, Rem -> 4。
    //
    // Unknown 和 Movement 会抛出 std::invalid_argument。
    [[nodiscard]] int SleepStageToClassLabel(
        SleepStage stage);

    // 将五分类标签转回睡眠阶段。
    //
    // [0, 4] 以外的标签会抛出 std::invalid_argument。
    [[nodiscard]] SleepStage ClassLabelToSleepStage(
        int label);

    // 从 Sleep-EDF annotation 解析出的睡眠阶段。
    struct SleepStageAnnotation
    {
        // W / N1 / N2 / N3 / REM / UNKNOWN / MOVEMENT。
        std::string stage;

        // annotation 相对于录制开始位置的精确时间。
        std::int64_t onsetTicks = 0;
        double onsetSeconds = 0.0;

        // 小于 0 表示 annotation 未提供有效持续时间。
        std::int64_t durationTicks = -1;
        double durationSeconds = -1.0;

        // EDF annotation 中的原始文本。
        std::string rawText;
    };

} // namespace eeg_to_hypnogram
