#pragma once

#include <cstdint>
#include <string>

namespace eeg_to_hypnogram
{
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