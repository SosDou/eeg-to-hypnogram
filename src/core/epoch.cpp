#include "eeg_to_hypnogram/epoch.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace eeg_to_hypnogram
{

    namespace
    {

        // EDFlib 使用 100 ns 作为一个时间 tick。
        // 1 秒 = 10,000,000 ticks。
        constexpr std::int64_t kTicksPerSecond = 10'000'000;

        // 将非负秒数转换为 ticks。
        std::int64_t SecondsToTicks(
            double seconds,
            const char *fieldName,
            bool allowZero)
        {
            if (!std::isfinite(seconds))
            {
                throw std::invalid_argument(
                    std::string(fieldName) + " must be finite.");
            }

            if (seconds < 0.0 || (!allowZero && seconds <= 0.0))
            {
                throw std::invalid_argument(
                    std::string(fieldName) +
                    (allowZero ? " must be >= 0." : " must be > 0."));
            }

            const long double scaled =
                static_cast<long double>(seconds) *
                static_cast<long double>(kTicksPerSecond);

            const long double maxTicks =
                static_cast<long double>(
                    std::numeric_limits<std::int64_t>::max());

            if (scaled > maxTicks)
            {
                throw std::overflow_error(
                    std::string(fieldName) + " is too large.");
            }

            const auto ticks =
                static_cast<std::int64_t>(std::llround(scaled));

            // 正数如果被舍入成了 0，说明 epoch 时间精度小于一个 tick。
            if (!allowZero && ticks <= 0)
            {
                throw std::invalid_argument(
                    std::string(fieldName) +
                    " is smaller than one time tick.");
            }

            return ticks;
        }

        // 将 ticks 转换为秒数。
        double TicksToSeconds(std::int64_t ticks)
        {
            return static_cast<double>(ticks) /
                   static_cast<double>(kTicksPerSecond);
        }

        // 安全执行两个非负 tick 值相加。
        std::int64_t CheckedAddTicks(
            std::int64_t left,
            std::int64_t right,
            const char *fieldName)
        {
            if (left < 0 || right < 0)
            {
                throw std::invalid_argument(
                    std::string(fieldName) +
                    " cannot contain negative ticks.");
            }

            if (left >
                std::numeric_limits<std::int64_t>::max() - right)
            {
                throw std::overflow_error(
                    std::string(fieldName) + " overflow.");
            }

            return left + right;
        }

        // 获取 annotation 的开始时间。
        //
        // 新 EdfReader 会同时填充 onsetTicks 和 onsetSeconds，
        // 因此正常情况下直接使用 onsetTicks。
        //
        // 为兼容手工构造测试数据或旧调用代码：
        // 当 onsetTicks 为 0、但 onsetSeconds 大于 0 时，
        // 从 onsetSeconds 回退转换。
        std::int64_t ResolveOnsetTicks(
            const SleepStageAnnotation &annotation)
        {
            if (annotation.onsetTicks < 0)
            {
                throw std::invalid_argument(
                    "annotation onsetTicks must be >= 0.");
            }

            if (annotation.onsetTicks != 0)
            {
                return annotation.onsetTicks;
            }

            if (!std::isfinite(annotation.onsetSeconds))
            {
                throw std::invalid_argument(
                    "annotation onsetSeconds must be finite.");
            }

            if (annotation.onsetSeconds < 0.0)
            {
                throw std::invalid_argument(
                    "annotation onsetSeconds must be >= 0.");
            }

            if (annotation.onsetSeconds > 0.0)
            {
                return SecondsToTicks(
                    annotation.onsetSeconds,
                    "annotation onsetSeconds",
                    true);
            }

            return 0;
        }

        // 获取 annotation 的持续时间。
        //
        // 优先级：
        // 1. durationTicks > 0；
        // 2. durationSeconds > 0；
        // 3. 回退为一个完整 epoch。
        //
        // 这保留了旧项目对缺失 duration 的处理行为。
        std::int64_t ResolveDurationTicks(
            const SleepStageAnnotation &annotation,
            std::int64_t epochTicks)
        {
            if (annotation.durationTicks > 0)
            {
                return annotation.durationTicks;
            }

            if (!std::isfinite(annotation.durationSeconds))
            {
                throw std::invalid_argument(
                    "annotation durationSeconds must be finite.");
            }

            if (annotation.durationSeconds > 0.0)
            {
                return SecondsToTicks(
                    annotation.durationSeconds,
                    "annotation durationSeconds",
                    false);
            }

            return epochTicks;
        }

        // 验证 annotation 按开始时间非递减排列。
        //
        // MNE 风格锚点依赖 annotation 顺序，因此不允许静默接受乱序输入。
        void ValidateAnnotationOrder(
            const std::vector<SleepStageAnnotation> &annotations)
        {
            if (annotations.empty())
            {
                return;
            }

            std::int64_t previousOnset =
                ResolveOnsetTicks(annotations.front());

            for (std::size_t index = 1;
                 index < annotations.size();
                 ++index)
            {
                const std::int64_t currentOnset =
                    ResolveOnsetTicks(annotations[index]);

                if (currentOnset < previousOnset)
                {
                    throw std::invalid_argument(
                        "annotations must be sorted by onset time.");
                }

                previousOnset = currentOnset;
            }
        }

        // 判断是否需要过滤当前阶段。
        bool ShouldSkipStage(
            const std::string &stage,
            bool dropUnknownAndMovement)
        {
            if (!dropUnknownAndMovement)
            {
                return false;
            }

            return stage == "UNKNOWN" ||
                   stage == "MOVEMENT";
        }

        // 构建一个 SleepEpoch 对象。
        SleepEpoch MakeEpoch(
            std::int64_t index,
            std::int64_t startTicks,
            std::int64_t durationTicks,
            const SleepStageAnnotation &annotation)
        {
            SleepEpoch epoch;

            epoch.index = index;

            epoch.startTicks = startTicks;
            epoch.startSeconds = TicksToSeconds(startTicks);

            epoch.durationTicks = durationTicks;
            epoch.durationSeconds =
                TicksToSeconds(durationTicks);

            epoch.stage = annotation.stage;
            epoch.rawText = annotation.rawText;

            return epoch;
        }

    } // namespace

    std::vector<SleepEpoch> EpochBuilder::BuildSleepEpochs(
        const std::vector<SleepStageAnnotation> &annotations,
        double epochSeconds)
    {
        const std::int64_t epochTicks =
            SecondsToTicks(
                epochSeconds,
                "epochSeconds",
                false);

        ValidateAnnotationOrder(annotations);

        std::vector<SleepEpoch> epochs;

        for (const auto &annotation : annotations)
        {
            const std::int64_t annotationStartTicks =
                ResolveOnsetTicks(annotation);

            const std::int64_t annotationDurationTicks =
                ResolveDurationTicks(annotation, epochTicks);

            const std::int64_t annotationEndTicks =
                CheckedAddTicks(
                    annotationStartTicks,
                    annotationDurationTicks,
                    "annotation end time");

            for (std::int64_t epochStartTicks =
                     annotationStartTicks;
                 epochStartTicks < annotationEndTicks;)
            {
                const std::int64_t remainingTicks =
                    annotationEndTicks - epochStartTicks;

                const std::int64_t currentDurationTicks =
                    std::min(epochTicks, remainingTicks);

                // 基础模式沿用旧项目语义：
                // index 表示相对于录制起点的时间槽位。
                const std::int64_t timelineIndex =
                    epochStartTicks / epochTicks;

                epochs.push_back(
                    MakeEpoch(
                        timelineIndex,
                        epochStartTicks,
                        currentDurationTicks,
                        annotation));

                // 避免在最后一个不完整 epoch 后继续循环。
                if (remainingTicks <= epochTicks)
                {
                    break;
                }

                epochStartTicks =
                    CheckedAddTicks(
                        epochStartTicks,
                        epochTicks,
                        "epoch start time");
            }
        }

        return epochs;
    }

    std::vector<SleepEpoch>
    EpochBuilder::BuildSleepEpochsMneStyle(
        const std::vector<SleepStageAnnotation> &annotations,
        double recordingDurationSeconds,
        const EpochBuildConfig &config)
    {
        const std::int64_t epochTicks =
            SecondsToTicks(
                config.epochSeconds,
                "config.epochSeconds",
                false);

        const std::int64_t recordingDurationTicks =
            SecondsToTicks(
                recordingDurationSeconds,
                "recordingDurationSeconds",
                false);

        const std::int64_t cropBeforeTicks =
            SecondsToTicks(
                config.cropBeforeSeconds,
                "config.cropBeforeSeconds",
                true);

        const std::int64_t cropAfterTicks =
            SecondsToTicks(
                config.cropAfterSeconds,
                "config.cropAfterSeconds",
                true);

        // 保留旧项目行为：少于三条 annotation 时无法选择锚点。
        if (annotations.size() < 3)
        {
            return {};
        }

        ValidateAnnotationOrder(annotations);

        const std::int64_t anchorStartTicks =
            ResolveOnsetTicks(annotations[1]);

        const std::int64_t anchorEndTicks =
            ResolveOnsetTicks(
                annotations[annotations.size() - 2]);

        const std::int64_t cropStartTicks =
            anchorStartTicks > cropBeforeTicks
                ? anchorStartTicks - cropBeforeTicks
                : 0;

        std::int64_t expandedCropEndTicks =
            std::numeric_limits<std::int64_t>::max();

        if (anchorEndTicks <=
            std::numeric_limits<std::int64_t>::max() -
                cropAfterTicks)
        {
            expandedCropEndTicks =
                anchorEndTicks + cropAfterTicks;
        }

        const std::int64_t cropEndTicks =
            std::min(
                recordingDurationTicks,
                expandedCropEndTicks);

        if (cropEndTicks <= cropStartTicks)
        {
            return {};
        }

        std::vector<SleepEpoch> epochs;

        for (const auto &annotation : annotations)
        {
            if (ShouldSkipStage(
                    annotation.stage,
                    config.dropUnknownAndMovement))
            {
                continue;
            }

            const std::int64_t annotationStartTicks =
                ResolveOnsetTicks(annotation);

            const std::int64_t annotationDurationTicks =
                ResolveDurationTicks(annotation, epochTicks);

            const std::int64_t annotationEndTicks =
                CheckedAddTicks(
                    annotationStartTicks,
                    annotationDurationTicks,
                    "annotation end time");

            const std::int64_t segmentStartTicks =
                std::max(
                    annotationStartTicks,
                    cropStartTicks);

            const std::int64_t segmentEndTicks =
                std::min(
                    annotationEndTicks,
                    cropEndTicks);

            if (segmentEndTicks <= segmentStartTicks)
            {
                continue;
            }

            // MNE 风格模式只生成完整 epoch。
            //
            // 使用整数 ticks 判断剩余长度，
            // 不需要旧代码中的 1e-9 浮点补偿。
            for (std::int64_t startTicks =
                     segmentStartTicks;
                 segmentEndTicks - startTicks >= epochTicks;)
            {
                // 这里的临时索引在排序后会被连续索引覆盖。
                const std::int64_t relativeIndex =
                    (startTicks - cropStartTicks) /
                    epochTicks;

                epochs.push_back(
                    MakeEpoch(
                        relativeIndex,
                        startTicks,
                        epochTicks,
                        annotation));

                if (segmentEndTicks - startTicks <
                    epochTicks * 2)
                {
                    break;
                }

                startTicks =
                    CheckedAddTicks(
                        startTicks,
                        epochTicks,
                        "epoch start time");
            }
        }

        // 使用 stable_sort，使相同开始时间的 epoch
        // 保持 annotation 输入中的相对顺序。
        std::stable_sort(
            epochs.begin(),
            epochs.end(),
            [](const SleepEpoch &left,
               const SleepEpoch &right)
            {
                return left.startTicks <
                       right.startTicks;
            });

        // MNE 风格输出使用过滤后的连续索引。
        for (std::size_t index = 0;
             index < epochs.size();
             ++index)
        {
            epochs[index].index =
                static_cast<std::int64_t>(index);
        }

        return epochs;
    }

} // namespace eeg_to_hypnogram