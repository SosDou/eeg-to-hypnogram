#include "eeg_to_hypnogram/edf_reader.h"
#include "eeg_to_hypnogram/epoch.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <limits>

namespace
{

    constexpr std::int64_t kTicksPerSecond = 10'000'000;

    std::int64_t ToTicks(double seconds)
    {
        return static_cast<std::int64_t>(
            std::llround(
                seconds *
                static_cast<double>(kTicksPerSecond)));
    }

    bool NearlyEqual(
        double left,
        double right,
        double tolerance = 1e-9)
    {
        return std::abs(left - right) <= tolerance;
    }

    void Require(
        bool condition,
        const std::string &message)
    {
        if (!condition)
        {
            throw std::runtime_error(message);
        }
    }

    template <typename Function>
    void RequireThrowsInvalidArgument(
        Function &&function,
        const std::string &message)
    {
        bool thrown = false;

        try
        {
            function();
        }
        catch (const std::invalid_argument &)
        {
            thrown = true;
        }

        Require(thrown, message);
    }

    eeg_to_hypnogram::SleepStageAnnotation MakeAnnotation(
        const std::string &stage,
        double onsetSeconds,
        double durationSeconds,
        const std::string &rawText = {})
    {
        eeg_to_hypnogram::SleepStageAnnotation annotation;

        annotation.stage = stage;

        annotation.onsetTicks =
            ToTicks(onsetSeconds);

        annotation.onsetSeconds =
            onsetSeconds;

        if (durationSeconds >= 0.0)
        {
            annotation.durationTicks =
                ToTicks(durationSeconds);
            annotation.durationSeconds =
                durationSeconds;
        }
        else
        {
            annotation.durationTicks = -1;
            annotation.durationSeconds = -1.0;
        }

        annotation.rawText = rawText;

        return annotation;
    }

    std::int64_t EpochStartSample(
        const eeg_to_hypnogram::SleepEpoch &epoch,
        double sampleRateHz)
    {
        const long double samplePosition =
            static_cast<long double>(epoch.startTicks) *
            static_cast<long double>(sampleRateHz) /
            static_cast<long double>(kTicksPerSecond);

        return static_cast<std::int64_t>(
            std::llround(samplePosition));
    }

    std::int64_t EpochSampleCount(
        const eeg_to_hypnogram::SleepEpoch &epoch,
        double sampleRateHz)
    {
        const long double sampleCount =
            static_cast<long double>(epoch.durationTicks) *
            static_cast<long double>(sampleRateHz) /
            static_cast<long double>(kTicksPerSecond);

        return static_cast<std::int64_t>(
            std::llround(sampleCount));
    }

    bool IsFilteredStage(const std::string &stage)
    {
        return stage == "UNKNOWN" ||
               stage == "MOVEMENT";
    }

    void TestBasicCompleteSplit()
    {
        const std::vector annotations{
            MakeAnnotation(
                "N2",
                0.0,
                90.0,
                "Sleep stage 2")};

        const auto epochs =
            eeg_to_hypnogram::EpochBuilder::
                BuildSleepEpochs(annotations);

        Require(
            epochs.size() == 3,
            "90-second annotation should create 3 epochs.");

        for (std::size_t index = 0;
             index < epochs.size();
             ++index)
        {
            const double expectedStart =
                static_cast<double>(index) * 30.0;

            Require(
                epochs[index].index ==
                    static_cast<std::int64_t>(index),
                "Basic epoch index is incorrect.");

            Require(
                NearlyEqual(
                    epochs[index].startSeconds,
                    expectedStart),
                "Basic epoch start time is incorrect.");

            Require(
                NearlyEqual(
                    epochs[index].durationSeconds,
                    30.0),
                "Basic epoch duration is incorrect.");

            Require(
                epochs[index].stage == "N2",
                "Basic epoch stage is incorrect.");

            Require(
                epochs[index].rawText ==
                    "Sleep stage 2",
                "Basic epoch raw text is incorrect.");
        }
    }

    void TestBasicPartialFinalEpoch()
    {
        const std::vector annotations{
            MakeAnnotation("N1", 0.0, 65.0)};

        const auto epochs =
            eeg_to_hypnogram::EpochBuilder::
                BuildSleepEpochs(annotations);

        Require(
            epochs.size() == 3,
            "65-second annotation should create 3 epochs.");

        Require(
            NearlyEqual(
                epochs[0].durationSeconds,
                30.0),
            "First epoch should be 30 seconds.");

        Require(
            NearlyEqual(
                epochs[1].durationSeconds,
                30.0),
            "Second epoch should be 30 seconds.");

        Require(
            NearlyEqual(
                epochs[2].durationSeconds,
                5.0),
            "Final partial epoch should be 5 seconds.");

        Require(
            epochs[2].durationTicks ==
                ToTicks(5.0),
            "Final partial epoch ticks are incorrect.");
    }

    void TestMissingDurationFallback()
    {
        const std::vector annotations{
            MakeAnnotation("REM", 60.0, -1.0)};

        const auto epochs =
            eeg_to_hypnogram::EpochBuilder::
                BuildSleepEpochs(annotations);

        Require(
            epochs.size() == 1,
            "Missing duration should create one epoch.");

        Require(
            NearlyEqual(
                epochs[0].durationSeconds,
                30.0),
            "Missing duration should fall back to 30 seconds.");

        Require(
            epochs[0].index == 2,
            "Epoch at 60 seconds should use timeline index 2.");
    }

    void TestSecondsFallback()
    {
        eeg_to_hypnogram::SleepStageAnnotation annotation;

        annotation.stage = "W";

        annotation.onsetTicks = 0;
        annotation.onsetSeconds = 30.0;

        annotation.durationTicks = -1;
        annotation.durationSeconds = 60.0;

        const auto epochs =
            eeg_to_hypnogram::EpochBuilder::
                BuildSleepEpochs({annotation});

        Require(
            epochs.size() == 2,
            "Seconds fallback should create 2 epochs.");

        Require(
            epochs[0].startTicks ==
                ToTicks(30.0),
            "onsetSeconds fallback is incorrect.");

        Require(
            epochs[1].startTicks ==
                ToTicks(60.0),
            "Second fallback epoch start is incorrect.");
    }

    void TestMneStyleFilteringAndContinuousIndex()
    {
        const std::vector annotations{
            MakeAnnotation("UNKNOWN", 0.0, 30.0),
            MakeAnnotation("W", 30.0, 60.0),
            MakeAnnotation("MOVEMENT", 90.0, 30.0),
            MakeAnnotation("N2", 120.0, 60.0),
            MakeAnnotation("N3", 180.0, 60.0),
            MakeAnnotation("UNKNOWN", 240.0, 30.0)};

        eeg_to_hypnogram::EpochBuildConfig config;

        config.epochSeconds = 30.0;
        config.cropBeforeSeconds = 0.0;
        config.cropAfterSeconds = 0.0;
        config.dropUnknownAndMovement = true;

        const auto epochs =
            eeg_to_hypnogram::EpochBuilder::
                BuildSleepEpochsMneStyle(
                    annotations,
                    300.0,
                    config);

        // 锚点：
        // 标注[1]      -> 30 秒
        // 标注[-2]     -> 180 秒
        //
        // 裁剪区间 = [30, 180)
        //
        // 清醒：30, 60
        // 运动阶段：被过滤
        // 第二阶段 N2：120, 150
        Require(
            epochs.size() == 4,
            "MNE-style build should create 4 epochs.");

        const std::vector<double> expectedStarts{
            30.0,
            60.0,
            120.0,
            150.0};

        for (std::size_t index = 0;
             index < epochs.size();
             ++index)
        {
            Require(
                epochs[index].index ==
                    static_cast<std::int64_t>(index),
                "MNE-style indexes must be continuous.");

            Require(
                NearlyEqual(
                    epochs[index].startSeconds,
                    expectedStarts[index]),
                "MNE-style epoch start is incorrect.");

            Require(
                NearlyEqual(
                    epochs[index].durationSeconds,
                    30.0),
                "MNE-style epochs must be complete.");
        }

        Require(
            epochs[0].stage == "W" &&
                epochs[1].stage == "W",
            "Wake epochs are incorrect.");

        Require(
            epochs[2].stage == "N2" &&
                epochs[3].stage == "N2",
            "N2 epochs are incorrect.");
    }

    void TestMneStyleKeepsInvalidStagesWhenConfigured()
    {
        const std::vector annotations{
            MakeAnnotation("UNKNOWN", 0.0, 30.0),
            MakeAnnotation("W", 30.0, 30.0),
            MakeAnnotation("MOVEMENT", 60.0, 30.0),
            MakeAnnotation("N2", 90.0, 30.0),
            MakeAnnotation("N3", 120.0, 30.0)};

        eeg_to_hypnogram::EpochBuildConfig config;

        config.cropBeforeSeconds = 0.0;
        config.cropAfterSeconds = 0.0;
        config.dropUnknownAndMovement = false;

        const auto epochs =
            eeg_to_hypnogram::EpochBuilder::
                BuildSleepEpochsMneStyle(
                    annotations,
                    180.0,
                    config);

        // 裁剪区间 = [30, 90)
        Require(
            epochs.size() == 2,
            "MNE-style build should keep MOVEMENT when filtering is disabled.");

        Require(
            epochs[0].stage == "W",
            "First unfiltered stage should be W.");

        Require(
            epochs[1].stage == "MOVEMENT",
            "Second unfiltered stage should be MOVEMENT.");
    }

    void TestMneStyleDropsPartialEpoch()
    {
        const std::vector annotations{
            MakeAnnotation("UNKNOWN", 0.0, 30.0),
            MakeAnnotation("N1", 30.0, 45.0),
            MakeAnnotation("N2", 75.0, 30.0),
            MakeAnnotation("N3", 105.0, 30.0)};

        eeg_to_hypnogram::EpochBuildConfig config;

        config.cropBeforeSeconds = 0.0;
        config.cropAfterSeconds = 0.0;

        const auto epochs =
            eeg_to_hypnogram::EpochBuilder::
                BuildSleepEpochsMneStyle(
                    annotations,
                    180.0,
                    config);

        // 锚点起点 = 30
        // 锚点终点 = 75
        // 第一阶段 N1 与裁剪区间的交集为 45 秒，只保留一个完整 30 秒时段。
        Require(
            epochs.size() == 1,
            "MNE-style build must drop partial epochs.");

        Require(
            NearlyEqual(
                epochs[0].startSeconds,
                30.0),
            "Partial epoch test start time is incorrect.");

        Require(
            NearlyEqual(
                epochs[0].durationSeconds,
                30.0),
            "Partial epoch test duration is incorrect.");
    }

    void TestTooFewMneAnnotations()
    {
        const std::vector annotations{
            MakeAnnotation("W", 0.0, 30.0),
            MakeAnnotation("N1", 30.0, 30.0)};

        const auto epochs =
            eeg_to_hypnogram::EpochBuilder::
                BuildSleepEpochsMneStyle(
                    annotations,
                    60.0);

        Require(
            epochs.empty(),
            "Fewer than 3 annotations should return an empty result.");
    }

    void TestInvalidArguments()
    {
        const std::vector annotations{
            MakeAnnotation("N2", 0.0, 30.0)};

        RequireThrowsInvalidArgument(
            [&annotations]()
            {
                eeg_to_hypnogram::EpochBuilder::
                    BuildSleepEpochs(
                        annotations,
                        0.0);
            },
            "Zero epoch duration must throw.");

        RequireThrowsInvalidArgument(
            [&annotations]()
            {
                eeg_to_hypnogram::EpochBuilder::
                    BuildSleepEpochs(
                        annotations,
                        -30.0);
            },
            "Negative epoch duration must throw.");

        RequireThrowsInvalidArgument(
            []()
            {
                eeg_to_hypnogram::EpochBuildConfig config;
                config.cropBeforeSeconds = -1.0;

                const std::vector mneAnnotations{
                    MakeAnnotation("W", 0.0, 30.0),
                    MakeAnnotation("N1", 30.0, 30.0),
                    MakeAnnotation("N2", 60.0, 30.0)};

                eeg_to_hypnogram::EpochBuilder::
                    BuildSleepEpochsMneStyle(
                        mneAnnotations,
                        90.0,
                        config);
            },
            "Negative cropBeforeSeconds must throw.");
    }

    void TestUnsortedAnnotations()
    {
        const std::vector annotations{
            MakeAnnotation("N2", 60.0, 30.0),
            MakeAnnotation("N1", 30.0, 30.0)};

        RequireThrowsInvalidArgument(
            [&annotations]()
            {
                eeg_to_hypnogram::EpochBuilder::
                    BuildSleepEpochs(annotations);
            },
            "Unsorted annotations must throw.");
    }

    void TestRealSleepEdfIntegrationIfConfigured()
    {
        const char *psgFilePath =
            std::getenv("EEG_TEST_EDF_FILE");

        const char *hypnogramFilePath =
            std::getenv("EEG_TEST_HYPNOGRAM_FILE");

        // 未配置真实数据路径时，只运行快速单元测试。
        if (psgFilePath == nullptr &&
            hypnogramFilePath == nullptr)
        {
            std::cout
                << "Sleep-EDF epoch integration test skipped: "
                << "environment variables are not configured.\n";

            return;
        }

        // 只配置一个路径通常表示 launch.json 或终端命令有误。
        Require(
            psgFilePath != nullptr &&
                hypnogramFilePath != nullptr,
            "Both EEG_TEST_EDF_FILE and "
            "EEG_TEST_HYPNOGRAM_FILE must be configured.");

        /*
         * 打开 PSG。
         *
         * PSG 文件不需要读取 annotation，因此传入 false，
         * 可以减少不必要的 annotation 解析。
         */
        eeg_to_hypnogram::EdfReader psgReader;
        psgReader.Open(psgFilePath, false);

        const auto &psgHeader =
            psgReader.Header();

        Require(
            psgHeader.fileDurationTicks > 0,
            "PSG file duration ticks must be positive.");

        Require(
            psgHeader.fileDurationSeconds > 0.0,
            "PSG file duration seconds must be positive.");

        Require(
            !psgHeader.signals.empty(),
            "PSG file must contain signals.");

        /*
         * 打开 Hypnogram。
         *
         * Hypnogram 必须在 Open 阶段加载 annotation。
         */
        eeg_to_hypnogram::EdfReader hypnogramReader;
        hypnogramReader.Open(hypnogramFilePath, true);

        Require(
            hypnogramReader.AnnotationsLoaded(),
            "Hypnogram annotations must be loaded.");

        const auto annotations =
            hypnogramReader.ReadSleepStageAnnotations();

        /*
         * 当前固定测试文件：
         * SC4001EC-Hypnogram.edf
         *
         * 这个断言用于确认测试数据没有被意外替换。
         */
        Require(
            annotations.size() == 154,
            "SC4001EC-Hypnogram.edf should contain "
            "154 sleep-stage annotations.");

        eeg_to_hypnogram::EpochBuildConfig config;

        config.epochSeconds = 30.0;
        config.cropBeforeSeconds = 30.0 * 60.0;
        config.cropAfterSeconds = 30.0 * 60.0;
        config.dropUnknownAndMovement = true;

        /*
         * 使用真实 annotation 构建 MNE 风格 epoch 时间线。
         */
        const auto epochs =
            eeg_to_hypnogram::EpochBuilder::
                BuildSleepEpochsMneStyle(
                    annotations,
                    psgHeader.fileDurationSeconds,
                    config);

        Require(
            !epochs.empty(),
            "Real Sleep-EDF epoch result must not be empty.");

        constexpr std::int64_t expectedEpochTicks =
            30 * kTicksPerSecond;

        /*
         * 验证全部真实 epoch 的通用不变量。
         */
        for (std::size_t index = 0;
             index < epochs.size();
             ++index)
        {
            const auto &epoch = epochs[index];

            Require(
                epoch.index ==
                    static_cast<std::int64_t>(index),
                "Real epoch indexes must be continuous.");

            Require(
                epoch.startTicks >= 0,
                "Real epoch start ticks must not be negative.");

            Require(
                epoch.durationTicks ==
                    expectedEpochTicks,
                "Every MNE-style real epoch must be 30 seconds.");

            Require(
                NearlyEqual(
                    epoch.durationSeconds,
                    30.0),
                "Every MNE-style real epoch must report "
                "30 duration seconds.");

            Require(
                !IsFilteredStage(epoch.stage),
                "Filtered real epochs must not contain "
                "UNKNOWN or MOVEMENT.");

            Require(
                epoch.startTicks <=
                    psgHeader.fileDurationTicks -
                        epoch.durationTicks,
                "Real epoch must remain inside PSG duration.");

            if (index > 0)
            {
                Require(
                    epochs[index - 1].startTicks <=
                        epoch.startTicks,
                    "Real epochs must be sorted by start time.");
            }
        }

        /*
         * 查找 Sleep-EDF 中使用的真实 EEG 通道。
         */
        const auto signalIndex =
            psgReader.FindSignalIndexByLabel(
                "EEG Fpz-Cz");

        Require(
            signalIndex.has_value(),
            "PSG file must contain EEG Fpz-Cz.");

        const auto &signal =
            psgHeader.signals.at(
                static_cast<std::size_t>(
                    signalIndex.value()));

        Require(
            signal.sampleRateHz > 0.0,
            "EEG Fpz-Cz sample rate must be positive.");

        Require(
            signal.sampleCount > 0,
            "EEG Fpz-Cz sample count must be positive.");

        /*
         * 检查每个 epoch 是否能够映射到合法的 PSG 样本范围。
         *
         * 这里只计算范围，不读取全部 epoch，
         * 避免集成测试消耗过多内存和时间。
         */
        for (const auto &epoch : epochs)
        {
            const std::int64_t startSample =
                EpochStartSample(
                    epoch,
                    signal.sampleRateHz);

            const std::int64_t sampleCount =
                EpochSampleCount(
                    epoch,
                    signal.sampleRateHz);

            Require(
                startSample >= 0,
                "Epoch start sample must not be negative.");

            Require(
                sampleCount > 0,
                "Epoch sample count must be positive.");

            Require(
                startSample <=
                    signal.sampleCount - sampleCount,
                "Epoch sample range must remain inside "
                "the EEG signal.");
        }

        /*
         * 实际读取一个 epoch 对应的 EEG 样本。
         *
         * 分别读取首、中、尾窗口，证明时间线不只是理论上有效，
         * 而是真能通过 EdfReader 映射到真实 PSG 数据。
         */
        const auto ReadAndVerifyEpochWindow =
            [&psgReader,
             signalIndex,
             &signal,
             &epochs](std::size_t epochIndex)
        {
            const auto &epoch =
                epochs.at(epochIndex);

            const std::int64_t startSample =
                EpochStartSample(
                    epoch,
                    signal.sampleRateHz);

            const std::int64_t sampleCount64 =
                EpochSampleCount(
                    epoch,
                    signal.sampleRateHz);

            Require(
                sampleCount64 <=
                    static_cast<std::int64_t>(
                        std::numeric_limits<int>::max()),
                "Epoch sample count exceeds "
                "ReadPhysicalSamples limit.");

            const int sampleCount =
                static_cast<int>(sampleCount64);

            const auto samples =
                psgReader.ReadPhysicalSamples(
                    signalIndex.value(),
                    startSample,
                    sampleCount);

            Require(
                samples.size() ==
                    static_cast<std::size_t>(sampleCount),
                "Real epoch EEG window size is incorrect.");
        };

        ReadAndVerifyEpochWindow(0);

        if (epochs.size() > 2)
        {
            ReadAndVerifyEpochWindow(
                epochs.size() / 2);
        }

        if (epochs.size() > 1)
        {
            ReadAndVerifyEpochWindow(
                epochs.size() - 1);
        }

        std::cout
            << "Sleep-EDF epoch integration test passed\n"
            << "annotations=" << annotations.size() << '\n'
            << "epochs=" << epochs.size() << '\n'
            << "signal=" << signal.label << '\n'
            << "sample_rate_hz=" << signal.sampleRateHz << '\n'
            << "samples_per_epoch="
            << EpochSampleCount(
                   epochs.front(),
                   signal.sampleRateHz)
            << '\n'
            << "first_epoch_start_seconds="
            << epochs.front().startSeconds
            << '\n'
            << "last_epoch_start_seconds="
            << epochs.back().startSeconds
            << '\n';
    }

} // namespace

int main()
{
    try
    {
        TestBasicCompleteSplit();
        TestBasicPartialFinalEpoch();
        TestMissingDurationFallback();
        TestSecondsFallback();

        TestMneStyleFilteringAndContinuousIndex();
        TestMneStyleKeepsInvalidStagesWhenConfigured();
        TestMneStyleDropsPartialEpoch();
        TestTooFewMneAnnotations();

        TestInvalidArguments();
        TestUnsortedAnnotations();

        TestRealSleepEdfIntegrationIfConfigured();

        std::cout
            << "All epoch tests passed.\n";

        return EXIT_SUCCESS;
    }
    catch (const std::exception &exception)
    {
        std::cerr
            << "epoch_test failed: "
            << exception.what()
            << '\n';

        return EXIT_FAILURE;
    }
}
