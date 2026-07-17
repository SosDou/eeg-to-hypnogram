#include "eeg_to_hypnogram/edf_reader.h"

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace
{

    void Require(
        bool condition,
        const std::string &message)
    {
        if (!condition)
        {
            throw std::runtime_error(
                "Test failed: " + message);
        }
    }

    template <
        typename ExceptionType,
        typename Function>
    void RequireThrows(
        Function &&function,
        const std::string &message)
    {
        try
        {
            std::forward<Function>(function)();
        }
        catch (const ExceptionType &)
        {
            return;
        }
        catch (const std::exception &exception)
        {
            throw std::runtime_error(
                "Test failed: " +
                message +
                "; unexpected exception: " +
                exception.what());
        }

        throw std::runtime_error(
            "Test failed: " +
            message +
            "; no exception was thrown.");
    }

    // 不需要真实 EDF 文件的基础状态测试。
    void RunStateTests()
    {
        using eeg_to_hypnogram::EdfReader;

        EdfReader reader;

        Require(
            !reader.IsOpen(),
            "a new reader must be closed");

        Require(
            !reader.AnnotationsLoaded(),
            "a new reader must not report loaded annotations");

        RequireThrows<std::runtime_error>(
            [&reader]()
            {
                static_cast<void>(
                    reader.Header());
            },
            "Header() must reject a closed reader");

        RequireThrows<std::invalid_argument>(
            [&reader]()
            {
                reader.Open("");
            },
            "Open() must reject an empty path");

        RequireThrows<std::runtime_error>(
            [&reader]()
            {
                reader.Open(
                    "__eeg_to_hypnogram_missing_file__.edf");
            },
            "Open() must reject a missing EDF file");

        Require(
            !reader.IsOpen(),
            "reader must remain closed after a failed Open()");

        // Close() 应当可以安全重复调用。
        reader.Close();
        reader.Close();
    }

    // 使用真实 PSG 文件进行集成测试。
    //
    // 通过环境变量 EEG_TEST_EDF_FILE 启用。
    void RunPsgIntegrationTest(
        const char *filePath)
    {
        using eeg_to_hypnogram::EdfReader;

        EdfReader reader;

        reader.Open(
            filePath,
            true);

        Require(
            reader.IsOpen(),
            "PSG reader must be open");

        Require(
            reader.AnnotationsLoaded(),
            "annotations must be marked as loaded");

        const auto &header =
            reader.Header();

        Require(
            header.signalCount > 0,
            "PSG file must contain at least one signal");

        Require(
            header.signals.size() ==
                static_cast<std::size_t>(
                    header.signalCount),
            "signal metadata count must match signalCount");

        const auto &firstSignal =
            header.signals.front();

        Require(
            !firstSignal.label.empty(),
            "first signal label must not be empty");

        const auto exactIndex =
            reader.FindSignalIndexByLabel(
                firstSignal.label);

        Require(
            exactIndex.has_value(),
            "exact signal lookup must find the first signal");

        Require(
            *exactIndex == 0,
            "first signal must have index zero");

        Require(
            reader.HasSignal(firstSignal.label),
            "HasSignal() must find the first signal");

        if (firstSignal.sampleCount > 0)
        {
            const int sampleCount =
                static_cast<int>(
                    std::min<std::int64_t>(
                        firstSignal.sampleCount,
                        10));

            const auto samples =
                reader.ReadPhysicalSamples(
                    0,
                    0,
                    sampleCount);

            Require(
                samples.size() ==
                    static_cast<std::size_t>(
                        sampleCount),
                "sample window size must match "
                "the requested in-range size");
        }

        // 验证移动构造是否正确转移底层 EDF 句柄。
        EdfReader movedReader(
            std::move(reader));

        Require(
            !reader.IsOpen(),
            "moved-from reader must be closed");

        Require(
            movedReader.IsOpen(),
            "move-constructed reader must own the EDF handle");

        movedReader.Close();
        movedReader.Close();

        std::cout
            << "PSG integration test passed: "
            << filePath
            << '\n';
    }

    // 使用真实 Hypnogram 文件测试 annotation。
    //
    // 通过环境变量 EEG_TEST_HYPNOGRAM_FILE 启用。
    void RunAnnotationIntegrationTest(
        const char *filePath)
    {
        using eeg_to_hypnogram::EdfReader;

        EdfReader reader;

        // 首先验证不读取 annotation 的模式。
        reader.Open(
            filePath,
            false);

        Require(
            !reader.AnnotationsLoaded(),
            "annotations must be disabled");

        RequireThrows<std::logic_error>(
            [&reader]()
            {
                static_cast<void>(
                    reader.ReadAnnotations());
            },
            "ReadAnnotations() must reject a file "
            "opened without annotations");

        // 重新打开并加载所有 annotation。
        reader.Open(
            filePath,
            true);

        const auto annotations =
            reader.ReadAnnotations();

        const auto stages =
            reader.ReadSleepStageAnnotations();

        Require(
            stages.size() <= annotations.size(),
            "parsed stage count cannot exceed "
            "raw annotation count");

        std::cout
            << "Annotation integration test passed: "
            << filePath
            << " (annotations="
            << annotations.size()
            << ", stages="
            << stages.size()
            << ")\n";
    }

} // namespace

int main()
{
    try
    {
        RunStateTests();

        const char *psgFile =
            std::getenv(
                "EEG_TEST_EDF_FILE");

        if (
            psgFile != nullptr &&
            *psgFile != '\0')
        {
            RunPsgIntegrationTest(
                psgFile);
        }
        else
        {
            std::cout
                << "EEG_TEST_EDF_FILE is not set; "
                << "PSG integration test skipped.\n";
        }

        const char *hypnogramFile =
            std::getenv(
                "EEG_TEST_HYPNOGRAM_FILE");

        if (
            hypnogramFile != nullptr &&
            *hypnogramFile != '\0')
        {
            RunAnnotationIntegrationTest(
                hypnogramFile);
        }
        else
        {
            std::cout
                << "EEG_TEST_HYPNOGRAM_FILE is not set; "
                << "annotation integration test skipped.\n";
        }

        std::cout
            << "All enabled edf_reader tests passed.\n";

        return 0;
    }
    catch (const std::exception &exception)
    {
        std::cerr
            << exception.what()
            << '\n';

        return 1;
    }
}