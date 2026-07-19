#pragma once

#include "eeg_to_hypnogram/sleep_stage.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace eeg_to_hypnogram
{

    // 单个 EDF 信号通道的信息。
    struct EdfSignalInfo
    {
        // 信号标签，例如 EEG Fpz-Cz。
        std::string label;

        // 该信号在整个文件中的总采样点数。
        std::int64_t sampleCount = 0;

        // 每个 data record 中包含的采样点数。
        int samplesPerDataRecord = 0;

        // 采样率，单位 Hz。
        double sampleRateHz = 0.0;

        // 物理值范围。
        double physicalMin = 0.0;
        double physicalMax = 0.0;

        // ADC 数字值范围。
        int digitalMin = 0;
        int digitalMax = 0;

        // 物理单位，例如 uV。
        std::string physicalDimension;

        // 预处理说明，例如滤波设置。
        std::string prefilter;

        // 传感器或导联说明。
        std::string transducer;
    };

    // EDF 文件头信息摘要。
    struct EdfHeaderInfo
    {
        // EDF 文件路径，仅用于日志和调试。
        std::string filePath;

        // EDFlib 文件类型。
        int fileType = 0;

        // 普通信号数量，不包含 annotation 虚拟通道。
        int signalCount = 0;

        // 文件时长，单位为 100 ns tick。
        std::int64_t fileDurationTicks = 0;

        // 文件时长，单位为秒。
        double fileDurationSeconds = 0.0;

        // 记录开始日期。
        int startDateDay = 0;
        int startDateMonth = 0;
        int startDateYear = 0;

        // 记录开始时间。
        int startTimeHour = 0;
        int startTimeMinute = 0;
        int startTimeSecond = 0;

        // annotation 数量。
        std::int64_t annotationCount = 0;

        // 各信号通道信息。
        std::vector<EdfSignalInfo> signals;
    };

    // 单条 EDF annotation。
    struct EdfAnnotation
    {
        // annotation 起始时间，单位为 100 ns tick。
        std::int64_t onsetTicks = 0;

        // annotation 持续时间，单位为 100 ns tick。
        // 小于 0 表示未提供。
        std::int64_t durationTicks = -1;

        // EDFlib 返回的原始持续时间字符串。
        std::string durationSecondsText;

        // annotation 正文。
        std::string text;
    };

    // EDF 文件读取器。
    //
    // 该类负责：
    // 1. EDF/EDF+ 文件打开和关闭
    // 2. 文件头读取
    // 3. 信号样本读取
    // 4. annotation 读取
    // 5. 调用公共 Sleep Stage 模块解析 Sleep-EDF 睡眠阶段文本
    class EdfReader
    {
    public:
        EdfReader() = default;
        ~EdfReader() noexcept;

        // EDF 句柄不允许复制。
        EdfReader(const EdfReader &) = delete;
        EdfReader &operator=(const EdfReader &) = delete;

        // 允许转移 EDF 句柄所有权。
        EdfReader(EdfReader &&other) noexcept;
        EdfReader &operator=(EdfReader &&other) noexcept;

        // 打开 EDF 文件。
        //
        // readAnnotations=true：
        // 在打开阶段读取所有 annotation。
        //
        // readAnnotations=false：
        // 不读取 annotation，后续调用 annotation 相关函数会抛异常。
        void Open(
            const std::string &filePath,
            bool readAnnotations = true);

        // 关闭当前文件。
        // 重复调用是安全的。
        void Close() noexcept;

        // 当前是否已打开 EDF 文件。
        [[nodiscard]] bool IsOpen() const noexcept;

        // 当前文件是否在打开阶段加载了 annotation。
        [[nodiscard]] bool AnnotationsLoaded() const noexcept;

        // 获取文件头。
        [[nodiscard]] const EdfHeaderInfo &Header() const;

        // 读取指定信号的一段物理值样本。
        //
        // 范围：
        // [startSample, startSample + sampleCount)
        [[nodiscard]] std::vector<double> ReadPhysicalSamples(
            int signalIndex,
            std::int64_t startSample,
            int sampleCount) const;

        // 读取指定信号的全部物理值样本。
        //
        // 这是 CLI 便利接口。
        // WebAssembly 和 Flutter 层应优先使用窗口读取接口。
        [[nodiscard]] std::vector<double> ReadAllPhysicalSamples(
            int signalIndex) const;

        // 根据通道 label 读取全部物理值样本。
        [[nodiscard]] std::vector<double> ReadAllPhysicalSamplesByLabel(
            const std::string &label) const;

        // 读取全部 annotation。
        [[nodiscard]] std::vector<EdfAnnotation> ReadAnnotations() const;

        // 根据关键词过滤 annotation。
        [[nodiscard]] std::vector<EdfAnnotation> ReadAnnotationsByKeyword(
            const std::string &keyword) const;

        // 解析 Sleep-EDF 睡眠阶段 annotation。
        [[nodiscard]] std::vector<SleepStageAnnotation>
        ReadSleepStageAnnotations() const;

        // 精确查找通道 label。
        //
        // 匹配时：
        // 1. 忽略大小写
        // 2. 压缩连续空白
        // 3. 忽略首尾空白
        [[nodiscard]] std::optional<int> FindSignalIndexByLabel(
            const std::string &label) const;

        // 模糊查找通道 label。
        [[nodiscard]] std::vector<int> FindSignalIndicesByLabelFuzzy(
            const std::string &keyword) const;

        // 判断通道是否存在。
        [[nodiscard]] bool HasSignal(
            const std::string &labelOrKeyword,
            bool fuzzy = false) const;

    private:
        // 检查文件是否已打开。
        void ThrowIfNotOpen() const;

        // 检查 annotation 是否已加载。
        void ThrowIfAnnotationsNotLoaded() const;

        // 将 EDFlib 错误码转换为可读文本。
        [[nodiscard]] static std::string BuildOpenErrorMessage(
            int edflibErrorCode);

        // EDFlib 文件句柄，-1 表示未打开。
        int handle_ = -1;

        // 打开文件时是否读取了 annotation。
        bool annotationsLoaded_ = false;

        // 缓存的文件头信息。
        EdfHeaderInfo header_{};
    };

} // namespace eeg_to_hypnogram
