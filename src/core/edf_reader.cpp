#include "eeg_to_hypnogram/edf_reader.h"

#include <edflib.h>

#include <cctype>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <utility>

namespace
{

    // EDFlib 使用 100 ns tick。
    constexpr double kEdfTimeDimension =
        static_cast<double>(EDFLIB_TIME_DIMENSION);

    // 安全地将 C 字符串转换成 std::string。
    std::string SafeCString(const char *value)
    {
        return value == nullptr
                   ? std::string{}
                   : std::string{value};
    }

    // 标签归一化：
    // 1. 转换为小写
    // 2. 连续空白压缩为单个空格
    // 3. 移除首尾空白
    std::string NormalizeLabel(std::string text)
    {
        std::string normalized;
        normalized.reserve(text.size());

        bool lastWasSpace = true;

        for (const unsigned char character : text)
        {
            const bool isSpace =
                std::isspace(character) != 0;

            if (isSpace)
            {
                if (!lastWasSpace)
                {
                    normalized.push_back(' ');
                    lastWasSpace = true;
                }

                continue;
            }

            normalized.push_back(
                static_cast<char>(std::tolower(character)));

            lastWasSpace = false;
        }

        if (!normalized.empty() &&
            normalized.back() == ' ')
        {
            normalized.pop_back();
        }

        return normalized;
    }

    // 将 EDFlib tick 转换为秒。
    double TicksToSeconds(std::int64_t ticks)
    {
        return static_cast<double>(ticks) /
               kEdfTimeDimension;
    }

    // 解析 annotation 持续时间。
    //
    // 优先使用 EDFlib 提供的持续时间字符串。
    // 如果没有字符串，则回退到 durationTicks。
    double ParseDurationSeconds(
        const std::string &value,
        std::int64_t durationTicks)
    {
        if (!value.empty())
        {
            char *endPointer = nullptr;

            const double parsed =
                std::strtod(value.c_str(), &endPointer);

            if (endPointer != value.c_str())
            {
                return parsed;
            }
        }

        if (durationTicks >= 0)
        {
            return TicksToSeconds(durationTicks);
        }

        return -1.0;
    }

    // 将 EDFlib 原始头信息转换为项目内部结构。
    //
    // 该函数只存在于 cpp 文件中，因此公共头文件不需要知道
    // edflib_hdr_t 的存在。
    eeg_to_hypnogram::EdfHeaderInfo BuildHeaderInfo(
        const edflib_hdr_t &rawHeader,
        const std::string &filePath)
    {
        if (rawHeader.edfsignals < 0 ||
            rawHeader.edfsignals > EDFLIB_MAXSIGNALS)
        {
            throw std::runtime_error(
                "EDF header contains an invalid signal count.");
        }

        if (rawHeader.annotations_in_file < 0)
        {
            throw std::runtime_error(
                "EDF header contains an invalid annotation count.");
        }

        eeg_to_hypnogram::EdfHeaderInfo header;

        header.filePath = filePath;
        header.fileType = rawHeader.filetype;
        header.signalCount = rawHeader.edfsignals;

        header.fileDurationTicks =
            rawHeader.file_duration;

        header.fileDurationSeconds =
            TicksToSeconds(rawHeader.file_duration);

        header.startDateDay =
            rawHeader.startdate_day;

        header.startDateMonth =
            rawHeader.startdate_month;

        header.startDateYear =
            rawHeader.startdate_year;

        header.startTimeHour =
            rawHeader.starttime_hour;

        header.startTimeMinute =
            rawHeader.starttime_minute;

        header.startTimeSecond =
            rawHeader.starttime_second;

        header.annotationCount =
            rawHeader.annotations_in_file;

        header.signals.reserve(
            static_cast<std::size_t>(
                rawHeader.edfsignals));

        for (
            int signalIndex = 0;
            signalIndex < rawHeader.edfsignals;
            ++signalIndex)
        {
            const auto &rawSignal =
                rawHeader.signalparam[signalIndex];

            eeg_to_hypnogram::EdfSignalInfo signal;

            signal.label =
                SafeCString(rawSignal.label);

            signal.sampleCount =
                rawSignal.smp_in_file;

            signal.samplesPerDataRecord =
                rawSignal.smp_in_datarecord;

            signal.physicalMin =
                rawSignal.phys_min;

            signal.physicalMax =
                rawSignal.phys_max;

            signal.digitalMin =
                rawSignal.dig_min;

            signal.digitalMax =
                rawSignal.dig_max;

            signal.physicalDimension =
                SafeCString(rawSignal.physdimension);

            signal.prefilter =
                SafeCString(rawSignal.prefilter);

            signal.transducer =
                SafeCString(rawSignal.transducer);

            if (rawHeader.datarecord_duration > 0)
            {
                const double dataRecordDurationSeconds =
                    TicksToSeconds(
                        rawHeader.datarecord_duration);

                signal.sampleRateHz =
                    static_cast<double>(
                        rawSignal.smp_in_datarecord) /
                    dataRecordDurationSeconds;
            }

            header.signals.push_back(
                std::move(signal));
        }

        return header;
    }

    // Open() 构建头信息期间使用的临时句柄保护器。
    //
    // 如果 BuildHeaderInfo() 抛异常，临时句柄会被自动关闭。
    class PendingEdfHandle final
    {
    public:
        explicit PendingEdfHandle(int handle) noexcept
            : handle_(handle)
        {
        }

        ~PendingEdfHandle()
        {
            if (handle_ >= 0)
            {
                edfclose_file(handle_);
            }
        }

        PendingEdfHandle(
            const PendingEdfHandle &) = delete;

        PendingEdfHandle &operator=(
            const PendingEdfHandle &) = delete;

        [[nodiscard]] int Release() noexcept
        {
            return std::exchange(handle_, -1);
        }

    private:
        int handle_ = -1;
    };

} // namespace

namespace eeg_to_hypnogram
{

    EdfReader::~EdfReader() noexcept
    {
        Close();
    }

    EdfReader::EdfReader(
        EdfReader &&other) noexcept
        : handle_(
              std::exchange(other.handle_, -1)),
          annotationsLoaded_(
              std::exchange(
                  other.annotationsLoaded_,
                  false)),
          header_(
              std::move(other.header_))
    {
    }

    EdfReader &EdfReader::operator=(
        EdfReader &&other) noexcept
    {
        if (this == &other)
        {
            return *this;
        }

        Close();

        handle_ =
            std::exchange(other.handle_, -1);

        annotationsLoaded_ =
            std::exchange(
                other.annotationsLoaded_,
                false);

        header_ =
            std::move(other.header_);

        return *this;
    }

    void EdfReader::Open(
        const std::string &filePath,
        bool readAnnotations)
    {
        if (filePath.empty())
        {
            throw std::invalid_argument(
                "EDF file path must not be empty.");
        }

        // 如果对象之前打开过文件，先清理旧状态。
        Close();

        edflib_hdr_t rawHeader{};

        const int annotationMode =
            readAnnotations
                ? EDFLIB_READ_ALL_ANNOTATIONS
                : EDFLIB_DO_NOT_READ_ANNOTATIONS;

        const int openResult =
            edfopen_file_readonly(
                filePath.c_str(),
                &rawHeader,
                annotationMode);

        if (openResult != 0 ||
            rawHeader.handle < 0)
        {
            const int errorCode =
                rawHeader.filetype < 0
                    ? rawHeader.filetype
                    : openResult;

            throw std::runtime_error(
                "Failed to open EDF file: " +
                filePath +
                " (" +
                BuildOpenErrorMessage(errorCode) +
                ")");
        }

        // 在正式提交给当前对象之前，由临时对象管理句柄。
        PendingEdfHandle pendingHandle(
            rawHeader.handle);

        // 这里可能发生字符串或 vector 分配。
        // 如果抛异常，pendingHandle 会自动关闭句柄。
        EdfHeaderInfo newHeader =
            BuildHeaderInfo(
                rawHeader,
                filePath);

        handle_ =
            pendingHandle.Release();

        annotationsLoaded_ =
            readAnnotations;

        header_ =
            std::move(newHeader);
    }

    void EdfReader::Close() noexcept
    {
        const int handleToClose =
            std::exchange(handle_, -1);

        annotationsLoaded_ = false;
        header_ = {};

        if (handleToClose >= 0)
        {
            edfclose_file(handleToClose);
        }
    }

    bool EdfReader::IsOpen() const noexcept
    {
        return handle_ >= 0;
    }

    bool EdfReader::AnnotationsLoaded() const noexcept
    {
        return IsOpen() &&
               annotationsLoaded_;
    }

    const EdfHeaderInfo &EdfReader::Header() const
    {
        ThrowIfNotOpen();
        return header_;
    }

    std::vector<double>
    EdfReader::ReadPhysicalSamples(
        int signalIndex,
        std::int64_t startSample,
        int sampleCount) const
    {
        ThrowIfNotOpen();

        if (signalIndex < 0 ||
            signalIndex >= header_.signalCount)
        {
            throw std::out_of_range(
                "Signal index out of range.");
        }

        if (startSample < 0)
        {
            throw std::invalid_argument(
                "Start sample must be greater than or equal to zero.");
        }

        if (sampleCount < 0)
        {
            throw std::invalid_argument(
                "Sample count must be greater than or equal to zero.");
        }

        const std::int64_t totalSamples =
            header_
                .signals[static_cast<std::size_t>(
                    signalIndex)]
                .sampleCount;

        if (totalSamples < 0)
        {
            throw std::runtime_error(
                "Signal sample count is invalid.");
        }

        if (startSample > totalSamples)
        {
            throw std::out_of_range(
                "Start sample is beyond the end of the signal.");
        }

        if (sampleCount == 0 ||
            startSample == totalSamples)
        {
            return {};
        }

        if (edfseek(
                handle_,
                signalIndex,
                startSample,
                EDFSEEK_SET) < 0)
        {
            throw std::runtime_error(
                "Failed to seek EDF signal.");
        }

        std::vector<double> samples(
            static_cast<std::size_t>(
                sampleCount));

        const int readCount =
            edfread_physical_samples(
                handle_,
                signalIndex,
                sampleCount,
                samples.data());

        if (readCount < 0)
        {
            throw std::runtime_error(
                "Failed to read physical EDF samples.");
        }

        // 文件末尾可能少于请求数量。
        samples.resize(
            static_cast<std::size_t>(
                readCount));

        return samples;
    }

    std::vector<double>
    EdfReader::ReadAllPhysicalSamples(
        int signalIndex) const
    {
        ThrowIfNotOpen();

        if (signalIndex < 0 ||
            signalIndex >= header_.signalCount)
        {
            throw std::out_of_range(
                "Signal index out of range.");
        }

        const std::int64_t totalSamples =
            header_
                .signals[static_cast<std::size_t>(
                    signalIndex)]
                .sampleCount;

        if (totalSamples < 0)
        {
            throw std::runtime_error(
                "Signal sample count is invalid.");
        }

        if (
            totalSamples >
            static_cast<std::int64_t>(
                std::numeric_limits<int>::max()))
        {
            throw std::runtime_error(
                "Signal sample count exceeds the current "
                "whole-signal read limit.");
        }

        return ReadPhysicalSamples(
            signalIndex,
            0,
            static_cast<int>(totalSamples));
    }

    std::vector<double>
    EdfReader::ReadAllPhysicalSamplesByLabel(
        const std::string &label) const
    {
        ThrowIfNotOpen();

        const std::optional<int> signalIndex =
            FindSignalIndexByLabel(label);

        if (!signalIndex.has_value())
        {
            throw std::out_of_range(
                "Signal label not found: " +
                label);
        }

        return ReadAllPhysicalSamples(
            *signalIndex);
    }

    std::vector<EdfAnnotation>
    EdfReader::ReadAnnotations() const
    {
        ThrowIfAnnotationsNotLoaded();

        if (
            header_.annotationCount >
            static_cast<std::int64_t>(
                std::numeric_limits<int>::max()))
        {
            throw std::runtime_error(
                "EDF annotation count exceeds the "
                "EDFlib annotation index limit.");
        }

        std::vector<EdfAnnotation> annotations;

        annotations.reserve(
            static_cast<std::size_t>(
                header_.annotationCount));

        const int annotationCount =
            static_cast<int>(
                header_.annotationCount);

        for (
            int annotationIndex = 0;
            annotationIndex < annotationCount;
            ++annotationIndex)
        {
            edflib_annotation_t rawAnnotation{};

            const int readResult =
                edf_get_annotation(
                    handle_,
                    annotationIndex,
                    &rawAnnotation);

            if (readResult != 0)
            {
                throw std::runtime_error(
                    "Failed to read EDF annotation index " +
                    std::to_string(annotationIndex) +
                    ".");
            }

            EdfAnnotation annotation;

            annotation.onsetTicks =
                rawAnnotation.onset;

            annotation.durationTicks =
                rawAnnotation.duration_l;

            annotation.durationSecondsText =
                SafeCString(
                    rawAnnotation.duration);

            annotation.text =
                SafeCString(
                    rawAnnotation.annotation);

            annotations.push_back(
                std::move(annotation));
        }

        return annotations;
    }

    std::vector<EdfAnnotation>
    EdfReader::ReadAnnotationsByKeyword(
        const std::string &keyword) const
    {
        ThrowIfAnnotationsNotLoaded();

        const std::string target =
            NormalizeLabel(keyword);

        if (target.empty())
        {
            return {};
        }

        const std::vector<EdfAnnotation> annotations =
            ReadAnnotations();

        std::vector<EdfAnnotation> filtered;
        filtered.reserve(annotations.size());

        for (const EdfAnnotation &annotation : annotations)
        {
            const std::string normalizedText =
                NormalizeLabel(annotation.text);

            if (
                normalizedText.find(target) !=
                std::string::npos)
            {
                filtered.push_back(annotation);
            }
        }

        return filtered;
    }

    std::vector<SleepStageAnnotation>
    EdfReader::ReadSleepStageAnnotations() const
    {
        ThrowIfAnnotationsNotLoaded();

        const std::vector<EdfAnnotation> annotations =
            ReadAnnotations();

        std::vector<SleepStageAnnotation> stages;
        stages.reserve(annotations.size());

        for (const EdfAnnotation &annotation : annotations)
        {
            SleepStage stage = SleepStage::Unknown;

            if (!TryParseSleepStage(
                    annotation.text,
                    stage))
            {
                continue;
            }

            SleepStageAnnotation item;

            item.stage =
                SleepStageToString(stage);

            item.onsetTicks =
                annotation.onsetTicks;

            item.onsetSeconds =
                TicksToSeconds(
                    annotation.onsetTicks);

            item.durationTicks =
                annotation.durationTicks;

            item.durationSeconds =
                ParseDurationSeconds(
                    annotation.durationSecondsText,
                    annotation.durationTicks);

            item.rawText =
                annotation.text;

            stages.push_back(
                std::move(item));
        }

        return stages;
    }

    std::optional<int>
    EdfReader::FindSignalIndexByLabel(
        const std::string &label) const
    {
        ThrowIfNotOpen();

        const std::string target =
            NormalizeLabel(label);

        if (target.empty())
        {
            return std::nullopt;
        }

        for (
            int signalIndex = 0;
            signalIndex < header_.signalCount;
            ++signalIndex)
        {
            const std::string current =
                NormalizeLabel(
                    header_
                        .signals[static_cast<std::size_t>(
                            signalIndex)]
                        .label);

            if (current == target)
            {
                return signalIndex;
            }
        }

        return std::nullopt;
    }

    std::vector<int>
    EdfReader::FindSignalIndicesByLabelFuzzy(
        const std::string &keyword) const
    {
        ThrowIfNotOpen();

        std::vector<int> indices;

        const std::string target =
            NormalizeLabel(keyword);

        if (target.empty())
        {
            return indices;
        }

        for (
            int signalIndex = 0;
            signalIndex < header_.signalCount;
            ++signalIndex)
        {
            const std::string current =
                NormalizeLabel(
                    header_
                        .signals[static_cast<std::size_t>(
                            signalIndex)]
                        .label);

            if (
                current.find(target) !=
                std::string::npos)
            {
                indices.push_back(signalIndex);
            }
        }

        return indices;
    }

    bool EdfReader::HasSignal(
        const std::string &labelOrKeyword,
        bool fuzzy) const
    {
        ThrowIfNotOpen();

        if (fuzzy)
        {
            return !FindSignalIndicesByLabelFuzzy(
                        labelOrKeyword)
                        .empty();
        }

        return FindSignalIndexByLabel(
                   labelOrKeyword)
            .has_value();
    }

    void EdfReader::ThrowIfNotOpen() const
    {
        if (!IsOpen())
        {
            throw std::runtime_error(
                "EDF file is not open.");
        }
    }

    void EdfReader::ThrowIfAnnotationsNotLoaded() const
    {
        ThrowIfNotOpen();

        if (!annotationsLoaded_)
        {
            throw std::logic_error(
                "EDF annotations were not loaded. "
                "Reopen the file with readAnnotations=true.");
        }
    }

    std::string EdfReader::BuildOpenErrorMessage(
        int edflibErrorCode)
    {
        switch (edflibErrorCode)
        {
        case EDFLIB_MALLOC_ERROR:
            return "out of memory";

        case EDFLIB_NO_SUCH_FILE_OR_DIRECTORY:
            return "file not found";

        case EDFLIB_FILE_CONTAINS_FORMAT_ERRORS:
            return "format error";

        case EDFLIB_MAXFILES_REACHED:
            return "too many open EDF files";

        case EDFLIB_FILE_READ_ERROR:
            return "file read error";

        case EDFLIB_FILE_ALREADY_OPENED:
            return "file already open";

        case EDFLIB_FILETYPE_ERROR:
            return "unsupported file type";

        case EDFLIB_FILE_WRITE_ERROR:
            return "file write error";

        case EDFLIB_NUMBER_OF_SIGNALS_INVALID:
            return "invalid number of signals";

        case EDFLIB_FILE_IS_DISCONTINUOUS:
            return "discontinuous file";

        case EDFLIB_INVALID_READ_ANNOTS_VALUE:
            return "invalid read annotation mode";

        case EDFLIB_ARCH_ERROR:
            return "architecture compatibility error";

        default:
            return "unknown EDFlib error: " +
                   std::to_string(edflibErrorCode);
        }
    }

} // namespace eeg_to_hypnogram
