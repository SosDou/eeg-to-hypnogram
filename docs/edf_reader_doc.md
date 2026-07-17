# `EdfReader` 技术实现说明

## 1. 模块职责

`EdfReader` 是基于 Teuniz EDFlib 的 C++ 封装层，负责将 EDF/EDF+ 文件中的底层 C 接口转换为资源安全、类型明确的现代 C++ 接口。

它提供以下能力：

- 打开和关闭 EDF/EDF+ 文件
- 读取文件头摘要
- 获取信号通道数量和通道元数据
- 精确或模糊查找信号通道
- 按样本区间读取物理值信号
- 读取整条信号通道
- 读取 EDF+ annotation
- 按关键词筛选 annotation
- 将 Sleep-EDF annotation 解析为标准睡眠阶段
- 使用 RAII 管理 EDFlib 文件句柄
- 支持移动语义并禁止资源对象复制

公共头文件只暴露项目内部 C++ 类型，不暴露任何 EDFlib 结构体。

---

## 2. 公共数据结构

### 2.1 `EdfSignalInfo`

`EdfSignalInfo` 描述单个信号通道的元数据。

```cpp
struct EdfSignalInfo
{
    std::string label;
    std::int64_t sampleCount = 0;
    int samplesPerDataRecord = 0;
    double sampleRateHz = 0.0;

    double physicalMin = 0.0;
    double physicalMax = 0.0;

    int digitalMin = 0;
    int digitalMax = 0;

    std::string physicalDimension;
    std::string prefilter;
    std::string transducer;
};
```

| 字段 | 含义 |
|---|---|
| `label` | 通道标签，例如 `EEG Fpz-Cz` |
| `sampleCount` | 通道在整个文件中的总采样点数 |
| `samplesPerDataRecord` | 每个 EDF data record 中的采样点数 |
| `sampleRateHz` | 根据 data record 时长计算出的采样率 |
| `physicalMin` / `physicalMax` | 物理值范围 |
| `digitalMin` / `digitalMax` | ADC 数字值范围 |
| `physicalDimension` | 物理单位，例如 `uV` |
| `prefilter` | 预处理或滤波说明 |
| `transducer` | 传感器或导联说明 |

采样率通过以下关系计算：

```text
sampleRateHz =
    samplesPerDataRecord / dataRecordDurationSeconds
```

EDFlib 时间单位为 100 ns tick，因此：

```text
dataRecordDurationSeconds =
    datarecord_duration / EDFLIB_TIME_DIMENSION
```

### 2.2 `EdfHeaderInfo`

`EdfHeaderInfo` 保存整个 EDF 文件头的摘要。

```cpp
struct EdfHeaderInfo
{
    std::string filePath;

    int fileType = 0;
    int signalCount = 0;

    std::int64_t fileDurationTicks = 0;
    double fileDurationSeconds = 0.0;

    int startDateDay = 0;
    int startDateMonth = 0;
    int startDateYear = 0;

    int startTimeHour = 0;
    int startTimeMinute = 0;
    int startTimeSecond = 0;

    std::int64_t annotationCount = 0;

    std::vector<EdfSignalInfo> signals;
};
```

文件持续时间同时保留 tick 和秒：

- tick 保留 EDFlib 原始整数精度
- 秒便于上层算法直接使用

### 2.3 `EdfAnnotation`

```cpp
struct EdfAnnotation
{
    std::int64_t onsetTicks = 0;
    std::int64_t durationTicks = -1;

    std::string durationSecondsText;
    std::string text;
};
```

`durationTicks < 0` 表示原始 annotation 未提供持续时间。

`durationSecondsText` 保留 EDFlib 返回的原始持续时间文本，便于处理不同 EDF 文件中的表示差异。

### 2.4 `SleepStageAnnotation`

```cpp
struct SleepStageAnnotation
{
    std::string stage;

    std::int64_t onsetTicks = 0;
    double onsetSeconds = 0.0;

    std::int64_t durationTicks = -1;
    double durationSeconds = -1.0;

    std::string rawText;
};
```

支持的标准化标签：

```text
W
N1
N2
N3
REM
UNKNOWN
MOVEMENT
```

Sleep-EDF 中的 Stage 3 和 Stage 4 会统一映射为 `N3`。

---

## 3. 资源管理

### 3.1 RAII

EDFlib 使用整数句柄表示已打开文件：

```cpp
int handle_ = -1;
```

约定：

```text
handle_ >= 0   文件已打开
handle_ == -1  文件未打开
```

析构函数自动关闭文件：

```cpp
EdfReader::~EdfReader() noexcept
{
    Close();
}
```

即使调用方因异常提前退出，底层 EDF 句柄也会被释放。

### 3.2 禁止复制

同一个 EDFlib 句柄不能由多个对象共同拥有：

```cpp
EdfReader(const EdfReader &) = delete;
EdfReader &operator=(const EdfReader &) = delete;
```

这避免了多个对象析构时重复调用 `edfclose_file()`。

### 3.3 移动语义

移动构造使用 `std::exchange()` 转移句柄：

```cpp
EdfReader::EdfReader(EdfReader &&other) noexcept
    : handle_(std::exchange(other.handle_, -1)),
      annotationsLoaded_(
          std::exchange(other.annotationsLoaded_, false)),
      header_(std::move(other.header_))
{
}
```

移动完成后：

- 新对象接管句柄
- 原对象句柄被重置为 `-1`
- 原对象不再拥有 EDF 资源

移动赋值会先关闭当前对象已经持有的文件，再接管新句柄。

---

## 4. 文件打开流程

### 4.1 annotation 加载模式

```cpp
void Open(
    const std::string &filePath,
    bool readAnnotations = true);
```

参数会映射为 EDFlib 的两种模式：

```cpp
const int annotationMode =
    readAnnotations
        ? EDFLIB_READ_ALL_ANNOTATIONS
        : EDFLIB_DO_NOT_READ_ANNOTATIONS;
```

当使用：

```cpp
reader.Open(path, false);
```

后续 annotation 接口会抛出 `std::logic_error`。这样可以明确区分：

```text
文件没有 annotation
```

和：

```text
打开时没有加载 annotation
```

### 4.2 调用 EDFlib

```cpp
edflib_hdr_t rawHeader{};

const int openResult =
    edfopen_file_readonly(
        filePath.c_str(),
        &rawHeader,
        annotationMode);
```

打开成功后，`rawHeader` 中包含：

- 文件句柄
- 文件类型
- 信号数量
- 文件时长
- data record 时长
- 开始日期和时间
- annotation 数量
- 每个信号的参数

### 4.3 强异常安全

文件打开成功后，头信息转换仍可能因为字符串或 `std::vector` 分配失败而抛异常。

实现中使用临时句柄保护器：

```cpp
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

    [[nodiscard]] int Release() noexcept
    {
        return std::exchange(handle_, -1);
    }

private:
    int handle_ = -1;
};
```

完整提交顺序：

```cpp
PendingEdfHandle pendingHandle(rawHeader.handle);

EdfHeaderInfo newHeader =
    BuildHeaderInfo(rawHeader, filePath);

handle_ = pendingHandle.Release();
annotationsLoaded_ = readAnnotations;
header_ = std::move(newHeader);
```

只有头信息完全构建成功后，句柄才正式交给 `EdfReader`。

如果构建失败，`PendingEdfHandle` 会自动关闭文件。

---

## 5. EDFlib 类型隔离

EDFlib 原始结构体只出现在 `.cpp` 文件中：

```cpp
EdfHeaderInfo BuildHeaderInfo(
    const edflib_hdr_t &rawHeader,
    const std::string &filePath);
```

公共头文件不包含 `<edflib.h>`，也不声明 EDFlib 结构体。

这样做可以：

1. 减少公共头文件对第三方库的耦合
2. 避免 EDFlib 头文件传播到其他编译单元
3. 降低未来替换底层 EDF 实现时的改动范围
4. 保持 `eeg_to_hypnogram_core` 的公共 API 稳定

头信息转换前会验证关键字段：

```cpp
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
```

随后逐个转换信号参数。

---

## 6. 信号读取

### 6.1 窗口读取

核心接口：

```cpp
std::vector<double> ReadPhysicalSamples(
    int signalIndex,
    std::int64_t startSample,
    int sampleCount) const;
```

读取范围：

```text
[startSample, startSample + sampleCount)
```

读取前校验：

```cpp
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
```

EDFlib 为每个信号维护独立读位置，因此读取前先定位：

```cpp
if (edfseek(
        handle_,
        signalIndex,
        startSample,
        EDFSEEK_SET) < 0)
{
    throw std::runtime_error(
        "Failed to seek EDF signal.");
}
```

然后读取物理值：

```cpp
std::vector<double> samples(
    static_cast<std::size_t>(sampleCount));

const int readCount =
    edfread_physical_samples(
        handle_,
        signalIndex,
        sampleCount,
        samples.data());
```

读取结果已经根据 EDF 头中的 digital/physical 范围转换为物理值，不是原始 ADC 整数。

接近文件末尾时，实际读取数量可能小于请求值：

```cpp
samples.resize(
    static_cast<std::size_t>(readCount));
```

返回 `vector` 的长度以真实读取数量为准。

### 6.2 整通道读取

整通道读取复用窗口接口：

```cpp
return ReadPhysicalSamples(
    signalIndex,
    0,
    static_cast<int>(totalSamples));
```

由于 EDFlib 的读取数量参数为 `int`，调用前会检查：

```cpp
if (totalSamples >
    static_cast<std::int64_t>(
        std::numeric_limits<int>::max()))
{
    throw std::runtime_error(
        "Signal sample count exceeds the current "
        "whole-signal read limit.");
}
```

较大的 EDF 文件应优先分窗口读取，避免一次性占用大量内存。

### 6.3 按标签读取

```cpp
const std::optional<int> signalIndex =
    FindSignalIndexByLabel(label);

if (!signalIndex.has_value())
{
    throw std::out_of_range(
        "Signal label not found: " + label);
}

return ReadAllPhysicalSamples(*signalIndex);
```

接口先解析标签，再复用按索引读取逻辑。

---

## 7. 通道标签查找

### 7.1 标签归一化

通道标签可能存在：

- 大小写差异
- 首尾空白
- 连续空格
- 制表符等其他空白字符

所有匹配都会先调用：

```cpp
std::string NormalizeLabel(std::string text);
```

归一化规则：

1. 转为小写
2. 连续空白压缩为一个普通空格
3. 删除首尾空白

例如：

```text
"  EEG   Fpz-Cz  "
```

会转换为：

```text
"eeg fpz-cz"
```

### 7.2 精确查找

```cpp
std::optional<int> FindSignalIndexByLabel(
    const std::string &label) const;
```

比较归一化后的完整标签。

找不到时返回 `std::nullopt`，不使用 `-1` 表示失败。

### 7.3 模糊查找

```cpp
std::vector<int> FindSignalIndicesByLabelFuzzy(
    const std::string &keyword) const;
```

使用归一化后的子串匹配：

```cpp
if (current.find(target) != std::string::npos)
{
    indices.push_back(signalIndex);
}
```

例如：

```text
eeg
fpz
eog
```

可用于匹配包含对应关键词的所有通道。

---

## 8. annotation 读取

annotation 接口首先检查加载状态：

```cpp
ThrowIfAnnotationsNotLoaded();
```

该检查同时保证：

- 文件已经打开
- 打开时使用了 `EDFLIB_READ_ALL_ANNOTATIONS`

项目内部使用 `std::int64_t` 保存 annotation 数量，但 `edf_get_annotation()` 使用 `int` 索引，因此读取前进行范围检查：

```cpp
if (header_.annotationCount >
    static_cast<std::int64_t>(
        std::numeric_limits<int>::max()))
{
    throw std::runtime_error(
        "EDF annotation count exceeds the "
        "EDFlib annotation index limit.");
}
```

逐条读取：

```cpp
for (int annotationIndex = 0;
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
}
```

原始字段转换：

```cpp
EdfAnnotation annotation;

annotation.onsetTicks =
    rawAnnotation.onset;

annotation.durationTicks =
    rawAnnotation.duration_l;

annotation.durationSecondsText =
    SafeCString(rawAnnotation.duration);

annotation.text =
    SafeCString(rawAnnotation.annotation);
```

---

## 9. annotation 关键词过滤

```cpp
std::vector<EdfAnnotation>
ReadAnnotationsByKeyword(
    const std::string &keyword) const;
```

处理流程：

1. 加载全部 annotation
2. 归一化关键词
3. 归一化每条 annotation 文本
4. 使用子串匹配筛选

```cpp
if (NormalizeLabel(annotation.text)
        .find(target) != std::string::npos)
{
    filtered.push_back(annotation);
}
```

匹配忽略大小写和多余空白。

---

## 10. Sleep-EDF 阶段解析

Sleep-EDF 常见 annotation：

```text
Sleep stage W
Sleep stage 1
Sleep stage 2
Sleep stage 3
Sleep stage 4
Sleep stage R
Sleep stage ?
Movement time
```

解析函数：

```cpp
bool TryParseSleepStage(
    const std::string &rawText,
    std::string *stageOut);
```

映射关系：

| 原始 annotation | 标准化结果 |
|---|---|
| `Sleep stage W` | `W` |
| `Sleep stage 1` / `Sleep stage N1` | `N1` |
| `Sleep stage 2` / `Sleep stage N2` | `N2` |
| `Sleep stage 3` | `N3` |
| `Sleep stage 4` | `N3` |
| `Sleep stage N3` | `N3` |
| `Sleep stage R` / `Sleep stage REM` | `REM` |
| `Sleep stage ?` | `UNKNOWN` |
| `Movement time` | `MOVEMENT` |

Stage 3 和 Stage 4 合并为 N3。

无法识别的 annotation 会被忽略，不会被强制映射为某个阶段。

---

## 11. 时间转换

EDFlib 使用：

```text
1 second = EDFLIB_TIME_DIMENSION ticks
```

转换函数：

```cpp
double TicksToSeconds(std::int64_t ticks)
{
    return static_cast<double>(ticks) /
           static_cast<double>(
               EDFLIB_TIME_DIMENSION);
}
```

annotation 持续时间优先解析原始字符串：

```cpp
const double parsed =
    std::strtod(value.c_str(), &endPointer);
```

如果字符串为空或无法解析，则使用 `durationTicks`。

如果两者都不存在，返回：

```text
-1.0
```

---

## 12. 错误处理

模块使用标准 C++ 异常区分错误类别。

| 异常 | 使用场景 |
|---|---|
| `std::invalid_argument` | 空路径、负数起始位置、负数读取数量 |
| `std::out_of_range` | 非法通道索引、超出信号范围、标签不存在 |
| `std::logic_error` | annotation 未加载却调用 annotation 接口 |
| `std::runtime_error` | 文件打开失败、底层读取失败、非法文件头 |

EDFlib 错误码被转换为可读文本：

```cpp
std::string EdfReader::BuildOpenErrorMessage(
    int edflibErrorCode);
```

示例：

```text
EDFLIB_NO_SUCH_FILE_OR_DIRECTORY
    -> file not found

EDFLIB_FILE_CONTAINS_FORMAT_ERRORS
    -> format error

EDFLIB_FILE_IS_DISCONTINUOUS
    -> discontinuous file
```

最终异常会带上文件路径：

```text
Failed to open EDF file: path/to/file.edf (file not found)
```

---

## 13. 线程安全

`ReadPhysicalSamples()` 在 C++ 接口上声明为 `const`，但内部调用：

```cpp
edfseek(...)
edfread_physical_samples(...)
```

会修改 EDFlib 为信号维护的内部读位置。

这里的 `const` 仅表示：

```text
不修改 EdfReader 的可见成员状态
```

不表示同一个实例可以被多个线程并发读取。

并行处理多个文件时，应为每个线程创建独立的 `EdfReader`。

---

## 14. CMake 集成

EDFlib 被编译为独立静态库：

```cmake
add_library(
    eeg_to_hypnogram_edflib
    STATIC
        edflib.c
        edflib.h
)
```

核心库将其作为私有依赖：

```cmake
target_link_libraries(
    eeg_to_hypnogram_core
    PRIVATE
        eeg_to_hypnogram::edflib
)
```

由于公共头文件不包含 EDFlib 类型，EDFlib 不需要成为公共依赖。

GNU/Clang 环境下启用大文件支持：

```cmake
target_compile_definitions(
    eeg_to_hypnogram_edflib
    PRIVATE
        _LARGEFILE64_SOURCE
        _LARGEFILE_SOURCE
)
```

同时设置：

```cmake
POSITION_INDEPENDENT_CODE ON
```

使静态库可以继续链接到共享库。

---

## 15. 使用示例

### 15.1 查看 EDF 通道

```cpp
#include "eeg_to_hypnogram/edf_reader.h"

#include <iostream>

int main()
{
    eeg_to_hypnogram::EdfReader reader;

    reader.Open("record-PSG.edf", true);

    const auto &header = reader.Header();

    std::cout
        << "duration: "
        << header.fileDurationSeconds
        << " seconds\n";

    for (std::size_t index = 0;
         index < header.signals.size();
         ++index)
    {
        const auto &signal =
            header.signals[index];

        std::cout
            << index
            << ": "
            << signal.label
            << ", "
            << signal.sampleRateHz
            << " Hz\n";
    }
}
```

### 15.2 读取 30 秒 EEG

```cpp
const auto signalIndex =
    reader.FindSignalIndexByLabel(
        "EEG Fpz-Cz");

if (!signalIndex.has_value())
{
    throw std::runtime_error(
        "EEG channel not found");
}

const int sampleRate = 100;
const int epochSeconds = 30;

const auto samples =
    reader.ReadPhysicalSamples(
        *signalIndex,
        0,
        sampleRate * epochSeconds);
```

100 Hz 下，30 秒对应 3000 个采样点。

### 15.3 读取睡眠阶段

```cpp
const auto stages =
    reader.ReadSleepStageAnnotations();

for (const auto &stage : stages)
{
    std::cout
        << stage.onsetSeconds
        << "s -> "
        << stage.stage
        << ", duration="
        << stage.durationSeconds
        << "s\n";
}
```

---

## 16. 测试覆盖

### 基础状态测试

不依赖真实 EDF 文件：

- 默认对象为关闭状态
- 未打开时访问头信息抛异常
- 空路径被拒绝
- 不存在文件时打开失败
- 打开失败后对象保持关闭状态
- `Close()` 可重复调用

### 真实数据集成测试

通过环境变量传入 PSG 和 Hypnogram 文件：

```bash
EEG_TEST_EDF_FILE=".../SC4001E0-PSG.edf" \
EEG_TEST_HYPNOGRAM_FILE=".../SC4001EC-Hypnogram.edf" \
./edf_reader_test
```

覆盖：

- PSG 文件打开
- 文件头和通道元数据读取
- 通道标签精确查找
- 物理值窗口读取
- 移动构造转移句柄
- annotation 加载状态检查
- Hypnogram annotation 读取
- Sleep-EDF 阶段解析

测试结果：

```text
annotations=154
stages=154
```

表示测试文件中的 154 条 annotation 全部成功解析为阶段记录。

---

## 17. 接口限制

### 17.1 整通道读取数量限制

EDFlib 的单次读取数量参数为 `int`，因此整通道读取不能超过：

```cpp
std::numeric_limits<int>::max()
```

超大信号应分窗口读取。

### 17.2 同一实例不支持并发读取

EDFlib 内部维护信号读位置，同一个实例不应同时执行多个读取操作。

### 17.3 路径必须在当前文件系统可访问

当前接口：

```cpp
void Open(const std::string &filePath);
```

要求文件路径可由当前运行环境访问。

### 17.4 annotation 模式不能动态切换

如果文件通过以下方式打开：

```cpp
reader.Open(path, false);
```

需要重新调用：

```cpp
reader.Open(path, true);
```

才能读取 annotation。

---

## 18. 关键实现要点

- 使用 RAII 自动释放 EDFlib 文件句柄
- 禁止复制并支持安全移动
- 使用临时句柄保护器保证 `Open()` 的强异常安全
- 将 EDFlib 类型限制在实现文件中
- 使用内部 C++ 数据结构隔离第三方 API
- 同时保留 tick 和秒两种时间表示
- 通过窗口读取控制内存占用
- 使用标签归一化提高通道匹配稳定性
- 将 Sleep-EDF Stage 3 和 Stage 4 统一为 N3
- 区分“没有 annotation”和“没有加载 annotation”
- 对 EDFlib 的 `int` 参数进行范围检查
- 使用真实 PSG 和 Hypnogram 文件验证读取与解析结果