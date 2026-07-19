#pragma once

#include <optional>
#include <string>
#include <vector>

namespace eeg_to_hypnogram
{

    // 数据目录扫描选项。初版只扫描一层目录，除非显式指定，否则不跟随符号链接。
    struct DatasetManifestScanConfig
    {
        bool recursive = false;
        bool followDirectorySymlinks = false;
    };

    // 来自一条 Sleep-EDF PSG 或 Hypnogram 文件名的解析信息。
    struct SleepEdfFileInfo
    {
        std::string path;
        std::string filename;

        // 稳定的受试者键，例如 SC400 或 ST701。
        std::string subjectId;

        // 稳定的 PSG/Hypnogram 配对键：Sleep-EDF 录制编码的前七个字符，
        // 例如 SC4001E 或 ST7011J。
        std::string recordingId;

        // 录制编码第六个字符里的夜次标识。
        std::string nightId;

        bool isPsg = false;
        bool isHypnogram = false;
    };

    // 一个无歧义的 PSG/Hypnogram 配对。
    struct SleepEdfFilePair
    {
        std::string subjectId;
        std::string recordingId;
        std::string nightId;
        std::string psgPath;
        std::string hypnogramPath;
    };

    // 扫描和配对数据集后的稳定、便于诊断的结果。
    struct DatasetManifest
    {
        std::vector<SleepEdfFilePair> pairs;

        std::vector<std::string> unmatchedPsgFiles;
        std::vector<std::string> unmatchedHypnogramFiles;

        // 重复键会从 pairs 中排除。调用方必须自行消除歧义，不能依赖目录枚举顺序。
        std::vector<std::string> duplicatePsgKeys;
        std::vector<std::string> duplicateHypnogramKeys;

        // 传入 BuildDatasetManifestFromFiles 的精确路径重复项。
        std::vector<std::string> duplicateInputPaths;

        // 不是 EDF 文件的普通文件。
        std::vector<std::string> ignoredFiles;

        // 文件名不符合受支持 Sleep-EDF SC/ST 规则的 EDF 文件。
        std::vector<std::string> unrecognizedEdfFiles;
    };

    struct SubjectSplitConfig
    {
        double trainRatio = 0.70;
        double validationRatio = 0.15;
        double testRatio = 0.15;
        unsigned int seed = 42;
    };

    struct DatasetSplit
    {
        std::vector<SleepEdfFilePair> train;
        std::vector<SleepEdfFilePair> validation;
        std::vector<SleepEdfFilePair> test;
    };

    // 解析一条路径，不打开 EDF 文件。匹配不区分大小写。
    // 示例：
    //   示例文件：SC4001E0-PSG.edf
    //   示例文件：SC4001EC-Hypnogram.edf
    //   示例文件：ST7011J0-PSG.edf
    //   示例文件：ST7011JP-Hypnogram.edf
    std::optional<SleepEdfFileInfo> ParseSleepEdfFilename(
        const std::string &path);

    // 从显式路径列表构建清单。路径不必真实存在；
    // 这个重载适合确定性测试，以及已经完成发现的调用方。
    DatasetManifest BuildDatasetManifestFromFiles(
        const std::vector<std::string> &paths);

    // 扫描调用方提供的目录，发现普通文件，并把文件名解析和配对工作
    // 委托给 BuildDatasetManifestFromFiles。
    DatasetManifest BuildDatasetManifest(
        const std::string &datasetDirectory,
        const DatasetManifestScanConfig &config = {});

    // 验证清单是否足够干净，可供下游做配对组装或受试者级划分。允许保留 ignoredFiles。
    void ValidateManifestForDatasetAssembly(const DatasetManifest &manifest);

    // 按唯一 subjectId 划分完整文件对。来自同一受试者的所有记录都会留在同一个 split 中。
    // 计数采用最大余数法；每个正比例的 split 都保证至少有一个受试者。
    DatasetSplit SplitDatasetBySubject(
        const std::vector<SleepEdfFilePair> &pairs,
        const SubjectSplitConfig &config = {});

    // 当受试者、recording 身份、PSG 路径或 Hypnogram 路径在不同 split 间重叠
    // （或在同一 split 内重复）时抛出 std::invalid_argument。
    void ValidateSubjectDisjointSplit(const DatasetSplit &split);

} // namespace eeg_to_hypnogram
