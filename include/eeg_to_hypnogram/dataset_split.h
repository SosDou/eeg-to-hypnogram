#pragma once

#include "eeg_to_hypnogram/dataset_manifest.h"

#include <cstdint>
#include <string>
#include <vector>

namespace eeg_to_hypnogram
{

    // 用于清单分区的受试者级划分配置。
    struct DatasetSplitConfig
    {
        double testFraction = 0.2;
        std::uint32_t randomSeed = 42;
    };

    // 带有受试者归属元数据的 train/test 清单划分结果。
    struct DatasetManifestSplit
    {
        DatasetManifest train;
        DatasetManifest test;

        std::vector<std::string> trainSubjectIds;
        std::vector<std::string> testSubjectIds;
    };

    // 按 subjectId 划分一个干净的 DatasetManifest。
    // 来自同一受试者的所有 pair 都会保留在同一个输出清单中。
    DatasetManifestSplit SplitDatasetManifestBySubject(
        const DatasetManifest &manifest,
        const DatasetSplitConfig &config = {});

} // namespace eeg_to_hypnogram
