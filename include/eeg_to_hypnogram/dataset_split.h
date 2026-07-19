#pragma once

#include "eeg_to_hypnogram/dataset_manifest.h"

#include <cstdint>
#include <string>
#include <vector>

namespace eeg_to_hypnogram
{

    // Subject-level split configuration for manifest partitioning.
    struct DatasetSplitConfig
    {
        double testFraction = 0.2;
        std::uint32_t randomSeed = 42;
    };

    // Train/test manifest split with subject membership metadata.
    struct DatasetManifestSplit
    {
        DatasetManifest train;
        DatasetManifest test;

        std::vector<std::string> trainSubjectIds;
        std::vector<std::string> testSubjectIds;
    };

    // Splits a clean DatasetManifest by subjectId.
    // All pairs from one subject remain in the same output manifest.
    DatasetManifestSplit SplitDatasetManifestBySubject(
        const DatasetManifest &manifest,
        const DatasetSplitConfig &config = {});

} // namespace eeg_to_hypnogram
