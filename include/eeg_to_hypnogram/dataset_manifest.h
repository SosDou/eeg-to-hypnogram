#pragma once

#include <optional>
#include <string>
#include <vector>

namespace eeg_to_hypnogram
{

// Dataset directory scanning options. The first version scans one directory
// level and does not follow symbolic links unless explicitly requested.
struct DatasetManifestScanConfig
{
    bool recursive = false;
    bool followDirectorySymlinks = false;
};

// Parsed information from one Sleep-EDF PSG or Hypnogram filename.
struct SleepEdfFileInfo
{
    std::string path;
    std::string filename;

    // Stable subject key, for example SC400 or ST701.
    std::string subjectId;

    // Stable PSG/Hypnogram pairing key: the first seven characters of the
    // Sleep-EDF recording code, for example SC4001E or ST7011J.
    std::string recordingId;

    // Night identifier from the sixth character of the recording code.
    std::string nightId;

    bool isPsg = false;
    bool isHypnogram = false;
};

// One unambiguous PSG/Hypnogram pair.
struct SleepEdfFilePair
{
    std::string subjectId;
    std::string recordingId;
    std::string nightId;
    std::string psgPath;
    std::string hypnogramPath;
};

// Stable, diagnostic-rich result of scanning and pairing a dataset.
struct DatasetManifest
{
    std::vector<SleepEdfFilePair> pairs;

    std::vector<std::string> unmatchedPsgFiles;
    std::vector<std::string> unmatchedHypnogramFiles;

    // A duplicated key is excluded from pairs. The caller must resolve the
    // ambiguity rather than relying on directory enumeration order.
    std::vector<std::string> duplicatePsgKeys;
    std::vector<std::string> duplicateHypnogramKeys;

    // Exact path duplicates supplied to BuildDatasetManifestFromFiles.
    std::vector<std::string> duplicateInputPaths;

    // Regular files that are not EDF files.
    std::vector<std::string> ignoredFiles;

    // EDF files whose names do not match the supported Sleep-EDF SC/ST rules.
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

// Parses one path without opening the EDF file. The match is case-insensitive.
// Supported examples:
//   SC4001E0-PSG.edf
//   SC4001EC-Hypnogram.edf
//   ST7011J0-PSG.edf
//   ST7011JP-Hypnogram.edf
std::optional<SleepEdfFileInfo> ParseSleepEdfFilename(
    const std::string &path);

// Builds a manifest from an explicit list of paths. Paths do not need to exist;
// this overload is useful for deterministic tests and callers that already
// performed discovery.
DatasetManifest BuildDatasetManifestFromFiles(
    const std::vector<std::string> &paths);

// Scans the caller-supplied directory, discovers regular files, and delegates
// filename parsing and pairing to BuildDatasetManifestFromFiles.
DatasetManifest BuildDatasetManifest(
    const std::string &datasetDirectory,
    const DatasetManifestScanConfig &config = {});

// Splits complete file pairs by unique subjectId. All recordings from one
// subject remain in one split. Counts are determined by largest remainder;
// every split with a positive ratio is guaranteed at least one subject.
DatasetSplit SplitDatasetBySubject(
    const std::vector<SleepEdfFilePair> &pairs,
    const SubjectSplitConfig &config = {});

// Throws std::invalid_argument when subjects, recording identities, PSG paths,
// or Hypnogram paths overlap between splits (or are duplicated within a split).
void ValidateSubjectDisjointSplit(const DatasetSplit &split);

} // namespace eeg_to_hypnogram
