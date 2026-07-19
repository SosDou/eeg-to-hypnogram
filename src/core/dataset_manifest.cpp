#include "eeg_to_hypnogram/dataset_manifest.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <random>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace
{

    namespace fs = std::filesystem;

    constexpr double kRatioTolerance = 1e-9;

    std::string ToUpperAscii(std::string text)
    {
        std::transform(
            text.begin(),
            text.end(),
            text.begin(),
            [](unsigned char value)
            {
                if (value >= static_cast<unsigned char>('a') &&
                    value <= static_cast<unsigned char>('z'))
                {
                    return static_cast<char>(value - 'a' + 'A');
                }
                return static_cast<char>(value);
            });
        return text;
    }

    std::string PortableFilename(const std::string &path)
    {
        const std::size_t separator = path.find_last_of("/\\");
        return separator == std::string::npos
                   ? path
                   : path.substr(separator + 1);
    }

    std::string NormalizePathText(const std::string &path)
    {
        if (path.empty())
        {
            return path;
        }

        std::string portable = path;
#ifdef _WIN32
        std::replace(portable.begin(), portable.end(), '/', '\\');
#else
        std::replace(portable.begin(), portable.end(), '\\', '/');
#endif
        return fs::path(portable).lexically_normal().generic_string();
    }

    std::string NormalizeScannedPath(const fs::path &path)
    {
        std::error_code error;
        const fs::path absolutePath = fs::absolute(path, error);
        if (error)
        {
            throw std::runtime_error(
                "Failed to make dataset path absolute: " +
                path.generic_string() + ": " + error.message());
        }
        return absolutePath.lexically_normal().generic_string();
    }

    bool HasEdfExtension(const std::string &filename)
    {
        const std::string upper = ToUpperAscii(filename);
        return upper.size() >= 4 &&
               upper.compare(upper.size() - 4, 4, ".EDF") == 0;
    }

    struct FileGroup
    {
        std::string subjectId;
        std::string recordingId;
        std::string nightId;
        std::vector<std::string> psgPaths;
        std::vector<std::string> hypnogramPaths;
    };

    bool PairLess(
        const eeg_to_hypnogram::SleepEdfFilePair &left,
        const eeg_to_hypnogram::SleepEdfFilePair &right)
    {
        if (left.subjectId != right.subjectId)
        {
            return left.subjectId < right.subjectId;
        }
        if (left.recordingId != right.recordingId)
        {
            return left.recordingId < right.recordingId;
        }
        if (left.psgPath != right.psgPath)
        {
            return left.psgPath < right.psgPath;
        }
        return left.hypnogramPath < right.hypnogramPath;
    }

    void SortUnique(std::vector<std::string> *values)
    {
        std::sort(values->begin(), values->end());
        values->erase(
            std::unique(values->begin(), values->end()),
            values->end());
    }

    void ValidateSplitConfig(
        const eeg_to_hypnogram::SubjectSplitConfig &config)
    {
        const std::array<double, 3> ratios = {
            config.trainRatio,
            config.validationRatio,
            config.testRatio,
        };

        for (double ratio : ratios)
        {
            if (!std::isfinite(ratio))
            {
                throw std::invalid_argument("Split ratios must be finite.");
            }
        }

        if (config.trainRatio <= 0.0)
        {
            throw std::invalid_argument("trainRatio must be > 0.");
        }
        if (config.validationRatio < 0.0)
        {
            throw std::invalid_argument("validationRatio must be >= 0.");
        }
        if (config.testRatio <= 0.0)
        {
            throw std::invalid_argument("testRatio must be > 0.");
        }

        const double sum =
            config.trainRatio +
            config.validationRatio +
            config.testRatio;

        if (std::abs(sum - 1.0) > kRatioTolerance)
        {
            throw std::invalid_argument(
                "trainRatio + validationRatio + testRatio must equal 1.");
        }
    }

    std::size_t BoundedRandom(
        std::mt19937 *generator,
        std::size_t upperExclusive)
    {
        if (upperExclusive == 0)
        {
            throw std::logic_error("BoundedRandom requires a positive bound.");
        }

        constexpr std::uint64_t range =
            static_cast<std::uint64_t>(
                std::numeric_limits<std::uint32_t>::max()) +
            1ULL;

        const std::uint64_t bound =
            static_cast<std::uint64_t>(upperExclusive);
        const std::uint64_t limit = range - (range % bound);

        std::uint64_t value = 0;
        do
        {
            value = static_cast<std::uint64_t>((*generator)());
        } while (value >= limit);

        return static_cast<std::size_t>(value % bound);
    }

    void StableShuffle(
        std::vector<std::string> *values,
        unsigned int seed)
    {
        std::mt19937 generator(seed);
        for (std::size_t size = values->size(); size > 1; --size)
        {
            const std::size_t swapIndex =
                BoundedRandom(&generator, size);
            std::swap((*values)[size - 1], (*values)[swapIndex]);
        }
    }

    std::array<std::size_t, 3> AllocateSubjectCounts(
        std::size_t subjectCount,
        const eeg_to_hypnogram::SubjectSplitConfig &config)
    {
        const std::array<double, 3> ratios = {
            config.trainRatio,
            config.validationRatio,
            config.testRatio,
        };

        std::array<std::size_t, 3> counts = {0, 0, 0};
        std::array<double, 3> remainders = {0.0, 0.0, 0.0};

        std::size_t assigned = 0;
        for (std::size_t index = 0; index < ratios.size(); ++index)
        {
            const double exact =
                static_cast<double>(subjectCount) * ratios[index];
            counts[index] = static_cast<std::size_t>(std::floor(exact));
            remainders[index] = exact - static_cast<double>(counts[index]);
            assigned += counts[index];
        }

        std::array<std::size_t, 3> order = {0, 1, 2};
        std::stable_sort(
            order.begin(),
            order.end(),
            [&](std::size_t left, std::size_t right)
            {
                return remainders[left] > remainders[right];
            });

        for (std::size_t remaining = subjectCount - assigned;
             remaining > 0;
             --remaining)
        {
            const std::size_t destination =
                order[(subjectCount - assigned - remaining) % order.size()];
            ++counts[destination];
        }

        // A positive-ratio split must not be empty. If largest remainder left one
        // empty, deterministically borrow from the largest split that has > 1.
        for (std::size_t target = 0; target < ratios.size(); ++target)
        {
            if (ratios[target] <= 0.0 || counts[target] > 0)
            {
                continue;
            }

            std::size_t donor = ratios.size();
            for (std::size_t candidate = 0;
                 candidate < ratios.size();
                 ++candidate)
            {
                if (counts[candidate] <= 1)
                {
                    continue;
                }
                if (donor == ratios.size() ||
                    counts[candidate] > counts[donor] ||
                    (counts[candidate] == counts[donor] &&
                     ratios[candidate] > ratios[donor]))
                {
                    donor = candidate;
                }
            }

            if (donor == ratios.size())
            {
                throw std::invalid_argument(
                    "Too few subjects to keep every positive-ratio split non-empty.");
            }

            --counts[donor];
            ++counts[target];
        }

        return counts;
    }

    std::string NormalizedValidationPath(const std::string &path)
    {
        std::string normalized = NormalizePathText(path);
#ifdef _WIN32
        normalized = ToUpperAscii(normalized);
#endif
        return normalized;
    }

    void ValidatePairFields(
        const eeg_to_hypnogram::SleepEdfFilePair &pair)
    {
        if (pair.subjectId.empty() ||
            pair.recordingId.empty() ||
            pair.psgPath.empty() ||
            pair.hypnogramPath.empty())
        {
            throw std::invalid_argument(
                "Every split pair must contain subjectId, recordingId, PSG path, and Hypnogram path.");
        }
    }

    void ThrowIfManifestFieldNotEmpty(
        const std::vector<std::string> &values,
        const std::string &category)
    {
        if (!values.empty())
        {
            throw std::invalid_argument(
                "dataset manifest contains " + category + ": " +
                std::to_string(values.size()));
        }
    }

    void ValidateManifestForDatasetAssemblyImpl(
        const eeg_to_hypnogram::DatasetManifest &manifest)
    {
        if (manifest.pairs.empty())
        {
            throw std::invalid_argument(
                "dataset manifest contains complete pairs: 0");
        }

        ThrowIfManifestFieldNotEmpty(
            manifest.unmatchedPsgFiles,
            "unmatched PSG files");
        ThrowIfManifestFieldNotEmpty(
            manifest.unmatchedHypnogramFiles,
            "unmatched Hypnogram files");
        ThrowIfManifestFieldNotEmpty(
            manifest.duplicatePsgKeys,
            "duplicate PSG keys");
        ThrowIfManifestFieldNotEmpty(
            manifest.duplicateHypnogramKeys,
            "duplicate Hypnogram keys");
        ThrowIfManifestFieldNotEmpty(
            manifest.duplicateInputPaths,
            "duplicate input paths");
        ThrowIfManifestFieldNotEmpty(
            manifest.unrecognizedEdfFiles,
            "unrecognized EDF files");
    }

} // namespace

namespace eeg_to_hypnogram
{

    std::optional<SleepEdfFileInfo> ParseSleepEdfFilename(
        const std::string &path)
    {
        const std::string filename = PortableFilename(path);

        // Recording code layout used by both Sleep-EDF cohorts:
        //   SC + 3 digits + night + acquisition letter + variant/scorer
        //   ST + 3 digits + night + acquisition letter + variant/scorer
        // PSG files use variant 0. Hypnogram files use the scorer identifier.
        static const std::regex pattern(
            R"(^(SC|ST)([0-9]{3})([12])([A-Z])([A-Z0-9])-(PSG|HYPNOGRAM)\.EDF$)",
            std::regex::icase);

        std::smatch match;
        if (!std::regex_match(filename, match, pattern))
        {
            return std::nullopt;
        }

        const std::string cohort = ToUpperAscii(match[1].str());
        const std::string subjectCode = match[2].str();
        const std::string night = match[3].str();
        const std::string acquisition = ToUpperAscii(match[4].str());
        const std::string variant = ToUpperAscii(match[5].str());
        const std::string kind = ToUpperAscii(match[6].str());

        const bool isPsg = kind == "PSG";
        const bool isHypnogram = kind == "HYPNOGRAM";

        // Official Sleep-EDF PSG names use 0 as the eighth recording character.
        if (isPsg && variant != "0")
        {
            return std::nullopt;
        }

        SleepEdfFileInfo info;
        info.path = path;
        info.filename = filename;
        info.subjectId = cohort + subjectCode;
        info.recordingId =
            info.subjectId + night + acquisition;
        info.nightId = night;
        info.isPsg = isPsg;
        info.isHypnogram = isHypnogram;
        return info;
    }

    DatasetManifest BuildDatasetManifestFromFiles(
        const std::vector<std::string> &paths)
    {
        DatasetManifest manifest;

        std::vector<std::string> normalizedPaths;
        normalizedPaths.reserve(paths.size());
        for (const std::string &path : paths)
        {
            normalizedPaths.push_back(NormalizePathText(path));
        }
        std::sort(normalizedPaths.begin(), normalizedPaths.end());

        std::vector<std::string> uniquePaths;
        uniquePaths.reserve(normalizedPaths.size());
        for (const std::string &path : normalizedPaths)
        {
            if (!uniquePaths.empty() && uniquePaths.back() == path)
            {
                manifest.duplicateInputPaths.push_back(path);
                continue;
            }
            uniquePaths.push_back(path);
        }

        std::map<std::string, FileGroup> groups;

        for (const std::string &path : uniquePaths)
        {
            const std::string filename = PortableFilename(path);
            if (!HasEdfExtension(filename))
            {
                manifest.ignoredFiles.push_back(path);
                continue;
            }

            const auto parsed = ParseSleepEdfFilename(path);
            if (!parsed.has_value())
            {
                manifest.unrecognizedEdfFiles.push_back(path);
                continue;
            }

            FileGroup &group = groups[parsed->recordingId];
            group.subjectId = parsed->subjectId;
            group.recordingId = parsed->recordingId;
            group.nightId = parsed->nightId;

            if (parsed->isPsg)
            {
                group.psgPaths.push_back(path);
            }
            else
            {
                group.hypnogramPaths.push_back(path);
            }
        }

        for (auto &entry : groups)
        {
            FileGroup &group = entry.second;
            std::sort(group.psgPaths.begin(), group.psgPaths.end());
            std::sort(
                group.hypnogramPaths.begin(),
                group.hypnogramPaths.end());

            const bool duplicatePsg = group.psgPaths.size() > 1;
            const bool duplicateHypnogram =
                group.hypnogramPaths.size() > 1;

            if (duplicatePsg)
            {
                manifest.duplicatePsgKeys.push_back(group.recordingId);
            }
            if (duplicateHypnogram)
            {
                manifest.duplicateHypnogramKeys.push_back(group.recordingId);
            }

            if (duplicatePsg || duplicateHypnogram)
            {
                continue;
            }

            if (group.psgPaths.size() == 1 &&
                group.hypnogramPaths.size() == 1)
            {
                manifest.pairs.push_back(
                    {
                        group.subjectId,
                        group.recordingId,
                        group.nightId,
                        group.psgPaths.front(),
                        group.hypnogramPaths.front(),
                    });
            }
            else if (group.psgPaths.size() == 1)
            {
                manifest.unmatchedPsgFiles.push_back(
                    group.psgPaths.front());
            }
            else if (group.hypnogramPaths.size() == 1)
            {
                manifest.unmatchedHypnogramFiles.push_back(
                    group.hypnogramPaths.front());
            }
        }

        std::sort(manifest.pairs.begin(), manifest.pairs.end(), PairLess);
        SortUnique(&manifest.unmatchedPsgFiles);
        SortUnique(&manifest.unmatchedHypnogramFiles);
        SortUnique(&manifest.duplicatePsgKeys);
        SortUnique(&manifest.duplicateHypnogramKeys);
        SortUnique(&manifest.duplicateInputPaths);
        SortUnique(&manifest.ignoredFiles);
        SortUnique(&manifest.unrecognizedEdfFiles);

        return manifest;
    }

    DatasetManifest BuildDatasetManifest(
        const std::string &datasetDirectory,
        const DatasetManifestScanConfig &config)
    {
        if (datasetDirectory.empty())
        {
            throw std::invalid_argument(
                "Dataset directory path must not be empty.");
        }

        const fs::path root(datasetDirectory);
        std::error_code error;

        const bool exists = fs::exists(root, error);
        if (error)
        {
            throw std::runtime_error(
                "Failed to inspect dataset directory: " + error.message());
        }
        if (!exists)
        {
            throw std::invalid_argument(
                "Dataset directory does not exist: " + datasetDirectory);
        }

        const bool isDirectory = fs::is_directory(root, error);
        if (error)
        {
            throw std::runtime_error(
                "Failed to inspect dataset directory type: " + error.message());
        }
        if (!isDirectory)
        {
            throw std::invalid_argument(
                "Dataset path is not a directory: " + datasetDirectory);
        }

        std::vector<std::string> files;

        const auto collectEntry =
            [&](const fs::directory_entry &entry)
        {
            std::error_code entryError;
            const bool isSymlink = entry.is_symlink(entryError);
            if (entryError)
            {
                throw std::runtime_error(
                    "Failed to inspect dataset entry: " +
                    entry.path().generic_string() + ": " +
                    entryError.message());
            }

            if (isSymlink && !config.followDirectorySymlinks)
            {
                return;
            }

            const bool isRegular = entry.is_regular_file(entryError);
            if (entryError)
            {
                throw std::runtime_error(
                    "Failed to inspect dataset entry type: " +
                    entry.path().generic_string() + ": " +
                    entryError.message());
            }

            if (isRegular)
            {
                files.push_back(NormalizeScannedPath(entry.path()));
            }
        };

        if (config.recursive)
        {
            fs::directory_options options =
                fs::directory_options::skip_permission_denied;
            if (config.followDirectorySymlinks)
            {
                options |= fs::directory_options::follow_directory_symlink;
            }

            for (const fs::directory_entry &entry :
                 fs::recursive_directory_iterator(root, options))
            {
                collectEntry(entry);
            }
        }
        else
        {
            for (const fs::directory_entry &entry :
                 fs::directory_iterator(
                     root,
                     fs::directory_options::skip_permission_denied))
            {
                collectEntry(entry);
            }
        }

        return BuildDatasetManifestFromFiles(files);
    }

    void ValidateManifestForDatasetAssembly(
        const DatasetManifest &manifest)
    {
        ValidateManifestForDatasetAssemblyImpl(manifest);
    }

    DatasetSplit SplitDatasetBySubject(
        const std::vector<SleepEdfFilePair> &pairs,
        const SubjectSplitConfig &config)
    {
        ValidateSplitConfig(config);

        if (pairs.empty())
        {
            throw std::invalid_argument(
                "Cannot split an empty list of Sleep-EDF pairs.");
        }

        std::vector<SleepEdfFilePair> stablePairs = pairs;
        for (const SleepEdfFilePair &pair : stablePairs)
        {
            ValidatePairFields(pair);
        }
        std::sort(stablePairs.begin(), stablePairs.end(), PairLess);

        std::vector<std::string> subjects;
        subjects.reserve(stablePairs.size());
        for (const SleepEdfFilePair &pair : stablePairs)
        {
            subjects.push_back(pair.subjectId);
        }
        SortUnique(&subjects);

        const std::size_t requiredSplits =
            config.validationRatio > 0.0 ? 3U : 2U;
        if (subjects.size() < requiredSplits)
        {
            throw std::invalid_argument(
                "Too few subjects for the requested non-empty splits.");
        }

        StableShuffle(&subjects, config.seed);
        const std::array<std::size_t, 3> counts =
            AllocateSubjectCounts(subjects.size(), config);

        std::unordered_map<std::string, int> destinationBySubject;
        destinationBySubject.reserve(subjects.size());

        std::size_t offset = 0;
        for (std::size_t index = 0; index < counts[0]; ++index)
        {
            destinationBySubject.emplace(subjects[offset++], 0);
        }
        for (std::size_t index = 0; index < counts[1]; ++index)
        {
            destinationBySubject.emplace(subjects[offset++], 1);
        }
        for (std::size_t index = 0; index < counts[2]; ++index)
        {
            destinationBySubject.emplace(subjects[offset++], 2);
        }

        DatasetSplit split;
        for (const SleepEdfFilePair &pair : stablePairs)
        {
            const auto found = destinationBySubject.find(pair.subjectId);
            if (found == destinationBySubject.end())
            {
                throw std::logic_error(
                    "Subject allocation is internally inconsistent.");
            }

            switch (found->second)
            {
            case 0:
                split.train.push_back(pair);
                break;
            case 1:
                split.validation.push_back(pair);
                break;
            case 2:
                split.test.push_back(pair);
                break;
            default:
                throw std::logic_error("Unexpected split destination.");
            }
        }

        ValidateSubjectDisjointSplit(split);
        return split;
    }

    void ValidateSubjectDisjointSplit(const DatasetSplit &split)
    {
        std::unordered_map<std::string, int> subjectOwner;
        std::unordered_map<std::string, int> recordingOwner;
        std::unordered_map<std::string, int> pathOwner;

        const std::array<const std::vector<SleepEdfFilePair> *, 3> allSplits = {
            &split.train,
            &split.validation,
            &split.test,
        };

        for (std::size_t splitIndex = 0;
             splitIndex < allSplits.size();
             ++splitIndex)
        {
            for (const SleepEdfFilePair &pair : *allSplits[splitIndex])
            {
                ValidatePairFields(pair);
                const int owner = static_cast<int>(splitIndex);

                const auto subjectResult =
                    subjectOwner.emplace(pair.subjectId, owner);
                if (!subjectResult.second &&
                    subjectResult.first->second != owner)
                {
                    throw std::invalid_argument(
                        "Subject appears in more than one split: " +
                        pair.subjectId);
                }

                const std::string recordingKey =
                    pair.subjectId + "\n" + pair.recordingId;
                const auto recordingResult =
                    recordingOwner.emplace(recordingKey, owner);
                if (!recordingResult.second)
                {
                    throw std::invalid_argument(
                        "Recording appears more than once in the dataset split: " +
                        pair.recordingId);
                }

                const std::array<std::string, 2> paths = {
                    NormalizedValidationPath(pair.psgPath),
                    NormalizedValidationPath(pair.hypnogramPath),
                };

                for (const std::string &path : paths)
                {
                    const auto pathResult = pathOwner.emplace(path, owner);
                    if (!pathResult.second)
                    {
                        throw std::invalid_argument(
                            "EDF path appears more than once in the dataset split: " +
                            path);
                    }
                }
            }
        }
    }

} // namespace eeg_to_hypnogram
