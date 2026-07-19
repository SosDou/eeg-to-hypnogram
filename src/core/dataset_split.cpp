#include "eeg_to_hypnogram/dataset_split.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace
{

    std::size_t BoundedRandom(
        std::mt19937 *generator,
        std::size_t upperExclusive)
    {
        if (upperExclusive == 0)
        {
            throw std::logic_error(
                "BoundedRandom requires a positive bound.");
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
        std::uint32_t seed)
    {
        std::mt19937 generator(seed);
        for (std::size_t size = values->size(); size > 1; --size)
        {
            const std::size_t swapIndex =
                BoundedRandom(&generator, size);
            std::swap((*values)[size - 1], (*values)[swapIndex]);
        }
    }

    void ValidateSplitConfig(
        const eeg_to_hypnogram::DatasetSplitConfig &config)
    {
        if (!std::isfinite(config.testFraction))
        {
            throw std::invalid_argument(
                "testFraction must be finite.");
        }

        if (config.testFraction <= 0.0)
        {
            throw std::invalid_argument(
                "testFraction must be > 0.");
        }

        if (config.testFraction >= 1.0)
        {
            throw std::invalid_argument(
                "testFraction must be < 1.");
        }
    }

    std::size_t ComputeTestSubjectCount(
        std::size_t subjectCount,
        double testFraction)
    {
        const double exact =
            static_cast<double>(subjectCount) * testFraction;

        std::size_t testCount =
            static_cast<std::size_t>(std::round(exact));

        if (testCount == 0)
        {
            testCount = 1;
        }
        if (testCount >= subjectCount)
        {
            testCount = subjectCount - 1;
        }

        return testCount;
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

    std::vector<std::string> CollectUniqueSubjects(
        const eeg_to_hypnogram::DatasetManifest &manifest)
    {
        std::vector<std::string> subjects;
        subjects.reserve(manifest.pairs.size());

        for (const auto &pair : manifest.pairs)
        {
            ValidatePairFields(pair);
            subjects.push_back(pair.subjectId);
        }

        std::sort(subjects.begin(), subjects.end());
        subjects.erase(
            std::unique(subjects.begin(), subjects.end()),
            subjects.end());
        return subjects;
    }

    std::unordered_set<std::string> MakeSubjectSet(
        const std::vector<std::string> &subjects)
    {
        return std::unordered_set<std::string>(
            subjects.begin(),
            subjects.end());
    }

    void ValidateSplitResult(
        const eeg_to_hypnogram::DatasetManifestSplit &split,
        std::size_t expectedPairCount)
    {
        const std::unordered_set<std::string> trainSubjects =
            MakeSubjectSet(split.trainSubjectIds);
        const std::unordered_set<std::string> testSubjects =
            MakeSubjectSet(split.testSubjectIds);

        if (trainSubjects.size() != split.trainSubjectIds.size())
        {
            throw std::logic_error(
                "Train subject list contains duplicates.");
        }
        if (testSubjects.size() != split.testSubjectIds.size())
        {
            throw std::logic_error(
                "Test subject list contains duplicates.");
        }

        for (const auto &subject : trainSubjects)
        {
            if (testSubjects.count(subject) != 0)
            {
                throw std::logic_error(
                    "Subject appears in both train and test splits: " +
                    subject);
            }
        }

        if (split.train.pairs.size() + split.test.pairs.size() !=
            expectedPairCount)
        {
            throw std::logic_error(
                "Split pair counts do not match the source manifest.");
        }

        for (const auto &pair : split.train.pairs)
        {
            if (trainSubjects.count(pair.subjectId) == 0)
            {
                throw std::logic_error(
                    "Train split contains an unexpected subject: " +
                    pair.subjectId);
            }
        }

        for (const auto &pair : split.test.pairs)
        {
            if (testSubjects.count(pair.subjectId) == 0)
            {
                throw std::logic_error(
                    "Test split contains an unexpected subject: " +
                    pair.subjectId);
            }
        }
    }

} // namespace

namespace eeg_to_hypnogram
{

    DatasetManifestSplit SplitDatasetManifestBySubject(
        const DatasetManifest &manifest,
        const DatasetSplitConfig &config)
    {
        ValidateManifestForDatasetAssembly(manifest);
        ValidateSplitConfig(config);

        const std::vector<std::string> subjects =
            CollectUniqueSubjects(manifest);

        if (subjects.size() < 2)
        {
            throw std::invalid_argument(
                "At least two subjects are required for a subject-level split.");
        }

        std::vector<std::string> shuffledSubjects = subjects;
        StableShuffle(&shuffledSubjects, config.randomSeed);

        const std::size_t testCount =
            ComputeTestSubjectCount(
                shuffledSubjects.size(),
                config.testFraction);

        DatasetManifestSplit split;
        split.testSubjectIds.assign(
            shuffledSubjects.begin(),
            shuffledSubjects.begin() +
                static_cast<std::ptrdiff_t>(testCount));
        split.trainSubjectIds.assign(
            shuffledSubjects.begin() +
                static_cast<std::ptrdiff_t>(testCount),
            shuffledSubjects.end());

        const std::unordered_set<std::string> testSubjects =
            MakeSubjectSet(split.testSubjectIds);

        split.train.pairs.reserve(manifest.pairs.size());
        split.test.pairs.reserve(manifest.pairs.size());

        for (const auto &pair : manifest.pairs)
        {
            if (testSubjects.count(pair.subjectId) != 0)
            {
                split.test.pairs.push_back(pair);
            }
            else
            {
                split.train.pairs.push_back(pair);
            }
        }

        ValidateSplitResult(split, manifest.pairs.size());
        return split;
    }

} // namespace eeg_to_hypnogram
