#include "eeg_to_hypnogram/dataset_manifest.h"
#include "eeg_to_hypnogram/dataset_split.h"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <functional>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

    using eeg_to_hypnogram::DatasetManifest;
    using eeg_to_hypnogram::DatasetManifestSplit;
    using eeg_to_hypnogram::DatasetSplitConfig;
    using eeg_to_hypnogram::SleepEdfFilePair;

    int gPassed = 0;

    void Require(bool condition, const std::string &message)
    {
        if (!condition)
        {
            throw std::runtime_error(message);
        }
    }

    template <typename ExceptionType, typename Function>
    void RequireThrows(Function &&function, const std::string &message)
    {
        bool threwExpected = false;
        try
        {
            function();
        }
        catch (const ExceptionType &)
        {
            threwExpected = true;
        }
        Require(threwExpected, message);
    }

    void Run(const std::string &name, const std::function<void()> &test)
    {
        test();
        ++gPassed;
        std::cout << "[PASS] " << name << '\n';
    }

    SleepEdfFilePair MakePair(
        const std::string &subject,
        const std::string &recording,
        const std::string &night = "1")
    {
        return {
            subject,
            recording,
            night,
            "/dataset/" + recording + "0-PSG.edf",
            "/dataset/" + recording + "C-Hypnogram.edf",
        };
    }

    DatasetManifest MakeManifest(
        std::vector<SleepEdfFilePair> pairs)
    {
        DatasetManifest manifest;
        manifest.pairs = std::move(pairs);
        return manifest;
    }

    std::set<std::string> SubjectSet(
        const std::vector<std::string> &subjects)
    {
        return std::set<std::string>(subjects.begin(), subjects.end());
    }

    std::set<std::string> SubjectsInPairs(
        const std::vector<SleepEdfFilePair> &pairs)
    {
        std::set<std::string> subjects;
        for (const auto &pair : pairs)
        {
            subjects.insert(pair.subjectId);
        }
        return subjects;
    }

    std::set<std::string> SplitSignature(
        const DatasetManifestSplit &split)
    {
        std::set<std::string> signature;
        for (const auto &subject : split.trainSubjectIds)
        {
            signature.insert("train:" + subject);
        }
        for (const auto &subject : split.testSubjectIds)
        {
            signature.insert("test:" + subject);
        }
        return signature;
    }

    std::vector<SleepEdfFilePair> FilterPairsBySubjects(
        const std::vector<SleepEdfFilePair> &pairs,
        const std::set<std::string> &subjects)
    {
        std::vector<SleepEdfFilePair> filtered;
        for (const auto &pair : pairs)
        {
            if (subjects.count(pair.subjectId) != 0)
            {
                filtered.push_back(pair);
            }
        }
        return filtered;
    }

    bool SamePair(
        const SleepEdfFilePair &left,
        const SleepEdfFilePair &right)
    {
        return left.subjectId == right.subjectId &&
               left.recordingId == right.recordingId &&
               left.nightId == right.nightId &&
               left.psgPath == right.psgPath &&
               left.hypnogramPath == right.hypnogramPath;
    }

    bool SamePairs(
        const std::vector<SleepEdfFilePair> &left,
        const std::vector<SleepEdfFilePair> &right)
    {
        if (left.size() != right.size())
        {
            return false;
        }
        for (std::size_t index = 0; index < left.size(); ++index)
        {
            if (!SamePair(left[index], right[index]))
            {
                return false;
            }
        }
        return true;
    }

    bool SameManifest(
        const DatasetManifest &left,
        const DatasetManifest &right)
    {
        return SamePairs(left.pairs, right.pairs) &&
               left.unmatchedPsgFiles == right.unmatchedPsgFiles &&
               left.unmatchedHypnogramFiles ==
                   right.unmatchedHypnogramFiles &&
               left.duplicatePsgKeys == right.duplicatePsgKeys &&
               left.duplicateHypnogramKeys == right.duplicateHypnogramKeys &&
               left.duplicateInputPaths == right.duplicateInputPaths &&
               left.ignoredFiles == right.ignoredFiles &&
               left.unrecognizedEdfFiles == right.unrecognizedEdfFiles;
    }

    bool SameSplit(
        const DatasetManifestSplit &left,
        const DatasetManifestSplit &right)
    {
        return SameManifest(left.train, right.train) &&
               SameManifest(left.test, right.test) &&
               left.trainSubjectIds == right.trainSubjectIds &&
               left.testSubjectIds == right.testSubjectIds;
    }

    std::vector<SleepEdfFilePair> MakeSubjectPairs(int subjectCount)
    {
        std::vector<SleepEdfFilePair> pairs;
        for (int index = 0; index < subjectCount; ++index)
        {
            const std::string subject =
                "SC" + std::to_string(400 + index);

            pairs.push_back(MakePair(subject, subject + "1E", "1"));
            if (index % 2 == 0)
            {
                pairs.push_back(MakePair(subject, subject + "2E", "2"));
            }
        }
        return pairs;
    }

    DatasetManifest MakeCleanManifestForSplitTests()
    {
        return MakeManifest(
            {
                MakePair("SC401", "SC4011E", "1"),
                MakePair("SC400", "SC4001E", "1"),
                MakePair("SC402", "SC4021E", "1"),
                MakePair("SC400", "SC4002E", "2"),
                MakePair("SC403", "SC4031E", "1"),
                MakePair("SC404", "SC4041E", "1"),
                MakePair("SC405", "SC4051E", "1"),
                MakePair("SC403", "SC4032E", "2"),
            });
    }

    void RequireSplitPreservesPairOrder(
        const DatasetManifest &source,
        const DatasetManifestSplit &split)
    {
        const std::set<std::string> trainSubjects =
            SubjectSet(split.trainSubjectIds);
        const std::set<std::string> testSubjects =
            SubjectSet(split.testSubjectIds);

        const auto expectedTrain =
            FilterPairsBySubjects(source.pairs, trainSubjects);
        const auto expectedTest =
            FilterPairsBySubjects(source.pairs, testSubjects);

        Require(
            SamePairs(expectedTrain, split.train.pairs),
            "Train pair order must follow the source manifest.");
        Require(
            SamePairs(expectedTest, split.test.pairs),
            "Test pair order must follow the source manifest.");
    }

    void TestDefaultConfiguration()
    {
        const DatasetSplitConfig config;
        Require(config.testFraction == 0.2, "Default testFraction changed.");
        Require(config.randomSeed == 42U, "Default randomSeed changed.");
    }

    void TestBasicSubjectSplit()
    {
        DatasetManifest manifest = MakeCleanManifestForSplitTests();
        manifest.ignoredFiles.push_back("/dataset/README.txt");

        const DatasetManifestSplit split =
            eeg_to_hypnogram::SplitDatasetManifestBySubject(manifest);

        eeg_to_hypnogram::ValidateManifestForDatasetAssembly(split.train);
        eeg_to_hypnogram::ValidateManifestForDatasetAssembly(split.test);

        Require(split.train.ignoredFiles.empty(), "Ignored files should not be copied to train.");
        Require(split.test.ignoredFiles.empty(), "Ignored files should not be copied to test.");

        const std::set<std::string> trainSubjects =
            SubjectSet(split.trainSubjectIds);
        const std::set<std::string> testSubjects =
            SubjectSet(split.testSubjectIds);

        Require(!split.train.pairs.empty(), "Train split must not be empty.");
        Require(!split.test.pairs.empty(), "Test split must not be empty.");
        Require(trainSubjects.size() == split.trainSubjectIds.size(), "Train subjects must be unique.");
        Require(testSubjects.size() == split.testSubjectIds.size(), "Test subjects must be unique.");
        Require(trainSubjects.size() == 5, "Default split must select 5 train subjects.");
        Require(testSubjects.size() == 1, "Default split must select 1 test subject.");
        Require(trainSubjects.size() + testSubjects.size() == 6, "All subjects must be allocated.");

        for (const auto &subject : trainSubjects)
        {
            Require(
                testSubjects.count(subject) == 0,
                "Train and test subjects must be disjoint.");
        }

        RequireSplitPreservesPairOrder(manifest, split);
        Require(
            split.train.pairs.size() + split.test.pairs.size() ==
                manifest.pairs.size(),
            "All pairs must be retained.");
    }

    void TestSameSubjectMultipleNightsStayTogether()
    {
        const DatasetManifest manifest = MakeManifest(
            {
                MakePair("SC400", "SC4001E", "1"),
                MakePair("SC401", "SC4011E", "1"),
                MakePair("SC400", "SC4002E", "2"),
                MakePair("SC402", "SC4021E", "1"),
                MakePair("SC403", "SC4031E", "1"),
            });

        const DatasetManifestSplit split =
            eeg_to_hypnogram::SplitDatasetManifestBySubject(manifest);

        const bool sc400InTrain =
            SubjectsInPairs(split.train.pairs).count("SC400") != 0;
        const bool sc400InTest =
            SubjectsInPairs(split.test.pairs).count("SC400") != 0;

        Require(sc400InTrain != sc400InTest, "A subject with two nights must stay in one split.");
    }

    void TestFixedSeedReproducibility()
    {
        const DatasetManifest manifest =
            MakeManifest(MakeSubjectPairs(10));

        DatasetSplitConfig config;
        config.randomSeed = 7;

        const DatasetManifestSplit first =
            eeg_to_hypnogram::SplitDatasetManifestBySubject(manifest, config);
        const DatasetManifestSplit second =
            eeg_to_hypnogram::SplitDatasetManifestBySubject(manifest, config);

        Require(SameSplit(first, second), "The same seed must produce identical splits.");
    }

    void TestDifferentSeeds()
    {
        const DatasetManifest manifest =
            MakeManifest(MakeSubjectPairs(10));

        DatasetSplitConfig baselineConfig;
        baselineConfig.randomSeed = 1;

        const DatasetManifestSplit baseline =
            eeg_to_hypnogram::SplitDatasetManifestBySubject(
                manifest,
                baselineConfig);
        const std::set<std::string> baselineSignature =
            SplitSignature(baseline);

        bool foundDifferentMembership = false;
        for (std::uint32_t seed = 2; seed <= 32; ++seed)
        {
            DatasetSplitConfig config;
            config.randomSeed = seed;
            const DatasetManifestSplit candidate =
                eeg_to_hypnogram::SplitDatasetManifestBySubject(
                    manifest,
                    config);
            if (SplitSignature(candidate) != baselineSignature)
            {
                foundDifferentMembership = true;
                break;
            }
        }

        Require(
            foundDifferentMembership,
            "At least one different seed should change subject membership.");
    }

    void TestInvalidConfiguration()
    {
        const DatasetManifest manifest =
            MakeManifest(MakeSubjectPairs(4));

        DatasetSplitConfig config;
        config.testFraction = 0.0;
        RequireThrows<std::invalid_argument>(
            [&]
            {
                (void)eeg_to_hypnogram::SplitDatasetManifestBySubject(
                    manifest,
                    config);
            },
            "testFraction=0 must be rejected.");

        config = {};
        config.testFraction = 1.0;
        RequireThrows<std::invalid_argument>(
            [&]
            {
                (void)eeg_to_hypnogram::SplitDatasetManifestBySubject(
                    manifest,
                    config);
            },
            "testFraction=1 must be rejected.");

        RequireThrows<std::invalid_argument>(
            []
            {
                (void)eeg_to_hypnogram::SplitDatasetManifestBySubject(
                    DatasetManifest{});
            },
            "An empty manifest must be rejected.");

        const DatasetManifest oneSubject =
            MakeManifest({MakePair("SC400", "SC4001E", "1")});
        RequireThrows<std::invalid_argument>(
            [&]
            {
                (void)eeg_to_hypnogram::SplitDatasetManifestBySubject(
                    oneSubject);
            },
            "A single subject cannot be split.");
    }

    void TestDirtyManifestRejected()
    {
        const std::vector<std::pair<std::string, std::function<void(DatasetManifest *)>>> dirtyCases =
            {
                {
                    "unmatched PSG",
                    [](DatasetManifest *manifest)
                    {
                        manifest->unmatchedPsgFiles.push_back(
                            "unmatched-PSG.edf");
                    },
                },
                {
                    "unmatched Hypnogram",
                    [](DatasetManifest *manifest)
                    {
                        manifest->unmatchedHypnogramFiles.push_back(
                            "unmatched-Hypnogram.edf");
                    },
                },
                {
                    "duplicate PSG key",
                    [](DatasetManifest *manifest)
                    {
                        manifest->duplicatePsgKeys.push_back("SC4001E");
                    },
                },
                {
                    "duplicate Hypnogram key",
                    [](DatasetManifest *manifest)
                    {
                        manifest->duplicateHypnogramKeys.push_back("SC4001E");
                    },
                },
                {
                    "duplicate input path",
                    [](DatasetManifest *manifest)
                    {
                        manifest->duplicateInputPaths.push_back(
                            "/dataset/SC4001E0-PSG.edf");
                    },
                },
                {
                    "unrecognized EDF",
                    [](DatasetManifest *manifest)
                    {
                        manifest->unrecognizedEdfFiles.push_back(
                            "/dataset/unknown.edf");
                    },
                },
            };

        for (const auto &dirtyCase : dirtyCases)
        {
            DatasetManifest manifest = MakeManifest(MakeSubjectPairs(4));
            manifest.ignoredFiles.push_back("/dataset/README.txt");
            dirtyCase.second(&manifest);

            RequireThrows<std::invalid_argument>(
                [&]
                {
                    (void)eeg_to_hypnogram::SplitDatasetManifestBySubject(
                        manifest);
                },
                "Dirty manifest must be rejected: " + dirtyCase.first);
        }
    }

    void RunRealDatasetCheckIfConfigured()
    {
        const char *datasetDirectory =
            std::getenv("EEG_SLEEP_EDF_DATASET_DIR");

        if (datasetDirectory == nullptr || *datasetDirectory == '\0')
        {
            std::cout
                << "[SKIP] optional real dataset split "
                << "(EEG_SLEEP_EDF_DATASET_DIR not set)\n";
            return;
        }

        eeg_to_hypnogram::DatasetManifestScanConfig scan;
        scan.recursive = true;

        const DatasetManifest manifest =
            eeg_to_hypnogram::BuildDatasetManifest(
                datasetDirectory,
                scan);

        Require(
            manifest.pairs.size() == 153,
            "Real Sleep-EDF manifest pair count must remain 153.");

        const DatasetManifestSplit split =
            eeg_to_hypnogram::SplitDatasetManifestBySubject(manifest);

        eeg_to_hypnogram::ValidateManifestForDatasetAssembly(split.train);
        eeg_to_hypnogram::ValidateManifestForDatasetAssembly(split.test);

        const std::set<std::string> trainSubjects =
            SubjectSet(split.trainSubjectIds);
        const std::set<std::string> testSubjects =
            SubjectSet(split.testSubjectIds);
        const std::set<std::string> allSubjects =
            SubjectsInPairs(manifest.pairs);

        Require(!split.train.pairs.empty(), "Real train split must not be empty.");
        Require(!split.test.pairs.empty(), "Real test split must not be empty.");
        Require(!trainSubjects.empty(), "Real train subjects must not be empty.");
        Require(!testSubjects.empty(), "Real test subjects must not be empty.");
        Require(
            trainSubjects.size() + testSubjects.size() ==
                allSubjects.size(),
            "Real subject counts must partition the manifest.");
        Require(
            allSubjects.size() == 78,
            "Real Sleep-EDF subject count must remain 78.");
        Require(
            trainSubjects.size() == 62,
            "Real Sleep-EDF train subject count must remain 62.");
        Require(
            testSubjects.size() == 16,
            "Real Sleep-EDF test subject count must remain 16.");
        for (const auto &subject : trainSubjects)
        {
            Require(
                testSubjects.count(subject) == 0,
                "Real train/test subjects must be disjoint.");
        }

        Require(
            split.train.pairs.size() + split.test.pairs.size() ==
                manifest.pairs.size(),
            "Real split must retain every pair.");
        Require(
            split.train.pairs.size() == 121,
            "Real Sleep-EDF train pair count must remain 121.");
        Require(
            split.test.pairs.size() == 32,
            "Real Sleep-EDF test pair count must remain 32.");

        std::cout << "real_subjects="
                  << (trainSubjects.size() + testSubjects.size()) << '\n';
        std::cout << "real_train_subjects=" << trainSubjects.size() << '\n';
        std::cout << "real_test_subjects=" << testSubjects.size() << '\n';
        std::cout << "real_train_pairs=" << split.train.pairs.size() << '\n';
        std::cout << "real_test_pairs=" << split.test.pairs.size() << '\n';
    }

} // namespace

int main()
{
    try
    {
        Run("default configuration", TestDefaultConfiguration);
        Run("basic subject split", TestBasicSubjectSplit);
        Run("multi-night subject stays together", TestSameSubjectMultipleNightsStayTogether);
        Run("fixed seed reproducibility", TestFixedSeedReproducibility);
        Run("different split seeds", TestDifferentSeeds);
        Run("invalid configuration", TestInvalidConfiguration);
        Run("dirty manifest rejection", TestDirtyManifestRejected);
        Run("optional real dataset split", RunRealDatasetCheckIfConfigured);

        std::cout << "Dataset Split tests passed\n";
        std::cout << "cases=" << gPassed << '\n';
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Dataset Split test failed: "
                  << error.what() << '\n';
        return 1;
    }
}
