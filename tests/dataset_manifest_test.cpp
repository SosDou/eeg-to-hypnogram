#include "eeg_to_hypnogram/dataset_manifest.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

namespace fs = std::filesystem;

using eeg_to_hypnogram::DatasetManifest;
using eeg_to_hypnogram::DatasetManifestScanConfig;
using eeg_to_hypnogram::DatasetSplit;
using eeg_to_hypnogram::SleepEdfFilePair;
using eeg_to_hypnogram::SubjectSplitConfig;

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

class TempDirectory final
{
public:
    explicit TempDirectory(const std::string &name)
    {
        static unsigned long long counter = 0;
        const auto timestamp =
            std::chrono::high_resolution_clock::now()
                .time_since_epoch()
                .count();
        path_ =
            fs::temp_directory_path() /
            ("eeg_to_hypnogram_manifest_" + name + "_" +
             std::to_string(timestamp) + "_" +
             std::to_string(++counter));
        fs::create_directories(path_);
    }

    ~TempDirectory()
    {
        std::error_code error;
        fs::remove_all(path_, error);
    }

    const fs::path &Path() const
    {
        return path_;
    }

private:
    fs::path path_;
};

fs::path Touch(const fs::path &directory, const std::string &filename)
{
    const fs::path path = directory / filename;
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    Require(static_cast<bool>(output), "Failed to create: " + path.string());
    return path;
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

std::set<std::string> Subjects(
    const std::vector<SleepEdfFilePair> &pairs)
{
    std::set<std::string> subjects;
    for (const auto &pair : pairs)
    {
        subjects.insert(pair.subjectId);
    }
    return subjects;
}

std::set<std::string> AllSubjects(const DatasetSplit &split)
{
    std::set<std::string> result = Subjects(split.train);
    const auto validation = Subjects(split.validation);
    const auto test = Subjects(split.test);
    result.insert(validation.begin(), validation.end());
    result.insert(test.begin(), test.end());
    return result;
}

std::set<std::string> SplitSignature(const DatasetSplit &split)
{
    std::set<std::string> signature;
    for (const auto &pair : split.train)
    {
        signature.insert("train:" + pair.subjectId);
    }
    for (const auto &pair : split.validation)
    {
        signature.insert("validation:" + pair.subjectId);
    }
    for (const auto &pair : split.test)
    {
        signature.insert("test:" + pair.subjectId);
    }
    return signature;
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

bool SameManifest(const DatasetManifest &left, const DatasetManifest &right)
{
    return SamePairs(left.pairs, right.pairs) &&
           left.unmatchedPsgFiles == right.unmatchedPsgFiles &&
           left.unmatchedHypnogramFiles == right.unmatchedHypnogramFiles &&
           left.duplicatePsgKeys == right.duplicatePsgKeys &&
           left.duplicateHypnogramKeys == right.duplicateHypnogramKeys &&
           left.duplicateInputPaths == right.duplicateInputPaths &&
           left.ignoredFiles == right.ignoredFiles &&
           left.unrecognizedEdfFiles == right.unrecognizedEdfFiles;
}

void TestDefaultConfiguration()
{
    const DatasetManifestScanConfig scan;
    Require(!scan.recursive, "Scanning must be non-recursive by default.");
    Require(!scan.followDirectorySymlinks, "Symlinks must not be followed by default.");

    const SubjectSplitConfig split;
    Require(split.trainRatio == 0.70, "Default train ratio changed.");
    Require(split.validationRatio == 0.15, "Default validation ratio changed.");
    Require(split.testRatio == 0.15, "Default test ratio changed.");
    Require(split.seed == 42U, "Default split seed changed.");
}

void TestInvalidDirectory()
{
    TempDirectory temp("invalid_directory");
    const fs::path missing = temp.Path() / "missing";
    RequireThrows<std::invalid_argument>(
        [&]
        {
            (void)eeg_to_hypnogram::BuildDatasetManifest(missing.string());
        },
        "A missing directory must throw invalid_argument.");

    const fs::path file = Touch(temp.Path(), "not-a-directory.txt");
    RequireThrows<std::invalid_argument>(
        [&]
        {
            (void)eeg_to_hypnogram::BuildDatasetManifest(file.string());
        },
        "A regular file passed as the dataset directory must throw.");
}

void TestEmptyDirectoryAndIgnoredFiles()
{
    TempDirectory temp("empty_and_ignored");
    DatasetManifest manifest =
        eeg_to_hypnogram::BuildDatasetManifest(temp.Path().string());
    Require(manifest.pairs.empty(), "An empty directory must have no pairs.");

    Touch(temp.Path(), "README.txt");
    Touch(temp.Path(), "SC4001E0-PSG.edf:Zone.Identifier");
    manifest = eeg_to_hypnogram::BuildDatasetManifest(temp.Path().string());
    Require(manifest.ignoredFiles.size() == 2, "Non-EDF files must be ignored and reported.");
    Require(manifest.unrecognizedEdfFiles.empty(), "Zone.Identifier must not be treated as EDF.");
}

void TestSleepEdfFilenameParsing()
{
    const auto cassettePsg =
        eeg_to_hypnogram::ParseSleepEdfFilename("SC4001E0-PSG.edf");
    Require(cassettePsg.has_value(), "Cassette PSG name must parse.");
    Require(cassettePsg->subjectId == "SC400", "Cassette subjectId changed.");
    Require(cassettePsg->recordingId == "SC4001E", "Cassette recordingId changed.");
    Require(cassettePsg->nightId == "1", "Cassette nightId changed.");
    Require(cassettePsg->isPsg && !cassettePsg->isHypnogram, "PSG kind flags are wrong.");

    const auto cassetteHyp =
        eeg_to_hypnogram::ParseSleepEdfFilename("sc4001ec-hypnogram.EDF");
    Require(cassetteHyp.has_value(), "Case-insensitive Hypnogram name must parse.");
    Require(cassetteHyp->recordingId == "SC4001E", "PSG/Hypnogram keys must match.");

    const auto telemetryHyp =
        eeg_to_hypnogram::ParseSleepEdfFilename(
            R"(C:\sleep-edf\ST7011JP-Hypnogram.edf)");
    Require(telemetryHyp.has_value(), "Windows-style telemetry path must parse.");
    Require(telemetryHyp->subjectId == "ST701", "Telemetry subjectId changed.");
    Require(telemetryHyp->recordingId == "ST7011J", "Telemetry recordingId changed.");
    Require(telemetryHyp->isHypnogram, "Telemetry Hypnogram kind changed.");

    Require(
        !eeg_to_hypnogram::ParseSleepEdfFilename("SC4001EA-PSG.edf").has_value(),
        "A PSG whose eighth recording character is not 0 must be rejected.");
    Require(
        !eeg_to_hypnogram::ParseSleepEdfFilename("unknown.edf").has_value(),
        "An unrelated EDF filename must not parse.");
}

void TestPairingAndStableOrdering()
{
    const std::vector<std::string> unordered = {
        "/data/ST7012JE-Hypnogram.edf",
        "/data/SC4011E0-PSG.edf",
        "/data/SC4002EC-Hypnogram.edf",
        "/data/ST7012J0-PSG.edf",
        "/data/SC4002E0-PSG.edf",
        "/data/SC4011EC-Hypnogram.edf",
    };

    const DatasetManifest first =
        eeg_to_hypnogram::BuildDatasetManifestFromFiles(unordered);

    auto reversed = unordered;
    std::reverse(reversed.begin(), reversed.end());
    const DatasetManifest second =
        eeg_to_hypnogram::BuildDatasetManifestFromFiles(reversed);

    Require(SameManifest(first, second), "Input order must not affect the manifest.");
    Require(first.pairs.size() == 3, "Expected three complete pairs.");
    Require(first.pairs[0].recordingId == "SC4002E", "Pairs must be sorted by subject and recording.");
    Require(first.pairs[1].recordingId == "SC4011E", "Cassette ordering changed.");
    Require(first.pairs[2].recordingId == "ST7012J", "Telemetry ordering changed.");
}

void TestUnmatchedAndUnrecognizedFiles()
{
    const DatasetManifest manifest =
        eeg_to_hypnogram::BuildDatasetManifestFromFiles(
            {
                "/data/SC4001E0-PSG.edf",
                "/data/SC4011EC-Hypnogram.edf",
                "/data/not-sleep-edf.edf",
                "/data/notes.csv",
            });

    Require(manifest.pairs.empty(), "Unmatched files must not create pairs.");
    Require(manifest.unmatchedPsgFiles.size() == 1, "Unmatched PSG must be reported.");
    Require(manifest.unmatchedHypnogramFiles.size() == 1, "Unmatched Hypnogram must be reported.");
    Require(manifest.unrecognizedEdfFiles.size() == 1, "Unparseable EDF must be reported separately.");
    Require(manifest.ignoredFiles.size() == 1, "Non-EDF file must be ignored.");
}

void TestDuplicateDetection()
{
    const DatasetManifest duplicatePsg =
        eeg_to_hypnogram::BuildDatasetManifestFromFiles(
            {
                "/a/SC4001E0-PSG.edf",
                "/b/SC4001E0-PSG.edf",
                "/a/SC4001EC-Hypnogram.edf",
            });
    Require(duplicatePsg.pairs.empty(), "Ambiguous PSG key must be excluded from pairs.");
    Require(duplicatePsg.duplicatePsgKeys == std::vector<std::string>({"SC4001E"}), "Duplicate PSG key changed.");

    const DatasetManifest duplicateHyp =
        eeg_to_hypnogram::BuildDatasetManifestFromFiles(
            {
                "/a/ST7011J0-PSG.edf",
                "/a/ST7011JP-Hypnogram.edf",
                "/b/ST7011JR-Hypnogram.edf",
            });
    Require(duplicateHyp.pairs.empty(), "Ambiguous Hypnogram key must be excluded from pairs.");
    Require(duplicateHyp.duplicateHypnogramKeys == std::vector<std::string>({"ST7011J"}), "Duplicate Hypnogram key changed.");
}

void TestDuplicateInputPath()
{
    const DatasetManifest manifest =
        eeg_to_hypnogram::BuildDatasetManifestFromFiles(
            {
                "/data/SC4001E0-PSG.edf",
                "/data/SC4001E0-PSG.edf",
                "/data/SC4001EC-Hypnogram.edf",
            });

    Require(manifest.duplicateInputPaths.size() == 1, "Exact duplicate input path must be reported.");
    Require(manifest.duplicatePsgKeys.empty(), "Exact duplicate path must not masquerade as a key collision.");
    Require(manifest.pairs.size() == 1, "A deduplicated exact path may still form one valid pair.");
}

void TestRecursiveScanning()
{
    TempDirectory temp("recursive");
    const fs::path nested = temp.Path() / "sleep-cassette";
    Touch(nested, "SC4001E0-PSG.edf");
    Touch(nested, "SC4001EC-Hypnogram.edf");

    const DatasetManifest shallow =
        eeg_to_hypnogram::BuildDatasetManifest(temp.Path().string());
    Require(shallow.pairs.empty(), "Default scanning must not recurse.");

    DatasetManifestScanConfig config;
    config.recursive = true;
    const DatasetManifest recursive =
        eeg_to_hypnogram::BuildDatasetManifest(temp.Path().string(), config);
    Require(recursive.pairs.size() == 1, "Explicit recursive scanning must find nested files.");
}

std::vector<SleepEdfFilePair> MakeSubjectDataset(int subjectCount)
{
    std::vector<SleepEdfFilePair> pairs;
    for (int index = 0; index < subjectCount; ++index)
    {
        const std::string number =
            index < 10 ? "40" + std::to_string(index)
                       : "4" + std::to_string(index);
        const std::string subject = "SC" + number;
        pairs.push_back(MakePair(subject, subject + "1E", "1"));
        pairs.push_back(MakePair(subject, subject + "2E", "2"));
    }
    return pairs;
}

void TestSubjectLevelSplit()
{
    const auto pairs = MakeSubjectDataset(10);
    const DatasetSplit split =
        eeg_to_hypnogram::SplitDatasetBySubject(pairs);

    eeg_to_hypnogram::ValidateSubjectDisjointSplit(split);

    const auto trainSubjects = Subjects(split.train);
    const auto validationSubjects = Subjects(split.validation);
    const auto testSubjects = Subjects(split.test);

    Require(trainSubjects.size() == 7, "Default 10-subject train count must be 7.");
    Require(validationSubjects.size() == 2, "Largest remainder must assign 2 validation subjects.");
    Require(testSubjects.size() == 1, "Largest remainder must assign 1 test subject.");
    Require(AllSubjects(split).size() == 10, "Every subject must be allocated exactly once.");

    for (const auto &subject : trainSubjects)
    {
        Require(validationSubjects.count(subject) == 0 && testSubjects.count(subject) == 0,
                "One subject must not cross split boundaries.");
    }
}

void TestFixedSeedReproducibilityAndInputImmutability()
{
    auto pairs = MakeSubjectDataset(12);
    std::reverse(pairs.begin(), pairs.end());
    const auto original = pairs;

    const DatasetSplit first =
        eeg_to_hypnogram::SplitDatasetBySubject(pairs);
    const DatasetSplit second =
        eeg_to_hypnogram::SplitDatasetBySubject(pairs);

    Require(SamePairs(pairs, original), "Subject splitting must not modify input pairs.");
    Require(SplitSignature(first) == SplitSignature(second), "Fixed seed split must be reproducible.");
    Require(SamePairs(first.train, second.train), "Repeated calls must not append old train results.");
    Require(SamePairs(first.validation, second.validation), "Repeated calls must not append old validation results.");
    Require(SamePairs(first.test, second.test), "Repeated calls must not append old test results.");
}

void TestDifferentSeeds()
{
    const auto pairs = MakeSubjectDataset(20);

    SubjectSplitConfig firstConfig;
    firstConfig.seed = 1;
    SubjectSplitConfig secondConfig;
    secondConfig.seed = 2;

    const DatasetSplit first =
        eeg_to_hypnogram::SplitDatasetBySubject(pairs, firstConfig);
    const DatasetSplit second =
        eeg_to_hypnogram::SplitDatasetBySubject(pairs, secondConfig);

    Require(SplitSignature(first) != SplitSignature(second), "Chosen different seeds should produce different memberships.");
}

void TestSplitRatioValidation()
{
    const auto pairs = MakeSubjectDataset(4);

    SubjectSplitConfig config;
    config.trainRatio = 0.0;
    RequireThrows<std::invalid_argument>(
        [&]
        {
            (void)eeg_to_hypnogram::SplitDatasetBySubject(pairs, config);
        },
        "trainRatio=0 must be rejected.");

    config = {};
    config.testRatio = 0.0;
    RequireThrows<std::invalid_argument>(
        [&]
        {
            (void)eeg_to_hypnogram::SplitDatasetBySubject(pairs, config);
        },
        "testRatio=0 must be rejected.");

    config = {};
    config.trainRatio = 0.8;
    config.validationRatio = 0.2;
    config.testRatio = 0.2;
    RequireThrows<std::invalid_argument>(
        [&]
        {
            (void)eeg_to_hypnogram::SplitDatasetBySubject(pairs, config);
        },
        "Ratios whose sum is not one must be rejected.");

    config = {};
    config.trainRatio = 0.75;
    config.validationRatio = 0.0;
    config.testRatio = 0.25;
    const DatasetSplit noValidation =
        eeg_to_hypnogram::SplitDatasetBySubject(pairs, config);
    Require(noValidation.validation.empty(), "validationRatio=0 must produce an empty validation split.");
    Require(!noValidation.train.empty() && !noValidation.test.empty(), "Train and test must remain non-empty.");
}

void TestSmallSubjectBoundaries()
{
    RequireThrows<std::invalid_argument>(
        []
        {
            (void)eeg_to_hypnogram::SplitDatasetBySubject({});
        },
        "An empty pair list must be rejected.");

    const auto oneSubject = MakeSubjectDataset(1);
    RequireThrows<std::invalid_argument>(
        [&]
        {
            SubjectSplitConfig config;
            config.trainRatio = 0.8;
            config.validationRatio = 0.0;
            config.testRatio = 0.2;
            (void)eeg_to_hypnogram::SplitDatasetBySubject(oneSubject, config);
        },
        "One subject cannot populate train and test.");

    const auto twoSubjects = MakeSubjectDataset(2);
    RequireThrows<std::invalid_argument>(
        [&]
        {
            (void)eeg_to_hypnogram::SplitDatasetBySubject(twoSubjects);
        },
        "Two subjects cannot populate three positive-ratio splits.");

    SubjectSplitConfig noValidation;
    noValidation.trainRatio = 0.5;
    noValidation.validationRatio = 0.0;
    noValidation.testRatio = 0.5;
    const DatasetSplit validTwo =
        eeg_to_hypnogram::SplitDatasetBySubject(twoSubjects, noValidation);
    Require(Subjects(validTwo.train).size() == 1, "Two-subject train count must be one.");
    Require(Subjects(validTwo.test).size() == 1, "Two-subject test count must be one.");
}

void TestLeakageDetection()
{
    DatasetSplit subjectLeak;
    subjectLeak.train.push_back(MakePair("SC400", "SC4001E"));
    subjectLeak.test.push_back(MakePair("SC400", "SC4002E", "2"));
    RequireThrows<std::invalid_argument>(
        [&]
        {
            eeg_to_hypnogram::ValidateSubjectDisjointSplit(subjectLeak);
        },
        "A subject crossing train and test must be rejected.");

    DatasetSplit pathLeak;
    auto trainPair = MakePair("SC400", "SC4001E");
    auto testPair = MakePair("SC401", "SC4011E");
    testPair.psgPath = trainPair.psgPath;
    pathLeak.train.push_back(trainPair);
    pathLeak.test.push_back(testPair);
    RequireThrows<std::invalid_argument>(
        [&]
        {
            eeg_to_hypnogram::ValidateSubjectDisjointSplit(pathLeak);
        },
        "A repeated PSG path must be rejected.");

    DatasetSplit recordingLeak;
    const auto duplicate = MakePair("SC400", "SC4001E");
    recordingLeak.train.push_back(duplicate);
    recordingLeak.train.push_back(duplicate);
    RequireThrows<std::invalid_argument>(
        [&]
        {
            eeg_to_hypnogram::ValidateSubjectDisjointSplit(recordingLeak);
        },
        "A duplicate recording within one split must be rejected.");
}

void RunRealDatasetCheckIfConfigured()
{
    const char *directory = std::getenv("EEG_SLEEP_EDF_DATASET_DIR");
    if (directory == nullptr || std::string(directory).empty())
    {
        std::cout << "[SKIP] optional real dataset scan (EEG_SLEEP_EDF_DATASET_DIR not set)\n";
        return;
    }

    DatasetManifestScanConfig scan;
    scan.recursive = true;
    const DatasetManifest manifest =
        eeg_to_hypnogram::BuildDatasetManifest(directory, scan);

    Require(!manifest.pairs.empty(), "Configured real dataset must contain at least one complete pair.");
    Require(manifest.duplicatePsgKeys.empty(), "Real dataset must not contain duplicate PSG keys.");
    Require(manifest.duplicateHypnogramKeys.empty(), "Real dataset must not contain duplicate Hypnogram keys.");

    std::cout << "real_pairs=" << manifest.pairs.size() << '\n';
    std::cout << "real_unmatched_psg=" << manifest.unmatchedPsgFiles.size() << '\n';
    std::cout << "real_unmatched_hypnogram=" << manifest.unmatchedHypnogramFiles.size() << '\n';

    const std::size_t preview = std::min<std::size_t>(3, manifest.pairs.size());
    for (std::size_t index = 0; index < preview; ++index)
    {
        std::cout << "pair[" << index << "]="
                  << manifest.pairs[index].recordingId << '\n';
    }

    const std::size_t subjectCount = [&]
    {
        std::set<std::string> values;
        for (const auto &pair : manifest.pairs)
        {
            values.insert(pair.subjectId);
        }
        return values.size();
    }();

    if (subjectCount >= 3)
    {
        const DatasetSplit split =
            eeg_to_hypnogram::SplitDatasetBySubject(manifest.pairs);
        eeg_to_hypnogram::ValidateSubjectDisjointSplit(split);
    }
    else if (subjectCount >= 2)
    {
        SubjectSplitConfig config;
        config.trainRatio = 0.5;
        config.validationRatio = 0.0;
        config.testRatio = 0.5;
        const DatasetSplit split =
            eeg_to_hypnogram::SplitDatasetBySubject(manifest.pairs, config);
        eeg_to_hypnogram::ValidateSubjectDisjointSplit(split);
    }
}

} // namespace

int main()
{
    try
    {
        Run("default configuration", TestDefaultConfiguration);
        Run("invalid directory", TestInvalidDirectory);
        Run("empty directory and ignored files", TestEmptyDirectoryAndIgnoredFiles);
        Run("Sleep-EDF filename parsing", TestSleepEdfFilenameParsing);
        Run("PSG and Hypnogram pairing", TestPairingAndStableOrdering);
        Run("unmatched and unrecognized files", TestUnmatchedAndUnrecognizedFiles);
        Run("duplicate detection", TestDuplicateDetection);
        Run("duplicate input paths", TestDuplicateInputPath);
        Run("recursive scanning", TestRecursiveScanning);
        Run("subject-level split", TestSubjectLevelSplit);
        Run("fixed seed reproducibility", TestFixedSeedReproducibilityAndInputImmutability);
        Run("different split seeds", TestDifferentSeeds);
        Run("split ratio validation", TestSplitRatioValidation);
        Run("small subject boundaries", TestSmallSubjectBoundaries);
        Run("subject leakage detection", TestLeakageDetection);
        Run("optional real dataset scan", RunRealDatasetCheckIfConfigured);

        std::cout << "Dataset Manifest tests passed\n";
        std::cout << "cases=" << gPassed << '\n';
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Dataset Manifest test failed: "
                  << error.what() << '\n';
        return 1;
    }
}
