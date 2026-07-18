#include "eeg_to_hypnogram/edf_reader.h"
#include "eeg_to_hypnogram/feature_extraction.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

    constexpr double kPi = 3.14159265358979323846;

    void Require(bool condition, const std::string &message)
    {
        if (!condition)
        {
            throw std::runtime_error(message);
        }
    }

    bool NearlyEqual(double lhs, double rhs, double absoluteTolerance = 1e-12, double relativeTolerance = 1e-9)
    {
        const double difference = std::abs(lhs - rhs);
        if (difference <= absoluteTolerance)
        {
            return true;
        }

        return difference <= relativeTolerance * std::max(std::abs(lhs), std::abs(rhs));
    }

    std::vector<double> MakeSineWave(double frequencyHz,
                                     double sampleRateHz,
                                     double durationSeconds,
                                     double amplitude = 1.0,
                                     double dcOffset = 0.0)
    {
        const std::size_t sampleCount = static_cast<std::size_t>(
            std::llround(sampleRateHz * durationSeconds));

        std::vector<double> samples(sampleCount, 0.0);
        for (std::size_t i = 0; i < sampleCount; ++i)
        {
            const double timeSeconds = static_cast<double>(i) / sampleRateHz;
            samples[i] = dcOffset + amplitude * std::sin(2.0 * kPi * frequencyHz * timeSeconds);
        }

        return samples;
    }

    std::vector<double> ComputeSingleChannelFeatures(const std::vector<double> &samples,
                                                     double sampleRateHz)
    {
        const eeg_to_hypnogram::EpochBatchData epochs = {
            {samples},
        };

        const auto features = eeg_to_hypnogram::ComputeEegPowerBandFeatures(
            epochs,
            {sampleRateHz});

        Require(features.size() == 1, "Expected exactly one output epoch.");
        Require(features.front().size() == 5, "Expected five default band features.");
        return features.front();
    }

    std::size_t MaxIndex(const std::vector<double> &values)
    {
        return static_cast<std::size_t>(
            std::distance(values.begin(), std::max_element(values.begin(), values.end())));
    }

    void PrintFeatures(const std::string &label, const std::vector<double> &features)
    {
        std::cout << label << ':';
        for (double value : features)
        {
            std::cout << ' ' << value;
        }
        std::cout << '\n';
    }

    void TestDefaultBands()
    {
        const auto bands = eeg_to_hypnogram::DefaultEegBands();
        Require(bands.size() == 5, "DefaultEegBands must return five bands.");

        const std::vector<std::string> expectedNames = {
            "delta", "theta", "alpha", "sigma", "beta"};
        const std::vector<double> expectedMin = {
            0.5, 4.5, 8.5, 11.5, 15.5};
        const std::vector<double> expectedMax = {
            4.5, 8.5, 11.5, 15.5, 30.0};

        for (std::size_t i = 0; i < bands.size(); ++i)
        {
            Require(bands[i].name == expectedNames[i], "Unexpected default band name.");
            Require(NearlyEqual(bands[i].fmin, expectedMin[i]), "Unexpected default band minimum.");
            Require(NearlyEqual(bands[i].fmax, expectedMax[i]), "Unexpected default band maximum.");
        }
    }

    void TestEmptyBatch()
    {
        const eeg_to_hypnogram::EpochBatchData epochs;
        const auto features = eeg_to_hypnogram::ComputeEegPowerBandFeatures(epochs, {100.0});
        Require(features.empty(), "Empty batch must return an empty feature matrix.");
    }

    void TestZeroSignal()
    {
        const std::vector<double> samples(3000, 0.0);
        const auto features = ComputeSingleChannelFeatures(samples, 100.0);

        for (double value : features)
        {
            Require(NearlyEqual(value, 0.0), "Zero signal must produce zero features.");
        }
    }

    void TestSineBandDominance()
    {
        struct Case
        {
            double frequencyHz;
            std::size_t expectedBandIndex;
            const char *label;
        };

        const std::vector<Case> cases = {
            {2.0, 0, "2Hz-delta"},
            {6.0, 1, "6Hz-theta"},
            {10.0, 2, "10Hz-alpha"},
            {13.0, 3, "13Hz-sigma"},
            {20.0, 4, "20Hz-beta"},
        };

        for (const auto &testCase : cases)
        {
            const auto samples = MakeSineWave(testCase.frequencyHz, 100.0, 30.0);
            const auto features = ComputeSingleChannelFeatures(samples, 100.0);
            PrintFeatures(testCase.label, features);

            Require(MaxIndex(features) == testCase.expectedBandIndex,
                    std::string("Unexpected dominant band for ") + testCase.label + '.');
        }
    }

    void TestAmplitudeAndDcOffsetInvariance()
    {
        const auto baseline = ComputeSingleChannelFeatures(
            MakeSineWave(10.0, 100.0, 30.0, 1.0, 0.0),
            100.0);
        const auto scaledAndOffset = ComputeSingleChannelFeatures(
            MakeSineWave(10.0, 100.0, 30.0, 7.0, 123.0),
            100.0);

        for (std::size_t i = 0; i < baseline.size(); ++i)
        {
            Require(NearlyEqual(baseline[i], scaledAndOffset[i], 1e-12, 1e-8),
                    "Normalized PSD features should be invariant to amplitude scaling and DC offset.");
        }
    }

    void TestFrequencyBandFirstOrdering()
    {
        const eeg_to_hypnogram::EpochBatchData epochs = {
            {
                MakeSineWave(2.0, 100.0, 30.0),
                MakeSineWave(10.0, 100.0, 30.0),
            },
        };

        const auto features = eeg_to_hypnogram::ComputeEegPowerBandFeatures(
            epochs,
            {100.0, 100.0});

        Require(features.size() == 1, "Expected one output epoch.");
        Require(features.front().size() == 10, "Expected ten features for two channels and five bands.");

        const auto &row = features.front();

        std::vector<double> channel0 = {row[0], row[2], row[4], row[6], row[8]};
        std::vector<double> channel1 = {row[1], row[3], row[5], row[7], row[9]};

        Require(MaxIndex(channel0) == 0, "Channel 0 should be delta-dominant.");
        Require(MaxIndex(channel1) == 2, "Channel 1 should be alpha-dominant.");
    }

    void TestInvalidInput()
    {
        bool threw = false;
        try
        {
            const eeg_to_hypnogram::EpochBatchData epochs = {{{0.0, 1.0}}};
            (void)eeg_to_hypnogram::ComputeEegPowerBandFeatures(epochs, {});
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        Require(threw, "Mismatched channel sample-rate count must throw.");

        threw = false;
        try
        {
            const eeg_to_hypnogram::EpochBatchData epochs = {{{0.0}}};
            (void)eeg_to_hypnogram::ComputeEegPowerBandFeatures(epochs, {100.0});
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        Require(threw, "A channel with fewer than two samples must throw.");
    }

    void RunEdfIntegrationTestIfEnabled()
    {
        const char *edfFilePath =
            std::getenv("EEG_TEST_EDF_FILE");

        if (edfFilePath == nullptr ||
            std::string(edfFilePath).empty())
        {
            std::cout
                << "Feature extraction EDF integration test skipped: "
                << "EEG_TEST_EDF_FILE is not set.\n";
            return;
        }

        constexpr const char *kSignalLabel =
            "EEG Fpz-Cz";

        constexpr double kEpochDurationSeconds =
            30.0;

        double epochStartSeconds =
            28830.0;

        const char *epochStartText =
            std::getenv(
                "EEG_TEST_EPOCH_START_SECONDS");

        if (epochStartText != nullptr &&
            !std::string(epochStartText).empty())
        {
            epochStartSeconds =
                std::stod(epochStartText);
        }

        Require(
            std::isfinite(epochStartSeconds) &&
                epochStartSeconds >= 0.0,
            "EDF integration epoch start must be finite and non-negative.");

        eeg_to_hypnogram::EdfReader reader;

        // PSG 文件不需要读取 annotation。
        reader.Open(edfFilePath, false);

        const auto signalIndex =
            reader.FindSignalIndexByLabel(
                kSignalLabel);

        Require(
            signalIndex.has_value(),
            "EEG Fpz-Cz signal was not found in the EDF file.");

        const auto &header =
            reader.Header();

        const auto &signalInfo =
            header.signals.at(
                static_cast<std::size_t>(
                    *signalIndex));

        const double sampleRateHz =
            signalInfo.sampleRateHz;

        Require(
            std::isfinite(sampleRateHz) &&
                sampleRateHz > 0.0,
            "EDF signal sample rate must be finite and positive.");

        const long long startSampleValue =
            std::llround(
                epochStartSeconds *
                sampleRateHz);

        const long long sampleCountValue =
            std::llround(
                kEpochDurationSeconds *
                sampleRateHz);

        Require(
            startSampleValue >= 0,
            "Calculated EDF start sample must be non-negative.");

        Require(
            sampleCountValue > 0,
            "Calculated EDF sample count must be positive.");

        Require(
            sampleCountValue <=
                static_cast<long long>(
                    std::numeric_limits<int>::max()),
            "Calculated EDF sample count exceeds the reader API limit.");

        const auto samples =
            reader.ReadPhysicalSamples(
                *signalIndex,
                static_cast<std::int64_t>(
                    startSampleValue),
                static_cast<int>(
                    sampleCountValue));

        Require(
            samples.size() ==
                static_cast<std::size_t>(
                    sampleCountValue),
            "EDF integration test did not read a complete 30-second epoch.");

        const eeg_to_hypnogram::EpochBatchData epochs = {
            {
                samples,
            },
        };

        const auto featureMatrix =
            eeg_to_hypnogram::
                ComputeEegPowerBandFeatures(
                    epochs,
                    {
                        sampleRateHz,
                    });

        Require(
            featureMatrix.size() == 1,
            "EDF integration test must produce exactly one feature row.");

        Require(
            featureMatrix.front().size() == 5,
            "Single-channel EDF epoch must produce five default band features.");

        const auto &features =
            featureMatrix.front();

        const std::vector<double> expectedFeatures = {
            0.090454796231647761,
            0.0059060047420707957,
            0.00096478103048554535,
            0.0008746020204499624,
            0.00053858707606568108,
        };

        Require(
            features.size() == expectedFeatures.size(),
            "EDF feature count does not match the golden result.");

        for (std::size_t index = 0;
             index < expectedFeatures.size();
             ++index)
        {
            Require(
                NearlyEqual(
                    features[index],
                    expectedFeatures[index],
                    1e-12,
                    1e-8),
                "EDF feature does not match the golden result at index " +
                    std::to_string(index) +
                    ". Expected " +
                    std::to_string(expectedFeatures[index]) +
                    ", got " +
                    std::to_string(features[index]) +
                    ".");
        }

        for (double value : features)
        {
            Require(
                std::isfinite(value),
                "EDF feature must be finite.");

            Require(
                value >= 0.0,
                "EDF feature must be non-negative.");
        }

        // 同一输入再次计算，用于确认结果稳定。
        const auto repeatedFeatureMatrix =
            eeg_to_hypnogram::
                ComputeEegPowerBandFeatures(
                    epochs,
                    {
                        sampleRateHz,
                    });

        Require(
            repeatedFeatureMatrix.size() == 1 &&
                repeatedFeatureMatrix.front().size() ==
                    features.size(),
            "Repeated EDF feature extraction returned an invalid shape.");

        for (std::size_t index = 0;
             index < features.size();
             ++index)
        {
            Require(
                NearlyEqual(
                    features[index],
                    repeatedFeatureMatrix.front()[index],
                    1e-12,
                    1e-10),
                "Repeated EDF feature extraction produced different results.");
        }

        const auto bands =
            eeg_to_hypnogram::DefaultEegBands();

        std::cout
            << std::setprecision(17);

        std::cout
            << "Feature extraction EDF integration test passed\n"
            << "signal=" << signalInfo.label << '\n'
            << "sample_rate_hz=" << sampleRateHz << '\n'
            << "epoch_start_seconds=" << epochStartSeconds << '\n'
            << "samples_per_epoch=" << samples.size() << '\n';

        for (std::size_t index = 0;
             index < features.size();
             ++index)
        {
            std::cout
                << bands[index].name
                << '='
                << features[index]
                << '\n';
        }
    }

} // namespace

int main()
{
    try
    {
        TestDefaultBands();
        TestEmptyBatch();
        TestZeroSignal();
        TestSineBandDominance();
        TestAmplitudeAndDcOffsetInvariance();
        TestFrequencyBandFirstOrdering();
        TestInvalidInput();

        RunEdfIntegrationTestIfEnabled();

        std::cout << "All feature extraction tests passed.\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Feature extraction test failed: " << error.what() << '\n';
        return 1;
    }
}
