#pragma once

#include <string>
#include <vector>

namespace eeg_to_hypnogram
{

    enum class PsdMethod
    {
        Welch,
        Multitaper,
    };

    struct WelchConfig
    {
        int nFft = 256;
        int nOverlap = 128;
    };

    struct MultitaperConfig
    {
        int numTapers = 7;
    };

    struct PsdConfig
    {
        PsdMethod method = PsdMethod::Welch;
        WelchConfig welch{};
        MultitaperConfig multitaper{};
    };

    struct FrequencyBand
    {
        std::string name;
        double fmin = 0.0;
        double fmax = 0.0;
    };

    using EpochBatchData = std::vector<std::vector<std::vector<double>>>;

    std::vector<FrequencyBand> DefaultEegBands();

    std::vector<std::vector<double>> ComputeEegPowerBandFeatures(
        const EpochBatchData &epochs,
        const std::vector<double> &channelSampleRatesHz,
        const std::vector<FrequencyBand> &bands = DefaultEegBands(),
        double spectrumFmin = 0.5,
        double spectrumFmax = 30.0,
        const PsdConfig &psdConfig = PsdConfig());

} // namespace eeg_to_hypnogram
