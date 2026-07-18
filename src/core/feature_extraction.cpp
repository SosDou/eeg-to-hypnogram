// 文件说明：
// - 本文件实现 EEG 频域特征提取的核心流程。
// - 主要职责是把按 epoch 切分后的多通道时域信号，转换为频段功率特征向量。
// - 具体包含：PSD 估计（Welch / Multitaper）、频谱归一化、按频段聚合并拼接。
// - 输出结果通常作为后续时序上下文拼接与分类模型训练的输入特征。
#include "eeg_to_hypnogram/feature_extraction.h"
#include <algorithm>
#include <cstddef>
#include <cmath>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{

    constexpr double kPi = 3.14159265358979323846;

    std::vector<double> ComputeWelchOneSidedPsd(const std::vector<double> &samples,
                                                double sampleRateHz,
                                                const eeg_to_hypnogram::WelchConfig &welch,
                                                double maxFrequencyHz,
                                                std::vector<double> *freqsOut)
    {
        // Welch PSD 的输入保护：至少需要 2 个采样点，且采样率/FFT 参数合法。
        if (samples.size() < 2)
        {
            throw std::invalid_argument("Each channel epoch must contain at least 2 samples.");
        }
        if (sampleRateHz <= 0.0)
        {
            throw std::invalid_argument("sampleRateHz must be > 0.");
        }
        if (welch.nFft < 2)
        {
            throw std::invalid_argument("welch.nFft must be >= 2.");
        }

        const std::size_t nSamples = samples.size();
        // 每段长度不超过数据长度；当 epoch 很短时自动缩短段长。
        const std::size_t nPerSeg = std::min<std::size_t>(static_cast<std::size_t>(welch.nFft), nSamples);
        std::size_t nOverlap = static_cast<std::size_t>(std::max(0, welch.nOverlap));
        if (nOverlap >= nPerSeg)
        {
            nOverlap = nPerSeg - 1;
        }
        // 滑窗步长 = 段长 - 重叠。
        const std::size_t step = nPerSeg - nOverlap;
        const std::size_t nFft = std::max<std::size_t>(static_cast<std::size_t>(welch.nFft), nPerSeg);
        const std::size_t nFreqBinsFull = (nFft / 2) + 1;

        std::size_t nFreqBins = nFreqBinsFull;
        if (maxFrequencyHz > 0.0)
        {
            // 可选频率上限：只保留 [0, maxFrequencyHz] 对应的频点，减少后续计算量。
            const std::size_t maxBin = static_cast<std::size_t>(std::floor((maxFrequencyHz * static_cast<double>(nFft)) / sampleRateHz));
            nFreqBins = std::min(nFreqBinsFull, maxBin + 1);
        }

        // Hann 窗：降低频谱泄漏。
        std::vector<double> window(nPerSeg, 0.0);
        for (std::size_t i = 0; i < nPerSeg; ++i)
        {
            window[i] = 0.5 * (1.0 - std::cos((2.0 * kPi * static_cast<double>(i)) / static_cast<double>(nPerSeg - 1)));
        }
        // 用窗函数能量做归一化，保证不同窗长下尺度一致。
        const double windowPower = std::inner_product(window.begin(), window.end(), window.begin(), 0.0);

        std::vector<double> psd(nFreqBins, 0.0);
        int segmentCount = 0;

        static std::mutex trigCacheMutex;
        static std::size_t cacheNFft = 0;
        static std::size_t cacheNPerSeg = 0;
        static std::size_t cacheNFreqBins = 0;
        static std::vector<std::vector<double>> cosTable;
        static std::vector<std::vector<double>> sinTable;

        {
            // 三角函数表缓存：相同 nFft/nPerSeg 配置复用，避免每个 epoch 重复计算 sin/cos。
            const std::lock_guard<std::mutex> lock(trigCacheMutex);
            if (cacheNFft != nFft || cacheNPerSeg != nPerSeg || cacheNFreqBins != nFreqBins)
            {
                cosTable.assign(nFreqBins, std::vector<double>(nPerSeg, 0.0));
                sinTable.assign(nFreqBins, std::vector<double>(nPerSeg, 0.0));
                for (std::size_t k = 0; k < nFreqBins; ++k)
                {
                    for (std::size_t t = 0; t < nPerSeg; ++t)
                    {
                        const double angle = (2.0 * kPi * static_cast<double>(k) * static_cast<double>(t)) / static_cast<double>(nFft);
                        cosTable[k][t] = std::cos(angle);
                        sinTable[k][t] = std::sin(angle);
                    }
                }

                cacheNFft = nFft;
                cacheNPerSeg = nPerSeg;
                cacheNFreqBins = nFreqBins;
            }
        }

        for (std::size_t start = 0; start + nPerSeg <= nSamples; start += step)
        {
            ++segmentCount;

            // 每段先减均值，去除直流偏置。
            const double segmentMean = std::accumulate(samples.begin() + static_cast<std::ptrdiff_t>(start),
                                                       samples.begin() + static_cast<std::ptrdiff_t>(start + nPerSeg),
                                                       0.0) /
                                       static_cast<double>(nPerSeg);

            for (std::size_t k = 0; k < nFreqBins; ++k)
            {
                double realPart = 0.0;
                double imagPart = 0.0;

                for (std::size_t t = 0; t < nPerSeg; ++t)
                {
                    const double x = (samples[start + t] - segmentMean) * window[t];
                    realPart += x * cosTable[k][t];
                    imagPart -= x * sinTable[k][t];
                }

                double power = (realPart * realPart + imagPart * imagPart) / (sampleRateHz * windowPower);
                if (k != 0 && !(nFft % 2 == 0 && k == (nFft / 2)))
                {
                    // 单边谱能量修正：除 DC 和 Nyquist 外，其它频点乘 2。
                    power *= 2.0;
                }

                psd[k] += power;
            }
        }

        if (segmentCount <= 0)
        {
            throw std::runtime_error("No Welch segment generated.");
        }

        for (double &value : psd)
        {
            // Welch 的核心：对所有分段功率谱取平均，降低方差。
            value /= static_cast<double>(segmentCount);
        }

        std::vector<double> freqs(nFreqBins, 0.0);
        for (std::size_t k = 0; k < nFreqBins; ++k)
        {
            freqs[k] = (static_cast<double>(k) * sampleRateHz) / static_cast<double>(nFft);
        }

        *freqsOut = std::move(freqs);
        return psd;
    }

    std::vector<double> ComputeMultitaperOneSidedPsd(const std::vector<double> &samples,
                                                     double sampleRateHz,
                                                     const eeg_to_hypnogram::MultitaperConfig &cfg,
                                                     std::vector<double> *freqsOut)
    {
        // Multitaper 版本的基本参数检查。
        if (samples.size() < 2)
        {
            throw std::invalid_argument("Each channel epoch must contain at least 2 samples.");
        }
        if (sampleRateHz <= 0.0)
        {
            throw std::invalid_argument("sampleRateHz must be > 0.");
        }
        if (cfg.numTapers < 1)
        {
            throw std::invalid_argument("multitaper.numTapers must be >= 1.");
        }

        const std::size_t n = samples.size();
        const std::size_t nFft = n;
        const std::size_t nFreqBins = (nFft / 2) + 1;

        const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(n);

        std::vector<double> psd(nFreqBins, 0.0);

        // 对每个 taper 分别估计谱，再取平均。
        for (int taperIdx = 1; taperIdx <= cfg.numTapers; ++taperIdx)
        {
            // Orthonormal sine tapers are used here as a lightweight approximation to DPSS tapers.
            std::vector<double> taper(n, 0.0);
            for (std::size_t t = 0; t < n; ++t)
            {
                taper[t] = std::sqrt(2.0 / static_cast<double>(n + 1)) *
                           std::sin((kPi * static_cast<double>(taperIdx) * static_cast<double>(t + 1)) / static_cast<double>(n + 1));
            }
            const double taperPower = std::inner_product(taper.begin(), taper.end(), taper.begin(), 0.0);

            for (std::size_t k = 0; k < nFreqBins; ++k)
            {
                double realPart = 0.0;
                double imagPart = 0.0;

                for (std::size_t t = 0; t < n; ++t)
                {
                    const double x = (samples[t] - mean) * taper[t];
                    const double angle = (2.0 * kPi * static_cast<double>(k) * static_cast<double>(t)) / static_cast<double>(nFft);
                    realPart += x * std::cos(angle);
                    imagPart -= x * std::sin(angle);
                }

                double power = (realPart * realPart + imagPart * imagPart) / (sampleRateHz * taperPower);
                if (k != 0 && !(nFft % 2 == 0 && k == (nFft / 2)))
                {
                    // 单边谱能量修正，同 Welch。
                    power *= 2.0;
                }
                psd[k] += power;
            }
        }

        for (double &value : psd)
        {
            // 多 taper 平均，减少单一窗导致的谱估计波动。
            value /= static_cast<double>(cfg.numTapers);
        }

        std::vector<double> freqs(nFreqBins, 0.0);
        for (std::size_t k = 0; k < nFreqBins; ++k)
        {
            freqs[k] = (static_cast<double>(k) * sampleRateHz) / static_cast<double>(nFft);
        }

        *freqsOut = std::move(freqs);
        return psd;
    }

    bool InRange(double value, double low, double high)
    {
        // 左闭右开区间，避免边界频点在相邻频段被重复计数。
        return value >= low && value < high;
    }

} // namespace

namespace eeg_to_hypnogram
{

    std::vector<FrequencyBand> DefaultEegBands()
    {
        return {
            {"delta", 0.5, 4.5},
            {"theta", 4.5, 8.5},
            {"alpha", 8.5, 11.5},
            {"sigma", 11.5, 15.5},
            {"beta", 15.5, 30.0},
        };
    }

    std::vector<std::vector<double>> ComputeEegPowerBandFeatures(
        const EpochBatchData &epochs,
        const std::vector<double> &channelSampleRatesHz,
        const std::vector<FrequencyBand> &bands,
        double spectrumFmin,
        double spectrumFmax,
        const PsdConfig &psdConfig)
    {
        if (spectrumFmin < 0.0 || spectrumFmax <= spectrumFmin)
        {
            throw std::invalid_argument("Invalid spectrum frequency range.");
        }
        if (bands.empty())
        {
            throw std::invalid_argument("bands must not be empty.");
        }

        std::vector<std::vector<double>> features;
        features.reserve(epochs.size());

        // 外层循环：逐个 epoch 计算特征。
        for (const auto &epochChannels : epochs)
        {
            if (epochChannels.empty())
            {
                throw std::invalid_argument("Each epoch must contain at least one channel.");
            }
            if (epochChannels.size() != channelSampleRatesHz.size())
            {
                throw std::invalid_argument("channelSampleRatesHz size must match epoch channel count.");
            }

            std::vector<std::vector<double>> normalizedPsdByChannel;
            std::vector<std::vector<double>> freqsByChannel;
            normalizedPsdByChannel.reserve(epochChannels.size());
            freqsByChannel.reserve(epochChannels.size());

            // 先按通道计算并归一化 PSD。
            for (std::size_t ch = 0; ch < epochChannels.size(); ++ch)
            {
                std::vector<double> freqs;
                std::vector<double> psd;
                if (psdConfig.method == PsdMethod::Multitaper)
                {
                    psd = ComputeMultitaperOneSidedPsd(epochChannels[ch], channelSampleRatesHz[ch], psdConfig.multitaper, &freqs);
                }
                else
                {
                    psd = ComputeWelchOneSidedPsd(epochChannels[ch], channelSampleRatesHz[ch], psdConfig.welch, spectrumFmax, &freqs);
                }

                std::vector<double> normalizedPsd = psd;

                double totalPower = 0.0;
                for (std::size_t i = 0; i < normalizedPsd.size(); ++i)
                {
                    // 只用目标总频带 [spectrumFmin, spectrumFmax) 的能量做归一化。
                    if (InRange(freqs[i], spectrumFmin, spectrumFmax))
                    {
                        totalPower += normalizedPsd[i];
                    }
                }
                if (totalPower <= 0.0)
                {
                    // 防止除零：若总能量异常为 0，则保持原值尺度。
                    totalPower = 1.0;
                }

                for (double &value : normalizedPsd)
                {
                    value /= totalPower;
                }

                normalizedPsdByChannel.push_back(std::move(normalizedPsd));
                freqsByChannel.push_back(std::move(freqs));
            }

            std::vector<double> epochFeature;
            epochFeature.reserve(epochChannels.size() * bands.size());

            // 频段优先拼接，和 Python np.concatenate(bands...) 对齐。
            for (const auto &band : bands)
            {
                for (std::size_t ch = 0; ch < epochChannels.size(); ++ch)
                {
                    const auto &normalizedPsd = normalizedPsdByChannel[ch];
                    const auto &freqs = freqsByChannel[ch];

                    double bandPowerSum = 0.0;
                    int bandBinCount = 0;

                    for (std::size_t i = 0; i < normalizedPsd.size(); ++i)
                    {
                        if (InRange(freqs[i], band.fmin, band.fmax))
                        {
                            bandPowerSum += normalizedPsd[i];
                            ++bandBinCount;
                        }
                    }

                    double bandMeanNormalized = 0.0;
                    if (bandBinCount > 0)
                    {
                        // 每个频段特征定义为该段内“归一化功率均值”。
                        bandMeanNormalized = bandPowerSum / static_cast<double>(bandBinCount);
                    }

                    epochFeature.push_back(bandMeanNormalized);
                }
            }

            features.push_back(std::move(epochFeature));
        }

        return features;
    }

} // namespace eeg_to_hypnogram
