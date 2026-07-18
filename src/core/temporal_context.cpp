#include "eeg_to_hypnogram/temporal_context.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace eeg_to_hypnogram
{

    std::vector<std::vector<double>> BuildTemporalContextFeatures(
        const std::vector<std::vector<double>> &features,
        const TemporalContextConfig &config)
    {
        // 空输入直接返回，即使上下文配置中包含负数也不抛异常。
        if (features.empty())
        {
            return {};
        }

        if (config.leftContext < 0 ||
            config.rightContext < 0)
        {
            throw std::invalid_argument(
                "Context size must be >= 0.");
        }

        const std::size_t numEpochs = features.size();
        const std::size_t baseDim = features.front().size();

        const std::size_t contextWidth =
            static_cast<std::size_t>(
                config.leftContext +
                config.rightContext +
                1);

        const std::size_t outputDim =
            baseDim * contextWidth;

        std::vector<std::vector<double>> output;

        output.assign(
            numEpochs,
            std::vector<double>(outputDim, 0.0));

        // 每个输出 epoch 按以下顺序拼接：
        //
        // [t-leftContext, ..., t-1, t, t+1, ..., t+rightContext]
        //
        // 到达记录左右边缘时，复制最近的合法 epoch。
        for (std::size_t epochIndex = 0;
             epochIndex < numEpochs;
             ++epochIndex)
        {
            std::size_t blockIndex = 0;

            for (int offset = -config.leftContext;
                 offset <= config.rightContext;
                 ++offset)
            {
                int sourceEpoch =
                    static_cast<int>(epochIndex) + offset;

                sourceEpoch = std::max(
                    0,
                    std::min(
                        sourceEpoch,
                        static_cast<int>(numEpochs) - 1));

                const auto &sourceFeatures =
                    features[static_cast<std::size_t>(
                        sourceEpoch)];

                if (sourceFeatures.size() != baseDim)
                {
                    throw std::runtime_error(
                        "Inconsistent feature dimension before temporal context.");
                }

                const std::size_t outputStart =
                    blockIndex * baseDim;

                std::copy(
                    sourceFeatures.begin(),
                    sourceFeatures.end(),
                    output[epochIndex].begin() +
                        static_cast<std::ptrdiff_t>(
                            outputStart));

                ++blockIndex;
            }
        }

        return output;
    }

} // namespace eeg_to_hypnogram