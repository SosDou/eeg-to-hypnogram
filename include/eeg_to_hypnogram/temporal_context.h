#pragma once

#include <vector>

namespace eeg_to_hypnogram
{

    // 时间上下文配置。
    // 对当前 epoch 的基础特征，分别拼接左侧和右侧相邻 epoch 的特征。
    struct TemporalContextConfig
    {
        // 当前 epoch 左侧拼接的上下文数量。
        int leftContext = 2;

        // 当前 epoch 右侧拼接的上下文数量。
        int rightContext = 2;
    };

    // 构建带时间上下文的特征矩阵。
    //
    // 输入：
    //   输入特征：基础特征矩阵，形状为
    //   [epoch 数量][基础特征维度]
    //
    // 配置：
    //   左右时间上下文配置。
    //
    // 返回值：
    //   上下文拼接后的特征矩阵。
    //
    // 输出行数与输入行数保持一致。
    // 输出维度为：
    //   基础特征维度 *
    //   (leftContext + 1 + rightContext)
    //
    // 到达记录边缘时，复制距离当前位置最近的合法 epoch。
    std::vector<std::vector<double>> BuildTemporalContextFeatures(
        const std::vector<std::vector<double>> &features,
        const TemporalContextConfig &config);

} // namespace eeg_to_hypnogram
