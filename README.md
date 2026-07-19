# EEG to Hypnogram

一个基于 C++ 的跨平台 EEG 睡眠分期核心项目。

## 项目结构

```text
eeg-to-hypnogram/
├── .gitignore                               # Git 忽略规则
├── CMakeLists.txt                           # 项目根 CMake 构建配置
├── CMakePresets.json                        # CMake 配置、构建与测试预设
├── README.md                                # 项目说明、构建方法与使用文档
│
├── .vscode/
│   ├── launch.json                          # VS Code GDB 调试配置
│   └── settings.json                        # VS Code 项目级工作区配置
│
├── apps/
│   └── cli/
│       └── main.cpp                         # 命令行程序入口
│
├── cmake/
│   └── edflib.cmake                         # EDFlib 第三方库 CMake target 配置
│
├── datasets/                                # 本地数据集
│
├── include/
│   └── eeg_to_hypnogram/
│       ├── dataset_builder.h                # 训练与推理数据集构建接口
│       ├── dataset_manifest.h               # Sleep-EDF 文件识别、配对、数据清单与受试者级划分接口
│       ├── edf_reader.h                     # EDF 信号与 Hypnogram 注释读取接口
│       ├── epoch.h                          # 30 秒睡眠 epoch 构建接口
│       ├── experiment_runner.h              # 实验配置、训练、预测与评估流程接口
│       ├── feature_extraction.h             # EEG 特征提取接口
│       ├── random_forest_baseline.h         # 随机森林训练、推理与模型持久化接口
│       ├── sleep_stage.h                    # 定义睡眠阶段枚举、标注结构及字符串与分类标签转换接口
│       └── temporal_context.h               # 时序上下文配置与特征拼接接口
│
├── models/                                  # 本地训练模型与模型元数据
│
├── src/
│   └── core/
│       ├── dataset_builder.cpp              # 数据读取、重采样与样本构建实现
│       ├── dataset_manifest.cpp             # 实现数据集目录扫描、文件名解析、PSG/Hypnogram 配对与受试者级划分
│       ├── edf_reader.cpp                   # EDF 信号读取与 Hypnogram 注释解析实现
│       ├── epoch.cpp                        # 睡眠 epoch 切分与标签对齐实现
│       ├── experiment_runner.cpp            # 训练、预测、评估与实验流程实现
│       ├── feature_extraction.cpp           # 频带功率等 EEG 特征提取实现
│       ├── random_forest_baseline.cpp       # 随机森林训练、推理和模型保存加载实现
│       ├── sleep_stage.cpp                  # 实现 Sleep-EDF 阶段解析、规范化和五分类标签映射
│       └── temporal_context.cpp             # 相邻 epoch 特征的时序上下文拼接与边界处理
│
├── tests/                                   # 核心模块单元测试与集成测试
│   ├── dataset_builder_test.cpp             # 数据集构建模块测试
│   ├── dataset_manifest_test.cpp            # 测试文件识别、稳定配对、异常报告、受试者划分与真实数据扫描
│   ├── edf_reader_test.cpp                  # EDF 读取器单元测试与真实数据集成测试
│   ├── epoch_test.cpp                       # 睡眠 epoch 构建、标签对齐与真实数据集成测试
│   ├── experiment_runner_test.cpp           # 实验配置与评估流程测试
│   ├── feature_extraction_test.cpp          # EEG 特征提取结果测试与真实数据集成测试
│   ├── random_forest_baseline_test.cpp      # 随机森林训练、推理与模型持久化测试
│   ├── sleep_stage_test.cpp                 # 测试睡眠阶段解析、N4 合并、标签转换与非法输入处理
│   └── temporal_context_test.cpp            # 时序上下文的测试
│
└── third_party/                             # 项目内固定版本的第三方依赖
    └── edflib/                              # Teuniz EDFlib 源码与许可证
        ├── edflib.c                         # EDFlib C 语言实现
        ├── edflib.h                         # EDFlib 公共头文件
        └── LICENSE                          # EDFlib 开源许可证
```
