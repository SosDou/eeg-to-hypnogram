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
├── docs/                                    # 各核心模块的技术实现文档
│   ├── dataset_builder_doc.md               # 数据集构建模块技术文档
│   ├── edf_reader_doc.md                    # EDF 与 Hypnogram 读取模块技术文档
│   ├── epoch_doc.md                         # 睡眠 epoch 构建模块技术文档
│   ├── experiment_runner_doc.md             # 实验、训练、预测与评估流程技术文档
│   ├── feature_extraction_doc.md            # EEG 特征提取模块技术文档
│   ├── random_forest_baseline_doc.md        # 随机森林模型模块技术文档
│   └── sleep_stage_doc.md                   # 睡眠阶段公共类型技术文档
│
├── include/
│   └── eeg_to_hypnogram/
│       ├── dataset_builder.h                # 训练与推理数据集构建接口
│       ├── edf_reader.h                     # EDF 信号与 Hypnogram 注释读取接口
│       ├── epoch.h                          # 30 秒睡眠 epoch 构建接口
│       ├── experiment_runner.h              # 实验配置、训练、预测与评估流程接口
│       ├── feature_extraction.h             # EEG 特征提取接口
│       ├── random_forest_baseline.h         # 随机森林训练、推理与模型持久化接口
│       └── sleep_stage.h                    # 睡眠阶段枚举、转换与公共类型
│
├── models/                                  # 本地训练模型与模型元数据
│
├── src/
│   └── core/
│       ├── dataset_builder.cpp              # 数据读取、重采样与样本构建实现
│       ├── edf_reader.cpp                   # EDF 信号读取与 Hypnogram 注释解析实现
│       ├── epoch.cpp                        # 睡眠 epoch 切分与标签对齐实现
│       ├── experiment_runner.cpp            # 训练、预测、评估与实验流程实现
│       ├── feature_extraction.cpp           # 频带功率等 EEG 特征提取实现
│       ├── random_forest_baseline.cpp       # 随机森林训练、推理和模型保存加载实现
│       └── sleep_stage.cpp                  # 睡眠阶段解析与字符串转换实现
│
├── tests/                                   # 核心模块单元测试与集成测试
│   ├── dataset_builder_test.cpp             # 数据集构建模块测试
│   ├── edf_reader_test.cpp                  # EDF 读取器单元测试与真实数据集成测试
│   ├── epoch_test.cpp                       # 睡眠 epoch 构建与标签对齐测试
│   ├── experiment_runner_test.cpp           # 实验配置与评估流程测试
│   ├── feature_extraction_test.cpp          # EEG 特征提取结果测试
│   ├── random_forest_baseline_test.cpp      # 随机森林训练、推理与模型持久化测试
│   └── sleep_stage_test.cpp                 # 睡眠阶段枚举与转换测试
│
└── third_party/                             # 项目内固定版本的第三方依赖
    └── edflib/                              # Teuniz EDFlib 源码与许可证
        ├── edflib.c                         # EDFlib C 语言实现
        ├── edflib.h                         # EDFlib 公共头文件
        └── LICENSE                          # EDFlib 开源许可证
```