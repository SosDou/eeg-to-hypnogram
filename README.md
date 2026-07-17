# EEG to Hypnogram

一个基于 C++ 的跨平台 EEG 睡眠分期核心项目。

## 项目结构

```text

eeg-to-hypnogram/
├── CMakeLists.txt
├── CMakePresets.json
├── README.md
├── .gitignore
│
├── apps/
│   └── cli/
│       └── main.cpp                         # 命令行入口
│
├── include/
│   └── eeg_to_hypnogram/
│       ├── dataset_builder.h                # 训练与推理数据构建接口
│       ├── edf_reader.h                     # EDF 与 Hypnogram 读取接口
│       ├── epoch.h                          # 30 秒睡眠 epoch 构建接口
│       ├── experiment_runner.h              # 实验与评估流程接口
│       ├── feature_extraction.h             # EEG 特征提取接口
│       ├── random_forest_baseline.h         # 随机森林模型接口
│       └── sleep_stage.h                    # 睡眠阶段公共类型
│
├── src/
│   └── core/
│       ├── dataset_builder.cpp              # 数据构建流程实现
│       ├── edf_reader.cpp                   # EDF 与注释解析实现
│       ├── epoch.cpp                        # 睡眠 epoch 构建实现
│       ├── experiment_runner.cpp            # 实验与评估流程实现
│       ├── feature_extraction.cpp           # EEG 特征提取实现
│       ├── random_forest_baseline.cpp       # 随机森林实现
│       └── sleep_stage.cpp                  # 睡眠阶段公共类型实现
│
├── models/
│   └── .gitkeep
│
└── datasets/
    ├── .gitkeep
    └── physionet_sleep_data/                # 本地 PhysioNet 数据

```