cmake_minimum_required(VERSION 3.20)

project(
    eeg_to_hypnogram
    VERSION 0.1.0
    LANGUAGES C CXX
)

# 生成 compile_commands.json，供 VS Code IntelliSense 和 clangd 使用。
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(
    EEG_TO_HYPNOGRAM_BUILD_CLI
    "Build the CLI application"
    OFF
)

# 创建 eeg_to_hypnogram::edflib target。
include(
    "${PROJECT_SOURCE_DIR}/cmake/edflib.cmake"
)

# 核心静态库。
add_library(
    eeg_to_hypnogram_core
    STATIC
        src/core/dataset_builder.cpp
        src/core/dataset_manifest.cpp
        src/core/dataset_split.cpp
        src/core/edf_reader.cpp
        src/core/epoch.cpp
        src/core/experiment_runner.cpp
        src/core/feature_extraction.cpp
        src/core/random_forest_baseline.cpp
        src/core/sleep_stage.cpp
        src/core/temporal_context.cpp
)

# 为核心库创建命名空间别名。
add_library(
    eeg_to_hypnogram::core
    ALIAS
    eeg_to_hypnogram_core
)

target_include_directories(
    eeg_to_hypnogram_core
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(
    eeg_to_hypnogram_core
    PUBLIC
        cxx_std_17
)

find_package(
    Threads
    REQUIRED
)

target_link_libraries(
    eeg_to_hypnogram_core
    PRIVATE
        Threads::Threads
)

# EDFlib 没有出现在公共头文件中，所以保持 PRIVATE。
target_link_libraries(
    eeg_to_hypnogram_core
    PRIVATE
        eeg_to_hypnogram::edflib
)

set_target_properties(
    eeg_to_hypnogram_core
    PROPERTIES
        CXX_EXTENSIONS OFF

        # 方便后续链接进 Flutter、Android 和 Linux 共享库。
        POSITION_INDEPENDENT_CODE ON
)

# 只给项目自己的代码启用严格警告。
# 不给第三方 EDFlib 强制开启这些警告。
if(MSVC)
    target_compile_options(
        eeg_to_hypnogram_core
        PRIVATE
            /W4
            /permissive-
    )
else()
    target_compile_options(
        eeg_to_hypnogram_core
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )
endif()

# CLI 暂时默认不构建。
if(EEG_TO_HYPNOGRAM_BUILD_CLI)
    add_executable(
        eeg_to_hypnogram_cli
        apps/cli/main.cpp
    )

    target_link_libraries(
        eeg_to_hypnogram_cli
        PRIVATE
            eeg_to_hypnogram::core
    )
endif()

# 启用 CTest，并创建 BUILD_TESTING 选项。
include(CTest)

if(BUILD_TESTING)
    # Dataset Builder 的内部测试辅助函数位于 dataset_builder.cpp 中。
    # 因此核心库本身也需要启用测试宏。
    target_compile_definitions(
        eeg_to_hypnogram_core
        PRIVATE
            EEG_TO_HYPNOGRAM_DATASET_BUILDER_TESTING=1
    )

    # Sleep Stage 测试。
    add_executable(
        sleep_stage_test
        tests/sleep_stage_test.cpp
    )

    target_link_libraries(
        sleep_stage_test
        PRIVATE
            eeg_to_hypnogram::core
    )

    add_test(
        NAME sleep_stage_test
        COMMAND sleep_stage_test
    )

    # EDF Reader 测试。
    add_executable(
        edf_reader_test
        tests/edf_reader_test.cpp
    )

    target_link_libraries(
        edf_reader_test
        PRIVATE
            eeg_to_hypnogram::core
    )

    add_test(
        NAME edf_reader_test
        COMMAND edf_reader_test
    )

    # Epoch 测试。
    add_executable(
        epoch_test
        tests/epoch_test.cpp
    )

    target_link_libraries(
        epoch_test
        PRIVATE
            eeg_to_hypnogram::core
    )

    add_test(
        NAME epoch_test
        COMMAND epoch_test
    )

    # Feature Extraction 测试。
    add_executable(
        feature_extraction_test
        tests/feature_extraction_test.cpp
    )

    target_link_libraries(
        feature_extraction_test
        PRIVATE
            eeg_to_hypnogram::core
    )

    add_test(
        NAME feature_extraction_test
        COMMAND feature_extraction_test
    )

    # Temporal Context 测试。
    add_executable(
        temporal_context_test
        tests/temporal_context_test.cpp
    )

    target_link_libraries(
        temporal_context_test
        PRIVATE
            eeg_to_hypnogram::core
    )

    add_test(
        NAME temporal_context_test
        COMMAND temporal_context_test
    )

    # Dataset Builder 测试。
    add_executable(
        dataset_builder_test
        tests/dataset_builder_test.cpp
    )

    # 让测试文件能够看到 dataset_builder_testing 命名空间中的声明。
    target_compile_definitions(
        dataset_builder_test
        PRIVATE
            EEG_TO_HYPNOGRAM_DATASET_BUILDER_TESTING=1
    )

    target_link_libraries(
        dataset_builder_test
        PRIVATE
            eeg_to_hypnogram::core

            # Dataset Builder 测试会直接调用 EDFlib 写入合成 EDF 文件。
            # 核心库对 EDFlib 是 PRIVATE 链接，因此这里必须再次显式链接。
            eeg_to_hypnogram::edflib
    )

    add_test(
        NAME dataset_builder_test
        COMMAND dataset_builder_test
    )

    # 随机森林算法测试
    add_executable(random_forest_baseline_test
        tests/random_forest_baseline_test.cpp
    )

    target_link_libraries(random_forest_baseline_test
        PRIVATE
            eeg_to_hypnogram_core
    )

    add_test(
        NAME random_forest_baseline_test
        COMMAND random_forest_baseline_test
    )

    # Experiment Runner 测试
    add_executable(experiment_runner_test
        tests/experiment_runner_test.cpp
    )

    target_link_libraries(experiment_runner_test
        PRIVATE
            eeg_to_hypnogram_core
    )

    add_test(
        NAME experiment_runner_test
        COMMAND experiment_runner_test
    )

    # Dataset Manifest 测试
    add_executable(dataset_manifest_test
        tests/dataset_manifest_test.cpp
    )

    target_link_libraries(dataset_manifest_test
        PRIVATE
            eeg_to_hypnogram_core
    )

    add_test(
        NAME dataset_manifest_test
        COMMAND dataset_manifest_test
    )

    # Dataset Split 测试
    add_executable(dataset_split_test
        tests/dataset_split_test.cpp
    )

    target_link_libraries(dataset_split_test
        PRIVATE
            eeg_to_hypnogram_core
    )

    add_test(
        NAME dataset_split_test
        COMMAND dataset_split_test
    )
endif()
