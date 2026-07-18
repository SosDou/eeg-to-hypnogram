cmake_minimum_required(VERSION 3.20)

project(
    eeg_to_hypnogram
    VERSION 0.1.0
    LANGUAGES C CXX
)

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

add_library(
    eeg_to_hypnogram_core
    STATIC
        src/core/edf_reader.cpp
        src/core/epoch.cpp
        src/core/feature_extraction.cpp
        src/core/sleep_stage.cpp
)

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

find_package(Threads REQUIRED)

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

        # 方便后续链接进 Flutter/Android/Linux 共享库。
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

include(CTest)

if(BUILD_TESTING)
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

    # Epoch 测试
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

    # Feature Extraction 测试
    add_executable(feature_extraction_test
        tests/feature_extraction_test.cpp
    )

    target_link_libraries(feature_extraction_test
        PRIVATE
            eeg_to_hypnogram::core
    )

    add_test(
        NAME feature_extraction_test
        COMMAND feature_extraction_test
    )
endif()