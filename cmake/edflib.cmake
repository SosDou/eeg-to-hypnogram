include_guard(GLOBAL)

# 允许后续通过 -DEEG_TO_HYPNOGRAM_EDFLIB_DIR=... 覆盖 EDFlib 位置。
set(
    EEG_TO_HYPNOGRAM_EDFLIB_DIR
    "${PROJECT_SOURCE_DIR}/third_party/edflib"
    CACHE PATH
    "Path to the vendored Teuniz EDFlib source directory"
)

set(
    EDFLIB_SOURCE_FILE
    "${EEG_TO_HYPNOGRAM_EDFLIB_DIR}/edflib.c"
)

set(
    EDFLIB_HEADER_FILE
    "${EEG_TO_HYPNOGRAM_EDFLIB_DIR}/edflib.h"
)

# 配置阶段立即检查依赖文件，避免到编译阶段才出现难以理解的错误。
if(
    NOT EXISTS "${EDFLIB_SOURCE_FILE}"
    OR NOT EXISTS "${EDFLIB_HEADER_FILE}"
)
    message(
        FATAL_ERROR
        "EDFlib source files were not found.\n"
        "Expected:\n"
        "  ${EDFLIB_SOURCE_FILE}\n"
        "  ${EDFLIB_HEADER_FILE}\n"
        "Download Teuniz EDFlib v1.27 into third_party/edflib before configuring."
    )
endif()

# EDFlib 是纯 C 库。
add_library(
    eeg_to_hypnogram_edflib
    STATIC
        "${EDFLIB_SOURCE_FILE}"
        "${EDFLIB_HEADER_FILE}"
)

# 提供统一的命名空间 target 名称。
add_library(
    eeg_to_hypnogram::edflib
    ALIAS
    eeg_to_hypnogram_edflib
)

# edf_reader.cpp 需要包含 <edflib.h>。
target_include_directories(
    eeg_to_hypnogram_edflib
    PUBLIC
        "${EEG_TO_HYPNOGRAM_EDFLIB_DIR}"
)

set_target_properties(
    eeg_to_hypnogram_edflib
    PROPERTIES
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        C_EXTENSIONS ON

        # 后续需要将静态核心库链接进 Flutter .so 等共享库。
        POSITION_INDEPENDENT_CODE ON
)

# EDFlib 官方针对 GNU/Clang 环境要求启用大文件支持。
# Emscripten 同样使用 Clang，因此也会经过这段配置。
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_definitions(
        eeg_to_hypnogram_edflib
        PRIVATE
            _LARGEFILE64_SOURCE
            _LARGEFILE_SOURCE
    )
endif()