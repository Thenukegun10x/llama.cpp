cmake_minimum_required(VERSION 3.31.6)

project("ai-chat" VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}"   CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "" FORCE)

# --------------------------------------------------------------------------
# AI Chat library
# --------------------------------------------------------------------------

if(DEFINED ANDROID_ABI)
    message(STATUS "Detected Android ABI: ${ANDROID_ABI}")
    if(ANDROID_ABI STREQUAL "arm64-v8a")
        set(GGML_SYSTEM_ARCH "ARM")
        set(GGML_CPU_KLEIDIAI ON)
        set(GGML_OPENMP ON)
    elseif(ANDROID_ABI STREQUAL "x86_64")
        set(GGML_SYSTEM_ARCH "x86")
        set(GGML_CPU_KLEIDIAI OFF)
        set(GGML_OPENMP OFF)
    else()
        message(FATAL_ERROR "Unsupported ABI: ${ANDROID_ABI}")
    endif()
endif()

set(LLAMA_SRC ${CMAKE_CURRENT_LIST_DIR}/../../../../../../)
add_subdirectory(${LLAMA_SRC} build-llama)

add_library(${CMAKE_PROJECT_NAME} SHARED
        ai_chat.cpp)

target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
        GGML_SYSTEM_ARCH=${GGML_SYSTEM_ARCH}
        GGML_CPU_KLEIDIAI=$<BOOL:${GGML_CPU_KLEIDIAI}>
        GGML_OPENMP=$<BOOL:${GGML_OPENMP}>
)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
        ${LLAMA_SRC}
        ${LLAMA_SRC}/common
        ${LLAMA_SRC}/include
        ${LLAMA_SRC}/ggml/include
        ${LLAMA_SRC}/ggml/src)

target_link_libraries(${CMAKE_PROJECT_NAME}
        llama
        llama-common
        android
        log)

# The NDK sysroot provides vulkan/vulkan.h but not the C++ vulkan/vulkan.hpp
# header; pull that from the Vulkan SDK when building the Vulkan backend.
if(DEFINED ENV{VULKAN_SDK})
    target_include_directories(ggml-vulkan PRIVATE "$ENV{VULKAN_SDK}/Include")
endif()

# --------------------------------------------------------------------------
# Stable Diffusion library (disabled to keep build footprint small)
# To enable: set SD_ENABLED=ON via -DSD_ENABLED=ON cmake arg
# --------------------------------------------------------------------------
option(SD_ENABLED "Build stable-diffusion.cpp image generation backend" ON)
set(SD_SRC ${CMAKE_CURRENT_LIST_DIR}/../../../../../../../stable-diffusion.cpp)
if(SD_ENABLED AND EXISTS ${SD_SRC}/CMakeLists.txt)
    message(STATUS "Stable Diffusion source found at ${SD_SRC}")

    set(SD_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(SD_BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
    set(SD_WEBP OFF CACHE BOOL "" FORCE)
    set(SD_WEBM OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

    if(DEFINED ENV{VULKAN_SDK})
        set(SD_VULKAN ON CACHE BOOL "" FORCE)
    endif()

    if(DEFINED ANDROID_ABI AND ANDROID_ABI STREQUAL "arm64-v8a")
        set(GGML_SYSTEM_ARCH "ARM" CACHE STRING "" FORCE)
        set(GGML_CPU_KLEIDIAI ON CACHE BOOL "" FORCE)
        set(GGML_OPENMP ON CACHE BOOL "" FORCE)
    endif()

    # stable-diffusion.cpp reuses the ggml target already built by llama.cpp above.
    # If the ggml versions diverge, initialize stable-diffusion.cpp's ggml submodule
    # and set SD_USE_SYSTEM_GGML=OFF to build a separate ggml.
    add_subdirectory(${SD_SRC} build-sd EXCLUDE_FROM_ALL)

    add_library(sd-chat SHARED sd_chat.cpp)

    target_include_directories(sd-chat PRIVATE
        ${SD_SRC}
        ${SD_SRC}/include
        ${SD_SRC}/ggml/include
        ${SD_SRC}/ggml/src)

    target_link_libraries(sd-chat
        stable-diffusion
        android
        log)
else()
    message(STATUS "Stable Diffusion source not found at ${SD_SRC} -- skipping image generation support")
endif()
