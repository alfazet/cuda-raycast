cmake_minimum_required(VERSION 4.0)
project(its_raining_spheres LANGUAGES C CXX CUDA)
include(FetchContent)

set(CMAKE_CUDA_STANDARD 23)

include_directories(include)
file(GLOB_RECURSE SOURCES src/*.cu)
add_executable(its_raining_spheres main.cu ${SOURCES})

# link CUDA
find_package(CUDAToolkit REQUIRED)
target_link_libraries(its_raining_spheres CUDA::cudart)
target_include_directories(its_raining_spheres PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# link dependencies
FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 7b6aead9fb88b3623e3b3725ebb42670cbe4c579 # v3.4
)
FetchContent_MakeAvailable(glfw)
target_link_libraries(its_raining_spheres glfw)
target_include_directories(its_raining_spheres PRIVATE ${glfw_SOURCE_DIR}/include)

FetchContent_Declare(
        glad
        GIT_REPOSITORY https://github.com/Dav1dde/glad.git
        GIT_TAG 73db193f853e2ee079bf3ca8a64aa2eaf6459043 # v2.0.8
        SOURCE_SUBDIR cmake
)
FetchContent_MakeAvailable(glad)
glad_add_library(glad_gl_core_46 STATIC API gl:core=4.6)
target_link_libraries(its_raining_spheres glad_gl_core_46)

FetchContent_Declare(
        glm
        GIT_REPOSITORY https://github.com/g-truc/glm
        GIT_TAG a532f5b1cf27d6a3c099437e6959cf7e398a0a67 # v1.0.2
)
FetchContent_MakeAvailable(glm)
target_link_libraries(its_raining_spheres glm)

FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui
        GIT_TAG 9a5d5c45f54b1301ea471622eddede70384243af # v1.92.4
)
FetchContent_MakeAvailable(imgui)
file(GLOB imgui_SOURCES ${imgui_SOURCE_DIR}/*.cpp ${imgui_SOURCE_DIR}/*.h)
add_library(
        imgui
        ${imgui_SOURCES}
        ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.h
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.h
)
#target_link_libraries(imgui PUBLIC glad)
#target_link_libraries(imgui PUBLIC glfw)
target_include_directories(
        imgui
        PUBLIC
        ${imgui_SOURCE_DIR}
        ${imgui_SOURCE_DIR}/backends
)
target_include_directories(its_raining_spheres PRIVATE ${imgui_SOURCE_DIR})
target_link_libraries(its_raining_spheres imgui)

set_target_properties(its_raining_spheres PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(its_raining_spheres PRIVATE PROJECT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
endif ()