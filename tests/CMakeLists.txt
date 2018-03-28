# include head files and link libraries
include_directories(${PROJECT_SOURCE_DIR}/core/include)

include_directories(${PROJECT_SOURCE_DIR}/externals/spdlog/include)
include_directories(${PROJECT_SOURCE_DIR}/externals/googletest/googletest/include)
include_directories(${PROJECT_SOURCE_DIR}/externals/dmlc/include)
include_directories(${PROJECT_SOURCE_DIR}/externals/xgboost-predictor/include)
include_directories(${PROJECT_SOURCE_DIR}/externals)
# add test executable
file(GLOB src_files "src/*/*.cpp")
set(SOURCE_FILES ${src_files} src/main.cpp)
add_executable(qs_test ${SOURCE_FILES})
target_link_libraries(qs_test
        qs_core
        boost_system
        gtest
        gtest_main
        )
