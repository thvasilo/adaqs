# include head files and link libraries
include_directories(include)
include_directories(${PROJECT_SOURCE_DIR}/core/include)
include_directories(${PROJECT_SOURCE_DIR}/externals)
include_directories(${PROJECT_SOURCE_DIR}/externals/spdlog/include)
include_directories(${PROJECT_SOURCE_DIR}/externals/dmlc/include)
include_directories(${PROJECT_SOURCE_DIR}/externals/xgboost-predictor/include)
# add profile executable
add_executable(qs_profile
        src/profile/qs_profile_main.cpp
        src/profile/qs_profile.cpp)
target_link_libraries(qs_profile
        qs_core
        boost_system
        )

IF(CMAKE_BUILD_TYPE MATCHES Profile)
    target_link_libraries(qs_profile profiler)
ENDIF(CMAKE_BUILD_TYPE MATCHES Profile)

# add eval executables
add_executable(qs_eval_time
        src/evaluation/eval_time.cpp
        src/profile/qs_profile.cpp)
target_link_libraries(qs_eval_time
        qs_core
        boost_system
        pthread
        )

add_executable(qs_eval_parameters
        src/evaluation/eval_parameters.cpp
        src/profile/qs_profile.cpp)
target_link_libraries(qs_eval_parameters
        qs_core
        boost_system
        )

# create log directory for spdlog
add_custom_command(TARGET qs_profile PRE_BUILD
        COMMAND rm -rf ${PROJECT_SOURCE_DIR}/qs_logs/app
        COMMAND mkdir -p ${PROJECT_SOURCE_DIR}/qs_logs/app)

# install targets
install (TARGETS qs_profile DESTINATION ${PROJECT_BINARY_DIR}/bin)
install (TARGETS qs_eval_time DESTINATION ${PROJECT_BINARY_DIR}/bin)
install (TARGETS qs_eval_parameters DESTINATION ${PROJECT_BINARY_DIR}/bin)