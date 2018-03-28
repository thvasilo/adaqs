# include head files
include_directories(include)
include_directories(${PROJECT_SOURCE_DIR}/externals)
include_directories(${PROJECT_SOURCE_DIR}/externals/spdlog/include)
# construct source files
file(GLOB src_files "src/*/*.cpp")
set(SOURCE_FILES ${src_files})
# add source files to this dynamic library
add_library(qs_core SHARED ${SOURCE_FILES})
target_link_libraries(qs_core
        boost_filesystem
        )
# install this target
install (TARGETS qs_core DESTINATION ${PROJECT_BINARY_DIR}/bin)
