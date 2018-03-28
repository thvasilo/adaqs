# initialize project
cmake_minimum_required(VERSION 3.5)
project(quickscorer LANGUAGES VERSION 1.0.0 C CXX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -march=native -pipe")
# add flags
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -march=native -pipe")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG")
set(CMAKE_CXX_FLAGS_PROFILE "-g -O2 -march=native -pipe -D_PROFILE")
set(CMAKE_EXE_LINKER_FLAGS_PROFILE "-lprofiler")
mark_as_advanced(
        CMAKE_CXX_FLAGS_PROFILE
        CMAKE_EXE_LINKER_FLAGS_PROFILE)
# update the documentation string of CMAKE_BUILD_TYPE for GUIs
set(CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel Profile."
        FORCE)
# set configuration macros
add_definitions(-DFOUR_BYTES_BANDWITH) # see core/include/conf/config.h for usage
# add global includes and libraries (e.g., boost and eigen)
include_directories(/usr/local/include)
link_directories(/usr/local/lib)
set(CMAKE_INSTALL_RPATH ".")
# add sub projects
add_subdirectory(externals)
add_subdirectory(core)
add_subdirectory(tests)
add_subdirectory(app)