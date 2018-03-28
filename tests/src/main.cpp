#include <iostream>
#include <gtest/gtest.h>
#include <log/qs_log.h>
#include <boost/filesystem.hpp>

using namespace std;
using namespace quickscorer;
namespace fs=boost::filesystem;

int main(int argc, char **argv) {
    string logger_name = "qs_unit_test_logger";
    string logger_dir_name = "qs_logs/unit_test";
    string logger_file_name = logger_dir_name + "/unit_test_log";
    auto logger_dir = fs::path(logger_dir_name);
    if (fs::exists(logger_dir)) fs::remove_all(logger_dir);
    fs::create_directories(logger_dir);

    QSLog::create_loggers(logger_name, logger_file_name);
    ::testing::InitGoogleTest(&argc, argv);
    auto result = RUN_ALL_TESTS();
    QSLog::release_loggers();
    return result;
}
