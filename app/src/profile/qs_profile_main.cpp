//
// Created by qfeng on 17-10-12.
//

#include "log/qs_log.h"
#include <boost/filesystem.hpp>
#include <gbtree_model/gbtree_model.h>
#include <qs_model/qs_model.h>
#include "profile/qs_profile.h"

#ifdef _PROFILE
#include <gperftools/profiler.h>
#endif

using namespace std;
using namespace quickscorer;
using namespace quickscorer_profile;
namespace fs=boost::filesystem;


int main(int argc, char **argv) {
    // initialize logger
    string logger_name = "qs_app_profile_logger";
    string logger_dir_name = "qs_logs/app";
    string logger_file_name = logger_dir_name + "/app_profile_log";
    auto logger_dir = fs::path(logger_dir_name);
    QSLog::create_loggers(logger_name, logger_file_name);
    // initialize models
    auto qs_model = QSModel::from_xgb_txt("tests/data/avazu-app-1000/model.dump", 1000);
    // load samples
    auto samples = load_samples("tests/data/avazu-app-1000/data.test", qs_model->_feature_size);
    // test model's efficiency

#ifdef _PROFILE
    ProfilerStart("qs.prof");
#endif

    const int C = 2;
    for (int i = 0; i < C; i++) {
        for (int j = 0; j < samples.size(); j++) {
            qs_model->predict(samples[j]);
        }
    }

#ifdef _PROFILE
    ProfilerStop();
#endif

    return 0;
}

