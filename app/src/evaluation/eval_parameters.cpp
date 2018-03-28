//
// Created by qfeng on 17-10-17.
//

#include "log/qs_log.h"
#include <boost/filesystem.hpp>
#include <gbtree_model/gbtree_model.h>
#include <qs_model/qs_model.h>
#include <iostream>
#include "profile/qs_profile.h"

using namespace std;
using namespace quickscorer;
using namespace quickscorer_profile;
namespace fs=boost::filesystem;


int main(int argc, char **argv) {
    // check arguments
    if (argc != 4) {
        std::cout << "Specify 2 arguments: (1) block size, (2) batch size and (3) repeat times." << std::endl;
        return 1;
    }
    int block_size = atoi(argv[1]);
    int batch_size = atoi(argv[2]);
    int repeat_times = atoi(argv[3]);
    // initialize logger
    string logger_name = "qs_app_eval_logger";
    string logger_dir_name = "qs_logs/app";
    string logger_file_name = logger_dir_name + "/app_eval_log";
    auto logger_dir = fs::path(logger_dir_name);
    QSLog::create_loggers(logger_name, logger_file_name);
    // initialize models
    string model_file = "tests/data/avazu-app-5000/model.dump";
    // load samples by batches
    auto samples = load_samples("tests/data/avazu-app-5000/data.test", 999963);
    auto batches = split_to_batches(samples, batch_size, repeat_times);
    // load qs model
    auto qs_ada_model = QSModel::from_xgb_txt(model_file, block_size, true, 50000);
    //evaluate the execution time
    auto begin_time = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < batches.size(); k++) {
        qs_ada_model->predict(*batches[k]);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - begin_time;
    std::cout << "average wall time with block_size: " << block_size << ", batch_size "
              << batch_size << ": " << diff.count() * 1000 / samples.size() / repeat_times << " ms of "
              << samples.size() << " samples repeat " << repeat_times << " times." << std::endl;
}

