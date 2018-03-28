//
// Created by qfeng on 17-10-12.
//

#include "log/qs_log.h"
#include <boost/filesystem.hpp>
#include <gbtree_model/gbtree_model.h>
#include <qs_model/qs_model.h>
#include <iostream>
#include "profile/qs_profile.h"
#include <xgboost-predictor/predictor.h>

using namespace std;
using namespace quickscorer;
using namespace quickscorer_profile;
namespace fs=boost::filesystem;


int main(int argc, char **argv) {
    // check arguments
    if (argc != 4) {
        std::cout << "Specify 3 arguments: (1) data set name (/tests/data/...), (2) repeat times and (3) tree numbers."
                  << std::endl;
        return 1;
    }
    const string data_name = argv[1];
    int C = atoi(argv[2]);
    int num_trees = atoi(argv[3]);
    std::cout << "data set: " << data_name << ", C = " << C << ", trees = " << num_trees << std::endl;
    // initialize logger
    string logger_name = "qs_app_eval_logger";
    string logger_dir_name = "qs_logs/app";
    string logger_file_name = logger_dir_name + "/app_eval_log";
    auto logger_dir = fs::path(logger_dir_name);
    QSLog::create_loggers(logger_name, logger_file_name);
    // initialize models
    string model_file = "tests/data/" + data_name + "/model.dump";
    auto gbtree_model = GBTreeModel::from_xgb_txt(model_file, num_trees);
    auto qs_ada_model = QSModel::from_xgb_txt(model_file, 10000, true, num_trees);
    auto qs_plain_model = QSModel::from_xgb_txt(model_file, 10000, false, num_trees);
    auto xgb_predictor = unique_ptr<xgboost::Predictor>(new xgboost::Predictor());
    xgb_predictor->Load("tests/data/" + data_name + "/model.bin");
    // load samples
    auto samples = load_samples("tests/data/" + data_name + "/data.test", gbtree_model->_feature_size);
    //evaluate the execution time of gbtree model
    auto begin_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < C; i++) {
        for (int j = 0; j < samples.size(); j++) {
            gbtree_model->predict(samples[j]);
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - begin_time;
    std::cout << "average wall time of gbtree model: " << diff.count() * 1000 / C / samples.size() << " ms"
              << std::endl;
    // evaluate the execution time of adaptive quickscorer model
    begin_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < C; i++) {
        for (int j = 0; j < samples.size(); j++) {
            qs_ada_model->predict(samples[j]);
        }
    }
    end_time = std::chrono::high_resolution_clock::now();
    diff = end_time - begin_time;
    std::cout << "average wall time of adaqs model: " << diff.count() * 1000 / C / samples.size() << " ms" << std::endl;
    // evaluate the execution time of plain quickscorer model
    begin_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < C; i++) {
        for (int j = 0; j < samples.size(); j++) {
            qs_plain_model->predict(samples[j]);
        }
    }
    end_time = std::chrono::high_resolution_clock::now();
    diff = end_time - begin_time;
    std::cout << "average wall time of plain qs model: " << diff.count() * 1000 / C / samples.size() << " ms"
              << std::endl;
    // evaluate the execution time of xgboost predictor
    begin_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < C; i++) {
        for (int j = 0; j < samples.size(); j++) {
            qs_plain_model->predict(samples[j]);
        }
    }
    end_time = std::chrono::high_resolution_clock::now();
    diff = end_time - begin_time;
    std::cout << "average wall time of xgboost predictor: " << diff.count() * 1000 / C / samples.size() << " ms"
              << std::endl << std::endl;

    return 0;
}

