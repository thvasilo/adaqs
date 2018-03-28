//
// Created by qfeng on 17-10-9.
//

#include <fstream>
#include <gtest/gtest.h>
#include "gbtree_model/gbtree_model.h"
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include "qs_model/qs_model.h"

using namespace std;

namespace quickscorer {

    class TestQSModel : public ::testing::Test {
    protected:

        virtual void SetUp() {
        }

        virtual void TearDown() {
        }

        void predict(string input, int block_size, bool from_text) {
            std::ifstream samples;
            std::ifstream preds;

            samples.open(input + "/data.test");
            preds.open(input + "/data.pred");

            auto gbtree_model = GBTreeModel::from_xgb_txt(input + "/model.dump");
            unique_ptr<QSModel> qs_model = nullptr;
            if(from_text) qs_model = QSModel::from_xgb_txt(input + "/model.dump", block_size);
            else qs_model = QSModel::from_xgb_bin(input + "/model.bin", block_size);

            string sample_line;
            string pred_line;
            while (!samples.eof() && !preds.eof()) {
                // read data
                getline(samples, sample_line);
                getline(preds, pred_line);
                if (sample_line.length() == 0 || pred_line.length() == 0) continue;
                // parse sample to features
                Eigen::SparseVector<float> features(gbtree_model->_feature_size);
                vector<string> splits;
                boost::algorithm::split(splits, sample_line, boost::algorithm::is_any_of(" "));
                for (int i = 1; i < splits.size(); ++i) {
                    vector<string> sub_splits;
                    boost::algorithm::split(sub_splits, splits[i], boost::algorithm::is_any_of(":"));
                    auto f_idx = boost::lexical_cast<uint64_t>(sub_splits[0]);
                    auto f_val = boost::lexical_cast<float>(sub_splits[1]);
                    if (f_idx < gbtree_model->_feature_size) {
                        features.coeffRef(f_idx) = f_val;
                    }
                }
                // parse pre-predicted values
                auto pre_predicted = boost::lexical_cast<float>(pred_line);
                // gbtree predicted values
                auto gbtree_predicted = gbtree_model->predict(features);
                // quickscorer predicted values
                auto qs_predicted = qs_model->predict(features);
                // check consistency
                EXPECT_NEAR(gbtree_predicted, qs_predicted, 1E-6);
                EXPECT_NEAR(pre_predicted, qs_predicted, 1E-4);
            }

            if (samples.is_open()) samples.close();
            if (preds.is_open()) preds.close();
        }


        void batch_predict(string input, int block_size, int batch_size, bool from_text) {
            std::ifstream samples;
            std::ifstream preds;

            samples.open(input + "/data.test");
            preds.open(input + "/data.pred");

            auto gbtree_model = GBTreeModel::from_xgb_txt(input + "/model.dump");
            unique_ptr<QSModel> qs_model = nullptr;
            if(from_text) qs_model = QSModel::from_xgb_txt(input + "/model.dump", block_size);
            else qs_model = QSModel::from_xgb_bin(input + "/model.bin", block_size);

            string sample_line;
            string pred_line;
            int curr_size = 0;
            vector<unique_ptr<vector<Eigen::SparseVector<float>>>> batch_samples;
            vector<unique_ptr<vector<float>>> batch_preds;
            unique_ptr<vector<Eigen::SparseVector<float>>> batch_s;
            unique_ptr<vector<float>> batch_p;
            while (!samples.eof() && !preds.eof()) {
                // read data
                getline(samples, sample_line);
                getline(preds, pred_line);
                if (sample_line.length() == 0 || pred_line.length() == 0) continue;
                // parse sample to features
                Eigen::SparseVector<float> features(gbtree_model->_feature_size);
                vector<string> splits;
                boost::algorithm::split(splits, sample_line, boost::algorithm::is_any_of(" "));
                for (int i = 1; i < splits.size(); ++i) {
                    vector<string> sub_splits;
                    boost::algorithm::split(sub_splits, splits[i], boost::algorithm::is_any_of(":"));
                    auto f_idx = boost::lexical_cast<uint64_t>(sub_splits[0]);
                    auto f_val = boost::lexical_cast<float>(sub_splits[1]);
                    if (f_idx < gbtree_model->_feature_size) {
                        features.coeffRef(f_idx) = f_val;
                    }
                }
                // parse pre-predicted values
                auto pre_predicted = boost::lexical_cast<float>(pred_line);
                // create batch samples and preds
                if (curr_size == 0) {
                    batch_s = unique_ptr<vector<Eigen::SparseVector<float>>>(new vector<Eigen::SparseVector<float>>());
                    batch_p = unique_ptr<vector<float>>(new vector<float>());
                }
                batch_s->emplace_back(features);
                batch_p->emplace_back(pre_predicted);
                if (++curr_size >= batch_size) {
                    batch_samples.emplace_back(std::move(batch_s));
                    batch_preds.emplace_back(std::move(batch_p));
                    curr_size = 0;
                }
            }
            if(batch_s && batch_s->size() > 0) {
                batch_samples.emplace_back(std::move(batch_s));
                batch_preds.emplace_back(std::move(batch_p));
            }
            // check prediction correctness
            for(int i = 0; i < batch_samples.size(); ++i) {
                // pre-predicted values
                auto pre_predicted = std::move(batch_preds[i]);
                // gbtree predicted values
                auto gbtree_predicted = gbtree_model->predict(*batch_samples[i]);
                // quickscorer predicted values
                auto qs_predicted = qs_model->predict(*batch_samples[i]);
                // check consistency
                for(int j = 0; j < gbtree_predicted.size(); ++j) {
                    EXPECT_NEAR(gbtree_predicted[j], qs_predicted[j], 1E-6);
                    EXPECT_NEAR((*pre_predicted)[j], qs_predicted[j], 1E-4);
                }
            }
            // close files
            if (samples.is_open()) samples.close();
            if (preds.is_open()) preds.close();
        }

    };


    TEST_F(TestQSModel, predict_from_text) {
        predict("tests/data/agaricus", 2, true);
        predict("tests/data/avazu-app-1000", 1001, true);
        predict("tests/data/avazu-app-1000", 501, true);
        predict("tests/data/avazu-app-1000", 101, true);
    }

    TEST_F(TestQSModel, batch_predict_from_text) {
        batch_predict("tests/data/avazu-app-1000", 1001, 10, true);
    }

    TEST_F(TestQSModel, predict_from_bin) {
        predict("tests/data/agaricus", 2, false);
        predict("tests/data/avazu-app-1000", 1001, false);
        predict("tests/data/avazu-app-1000", 501, false);
        predict("tests/data/avazu-app-1000", 101, false);
    }

    TEST_F(TestQSModel, batch_predict_from_bin) {
        batch_predict("tests/data/avazu-app-1000", 1001, 10, false);
    }

}
