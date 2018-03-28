//
// Created by qfeng on 17-10-9.
//

#include <fstream>
#include <gtest/gtest.h>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <xgboost-predictor/predictor.h>
#include <gbtree_model/gbtree_model.h>
#include <dmlc/io.h>

using namespace std;

namespace quickscorer {

    class TestXgbPredictor : public ::testing::Test {
    protected:

        virtual void SetUp() {

        }

        virtual void TearDown() {

        }
    };

    void predict(string data_set, int num_trees) {
        std::ifstream _samples;
        std::ifstream _preds;

        _samples.open("tests/data/" + data_set + "/data.test");
        _preds.open("tests/data/" + data_set + "/data.pred");

        auto gbtree_model = GBTreeModel::from_xgb_txt("tests/data/" + data_set + "/model.dump");

        auto xgb_predictor = unique_ptr<xgboost::Predictor>(new xgboost::Predictor());
        xgb_predictor->Load("tests/data/" + data_set + "/model.bin");

        string sample_line;
        string pred_line;
        while (!_samples.eof() && !_preds.eof()) {
            // read data
            getline(_samples, sample_line);
            getline(_preds, pred_line);
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
            // parse pre-predicted values;
            auto pre_predicted = boost::lexical_cast<float>(pred_line);
            // gbtree predict values
            auto gbtree_predicted = gbtree_model->predict(features);
            // xgb predict values
            auto xgb_predicted = xgb_predictor->Predict(features, false, num_trees);
            // check consistency
            EXPECT_NEAR(gbtree_predicted, xgb_predicted, 1E-6);
            EXPECT_NEAR(pre_predicted, xgb_predicted, 1E-6);
        }

        if (_samples.is_open()) _samples.close();
        if (_preds.is_open()) _preds.close();
    }

    TEST_F(TestXgbPredictor, predict) {
        predict("agaricus", 2);
        predict("avazu-app-50", 50);
        predict("avazu-app-1000", 1000);
    }

}
