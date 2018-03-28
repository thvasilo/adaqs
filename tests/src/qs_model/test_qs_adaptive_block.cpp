//
// Created by qfeng on 17-10-9.
//

#include <fstream>
#include <gtest/gtest.h>
#include "gbtree_model/gbtree_model.h"
#include "qs_model/qs_plain_block.h"
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <qs_model/qs_adaptive_block.h>

using namespace std;

namespace quickscorer {

    class TestQSAdaBlock : public ::testing::Test {
    protected:

        virtual void SetUp() {

        }

        virtual void TearDown() {
        }

        void predict(string input, int num_trees) {
            std::ifstream samples;
            std::ifstream preds;

            samples.open(input + "/data.test");
            preds.open(input + "/data.pred");

            auto gbtree_model = GBTreeModel::from_xgb_txt(input + "/model.dump");
            auto qmb = QSAdaBlock::from_gbtree_model(*gbtree_model.get(), 0, num_trees);

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
                // quickscorer block predicted values
                auto qs_predicted = qmb->predict(features);
                // check consistency
                EXPECT_NEAR(gbtree_predicted, qs_predicted, 1E-6);
                EXPECT_NEAR(pre_predicted, qs_predicted, 1E-4);
            }

            if (samples.is_open()) samples.close();
            if (preds.is_open()) preds.close();
        }

    };

    TEST_F(TestQSAdaBlock, predict) {
        predict("tests/data/agaricus", 2);
        predict("tests/data/avazu-app-50", 50);
        predict("tests/data/avazu-app-1000", 1000);
        predict("tests/data/avazu-app-5000", 5000);
    }

}
