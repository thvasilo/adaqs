//
// Created by qfeng on 17-9-30.
//

#include <gtest/gtest.h>
#include <gbtree_model/gbtree_model.h>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include "gbtree_model/leaf_node.h"

using namespace std;

namespace quickscorer {


    class TestGBTreeModelFromText : public ::testing::Test {
    protected:
        std::ifstream _samples;
        std::ifstream _preds;

        virtual void SetUp() {
            _samples.open("tests/data/agaricus/data.test");
            _preds.open("tests/data/agaricus/data.pred");
        }

        virtual void TearDown() {
            if (_samples.is_open()) _samples.close();
            if (_preds.is_open()) _preds.close();
        }

    };

    TEST_F(TestGBTreeModelFromText, from_xgb_txt) {
        auto model = GBTreeModel::from_xgb_txt("tests/data/agaricus/model.dump");

        // check the number of trees
        EXPECT_EQ(model->_trees.size(), 2);

        // check the first tree
        {
            auto node_0 = model->_trees[0].get();
            EXPECT_EQ(node_0->_split_feature, 29);
            EXPECT_NEAR(node_0->_split_value, 0, 1E-6);
            EXPECT_TRUE(node_0->_default_to_left);
            EXPECT_EQ(node_0->_depth, 0);

            auto node_1 = dynamic_cast<InnerNode *>(node_0->_left_child.get());
            EXPECT_EQ(node_1->_split_feature, 56);
            EXPECT_NEAR(node_1->_split_value, 0, 1E-6);
            EXPECT_TRUE(node_1->_default_to_left);
            EXPECT_EQ(node_1->_depth, 1);

            auto node_3 = dynamic_cast<InnerNode *>(node_1->_left_child.get());
            EXPECT_EQ(node_3->_split_feature, 60);
            EXPECT_NEAR(node_3->_split_value, 0, 1E-6);
            EXPECT_TRUE(node_3->_default_to_left);
            EXPECT_EQ(node_3->_depth, 2);

            auto node_7 = dynamic_cast<LeafNode *>(node_3->_left_child.get());
            EXPECT_NEAR(node_7->_value, 1.90175, 1E-6);
            EXPECT_EQ(node_7->_depth, 3);

            auto node_8 = dynamic_cast<LeafNode *>(node_3->_right_child.get());
            EXPECT_NEAR(node_8->_value, -1.95062, 1E-6);
            EXPECT_EQ(node_8->_depth, 3);

            auto node_4 = dynamic_cast<InnerNode *>(node_1->_right_child.get());
            EXPECT_EQ(node_4->_split_feature, 21);
            EXPECT_NEAR(node_4->_split_value, 0, 1E-6);
            EXPECT_TRUE(node_4->_default_to_left);
            EXPECT_EQ(node_4->_depth, 2);

            auto node_9 = dynamic_cast<LeafNode *>(node_4->_left_child.get());
            EXPECT_NEAR(node_9->_value, 1.77778, 1E-6);
            EXPECT_EQ(node_9->_depth, 3);

            auto node_10 = dynamic_cast<LeafNode *>(node_4->_right_child.get());
            EXPECT_NEAR(node_10->_value, -1.98104, 1E-6);
            EXPECT_EQ(node_10->_depth, 3);

            auto node_2 = dynamic_cast<InnerNode *>(node_0->_right_child.get());
            EXPECT_EQ(node_2->_split_feature, 109);
            EXPECT_NEAR(node_2->_split_value, 0, 1E-6);
            EXPECT_TRUE(node_2->_default_to_left);
            EXPECT_EQ(node_2->_depth, 1);

            auto node_5 = dynamic_cast<InnerNode *>(node_2->_left_child.get());
            EXPECT_EQ(node_5->_split_feature, 67);
            EXPECT_NEAR(node_5->_split_value, 0, 1E-6);
            EXPECT_TRUE(node_5->_default_to_left);
            EXPECT_EQ(node_5->_depth, 2);

            auto node_11 = dynamic_cast<LeafNode *>(node_5->_left_child.get());
            EXPECT_NEAR(node_11->_value, -1.98531, 1E-6);
            EXPECT_EQ(node_11->_depth, 3);

            auto node_12 = dynamic_cast<LeafNode *>(node_5->_right_child.get());
            EXPECT_NEAR(node_12->_value, 0.808511, 1E-6);
            EXPECT_EQ(node_12->_depth, 3);

            auto node_6 = dynamic_cast<LeafNode *>(node_2->_right_child.get());
            EXPECT_NEAR(node_6->_value, 1.85965, 1E-6);
            EXPECT_EQ(node_6->_depth, 2);
        }

        // check the second tree
        {
            auto node_0 = model->_trees[1].get();
            EXPECT_EQ(node_0->_split_feature, 29);
            EXPECT_NEAR(node_0->_split_value, 0, 1E-6);
            EXPECT_TRUE(node_0->_default_to_left);
            EXPECT_EQ(node_0->_depth, 0);

            auto node_1 = dynamic_cast<InnerNode *>(node_0->_left_child.get());
            EXPECT_EQ(node_1->_split_feature, 21);
            EXPECT_NEAR(node_1->_split_value, 0, 1E-6);
            EXPECT_TRUE(node_1->_default_to_left);
            EXPECT_EQ(node_1->_depth, 1);

            auto node_3 = dynamic_cast<LeafNode *>(node_1->_left_child.get());
            EXPECT_NEAR(node_3->_value, 1.1457, 1E-6);
            EXPECT_EQ(node_3->_depth, 2);

            auto node_4 = dynamic_cast<InnerNode *>(node_1->_right_child.get());
            EXPECT_EQ(node_4->_split_feature, 36);
            EXPECT_NEAR(node_4->_split_value, 0, 1E-6);
            EXPECT_TRUE(node_4->_default_to_left);
            EXPECT_EQ(node_4->_depth, 2);

            auto node_7 = dynamic_cast<LeafNode *>(node_4->_left_child.get());
            EXPECT_NEAR(node_7->_value, -6.87558, 1E-6);
            EXPECT_EQ(node_7->_depth, 3);

            auto node_8 = dynamic_cast<LeafNode *>(node_4->_right_child.get());
            EXPECT_NEAR(node_8->_value, -0.127376, 1E-6);
            EXPECT_EQ(node_8->_depth, 3);

            auto node_2 = dynamic_cast<InnerNode *>(node_0->_right_child.get());
            EXPECT_EQ(node_2->_split_feature, 109);
            EXPECT_NEAR(node_2->_split_value, 0, 1E-6);
            EXPECT_TRUE(node_2->_default_to_left);
            EXPECT_EQ(node_2->_depth, 1);

            auto node_5 = dynamic_cast<InnerNode *>(node_2->_left_child.get());
            EXPECT_EQ(node_5->_split_feature, 39);
            EXPECT_NEAR(node_5->_split_value, 0, 1E-6);
            EXPECT_TRUE(node_5->_default_to_left);
            EXPECT_EQ(node_5->_depth, 2);

            auto node_9 = dynamic_cast<LeafNode *>(node_5->_left_child.get());
            EXPECT_NEAR(node_9->_value, -0.0386054, 1E-6);
            EXPECT_EQ(node_9->_depth, 3);

            auto node_10 = dynamic_cast<LeafNode *>(node_5->_right_child.get());
            EXPECT_NEAR(node_10->_value, -1.15275, 1E-6);
            EXPECT_EQ(node_10->_depth, 3);

            auto node_6 = dynamic_cast<LeafNode *>(node_2->_right_child.get());
            EXPECT_NEAR(node_6->_value, 0.994744, 1E-6);
            EXPECT_EQ(node_6->_depth, 2);
        }

        // check features
        {
            EXPECT_EQ(model->_feature_size, 110);

            EXPECT_EQ(model->_tree_features[0]->size(), 6);
            EXPECT_TRUE(model->_tree_features[0]->find(29) != model->_tree_features[0]->end());
            EXPECT_TRUE(model->_tree_features[0]->find(56) != model->_tree_features[0]->end());
            EXPECT_TRUE(model->_tree_features[0]->find(60) != model->_tree_features[0]->end());
            EXPECT_TRUE(model->_tree_features[0]->find(21) != model->_tree_features[0]->end());
            EXPECT_TRUE(model->_tree_features[0]->find(109) != model->_tree_features[0]->end());
            EXPECT_TRUE(model->_tree_features[0]->find(67) != model->_tree_features[0]->end());

            EXPECT_EQ(model->_tree_features[1]->size(), 5);
            EXPECT_TRUE(model->_tree_features[1]->find(29) != model->_tree_features[0]->end());
            EXPECT_TRUE(model->_tree_features[1]->find(21) != model->_tree_features[0]->end());
            EXPECT_TRUE(model->_tree_features[1]->find(36) != model->_tree_features[0]->end());
            EXPECT_TRUE(model->_tree_features[1]->find(109) != model->_tree_features[0]->end());
            EXPECT_TRUE(model->_tree_features[1]->find(39) != model->_tree_features[0]->end());
        }
    }

    TEST_F(TestGBTreeModelFromText, num_trees) {
        auto model1 = GBTreeModel::from_xgb_txt("tests/data/agaricus/model.dump", 1);
        EXPECT_EQ(model1->_trees.size(), 1);
        auto model2 = GBTreeModel::from_xgb_txt("tests/data/agaricus/model.dump", 2);
        EXPECT_EQ(model2->_trees.size(), 2);
        auto model3 = GBTreeModel::from_xgb_txt("tests/data/agaricus/model.dump", 0);
        EXPECT_EQ(model3->_trees.size(), 2);
    }

    TEST_F(TestGBTreeModelFromText, predict) {
        auto model = GBTreeModel::from_xgb_txt("tests/data/agaricus/model.dump");

        string sample_line;
        string pred_line;
        while (!_samples.eof() && !_preds.eof()) {
            // read data
            getline(_samples, sample_line);
            getline(_preds, pred_line);
            if (sample_line.length() == 0 || pred_line.length() == 0) continue;
            // parse sample to features
            Eigen::SparseVector<float> features(model->_feature_size);
            vector<string> splits;
            boost::algorithm::split(splits, sample_line, boost::algorithm::is_any_of(" "));
            for (int i = 1; i < splits.size(); ++i) {
                vector<string> sub_splits;
                boost::algorithm::split(sub_splits, splits[i], boost::algorithm::is_any_of(":"));
                auto f_idx = boost::lexical_cast<uint64_t>(sub_splits[0]);
                auto f_val = boost::lexical_cast<float>(sub_splits[1]);
                if (f_idx < model->_feature_size) {
                    features.coeffRef(f_idx) = f_val;
                }
            }
            // parse pre-predicted values;
            auto pre_predicted = boost::lexical_cast<float>(pred_line);
            // predict values
            auto predicted = model->predict(features);
            // check consistency
            EXPECT_NEAR(pre_predicted, predicted, 1E-6);
        }
    }
}