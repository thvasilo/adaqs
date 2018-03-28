//
// Created by qfeng on 17-10-12.
//

#include "profile/qs_profile.h"
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>


namespace quickscorer_profile {

    vector<Eigen::SparseVector<float>> load_samples(string input_path, int feature_size) {
        std::ifstream samples_f;
        samples_f.open(input_path);

        vector<Eigen::SparseVector<float>> samples;

        string line;
        while (!samples_f.eof()) {
            // read data
            getline(samples_f, line);
            if (line.length() == 0) continue;
            // parse data to features
            Eigen::SparseVector<float> features(feature_size);
            vector<string> splits;
            boost::algorithm::split(splits, line, boost::algorithm::is_any_of(" "));
            for (int i = 1; i < splits.size(); ++i) {
                if (splits[i].length() == 0) continue;
                vector<string> sub_splits;
                boost::algorithm::split(sub_splits, splits[i], boost::algorithm::is_any_of(":"));
                auto f_idx = boost::lexical_cast<uint64_t>(sub_splits[0]);
                auto f_val = boost::lexical_cast<float>(sub_splits[1]);
                if (f_idx < feature_size) {
                    features.coeffRef(f_idx) = f_val;
                }
            }
            samples.push_back(features);
        }
        if (samples_f.is_open()) samples_f.close();
        return samples;
    }

    vector<unique_ptr<vector<Eigen::SparseVector<float>>>>
    split_to_batches(vector<Eigen::SparseVector<float>> &input, int batch_size, int repeat_times) {
        vector<unique_ptr<vector<Eigen::SparseVector<float>>>> results;

        int curr_size = 0;
        unique_ptr<vector<Eigen::SparseVector<float>>> batch;
        for (int j = 0; j < repeat_times; ++j) {
            for (int i = 0; i < input.size(); ++i) {
                if (curr_size == 0) {
                    batch = unique_ptr<vector<Eigen::SparseVector<float>>>(new vector<Eigen::SparseVector<float>>());
                }
                batch->emplace_back(input[i]);
                if (++curr_size >= batch_size) {
                    results.emplace_back(std::move(batch));
                    curr_size = 0;
                }
            }
        }
        if (curr_size > 0) {
            results.emplace_back(std::move((batch)));
        }

        return results;
    }

}