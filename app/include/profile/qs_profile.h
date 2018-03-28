//
// Created by qfeng on 17-10-12.
//

#ifndef QUICKSCORER_QS_PROFILE_H
#define QUICKSCORER_QS_PROFILE_H

#include <string>
#include <Eigen/Sparse>
#include <memory>

using namespace std;


namespace quickscorer_profile {

    vector<Eigen::SparseVector<float>> load_samples(string input_path, int feature_size);

    vector<unique_ptr<vector<Eigen::SparseVector<float>>>>
    split_to_batches(vector<Eigen::SparseVector<float>> &input, int batch_size, int repeat_times);

}


#endif //QUICKSCORER_QS_PROFILE_H
