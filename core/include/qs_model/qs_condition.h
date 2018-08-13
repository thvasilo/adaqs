//
// Created by qfeng on 17-10-13.
//

#ifndef QUICKSCORER_QS_CONTENT_H
#define QUICKSCORER_QS_CONTENT_H

#include <sparsehash/dense_hash_map>
#include "conf/config.h"

namespace quickscorer {

    class QSConditions {
    public:
        std::vector<float> _thresholds;
        std::vector<uint32_t> _tree_ids;
        std::vector<BANDWITH_TYPE> _bitvectors;
        google::dense_hash_map<uint64_t, std::pair<uint32_t, uint32_t>> _offsets;

        QSConditions() {
            _offsets.set_empty_key(std::numeric_limits<uint64_t>::max());
        }
    };

}


#endif //QUICKSCORER_QS_CONTENT_H
