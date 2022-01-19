#include "mmutil.hh"
#include "mmutil_spectral.hh"
#include "mmutil_match.hh"
#include "mmutil_util.hh"
#include "mmutil_index.hh"
#include "progress.hh"

#ifndef MMUTIL_BBKNN_HH_
#define MMUTIL_BBKNN_HH_

int build_bbknn(const svd_out_t &svd,
                const std::vector<std::vector<Index>> &batch_index_set,
                const std::size_t knn,
                std::vector<std::tuple<Index, Index, Scalar>> &knn_index,
                const std::size_t KNN_BILINK,
                const std::size_t KNN_NNLIST,
                const std::size_t NUM_THREADS);

#endif
