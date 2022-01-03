#include "mmutil.hh"
#include "mmutil_spectral.hh"
#include "mmutil_match.hh"
#include "mmutil_util.hh"
#include "mmutil_index.hh"
#include "progress.hh"

#ifndef MMUTIL_BBKNN_HH_
#define MMUTIL_BBKNN_HH_

SpMat build_bbknn(const svd_out_t &svd,
                  const std::vector<std::vector<Index>> &batch_index_set,
                  const std::size_t knn,
                  const std::size_t KNN_BILINK,
                  const std::size_t KNN_NNLIST);

#endif