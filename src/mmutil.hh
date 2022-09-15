// #include <boost/graph/adjacency_list.hpp>
// #include <boost/graph/connected_components.hpp>
// #include <boost/lexical_cast.hpp>

// [[Rcpp::plugins(cpp14)]]
#include <Rcpp.h>
// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
// [[Rcpp::depends(RcppProgress)]]

// [[Rcpp::plugins(openmp)]]
#include <omp.h>

// [[Rcpp::depends(dqrng, sitmo, BH)]]
#include <dqrng.h>
#include <dqrng_distribution.h>
#include <boost/random/binomial_distribution.hpp>
#include <boost/random/poisson_distribution.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <xoshiro.h>

#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>
#include <string>
#include <tuple>
#include <utility>
#include <unordered_map>

#include "eigen_util.hh"
#include "std_util.hh"
#include "math.hh"
#include "util.hh"
#include "check.hh"

#ifdef __cplusplus
extern "C" {
#endif

#include "bgzf.h"
#include "kstring.h"

#ifdef __cplusplus
}
#endif

#ifndef MMUTIL_HH_
#define MMUTIL_HH_

using Scalar = float;
using SpMat = Eigen::SparseMatrix<Scalar, Eigen::RowMajor, std::ptrdiff_t>;
using Index = SpMat::Index;
using MSpMat = Eigen::MappedSparseMatrix<Scalar>;

using Mat = typename Eigen::
    Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using Vec = typename Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using IntMat = typename Eigen::
    Matrix<std::ptrdiff_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using IntVec = typename Eigen::Matrix<std::ptrdiff_t, Eigen::Dynamic, 1>;

/////////////////////////////
// simple helper functions //
/////////////////////////////

std::tuple<Index, Index, Scalar>
parse_triplet(const std::tuple<Index, Index, Scalar> &tt);

std::tuple<Index, Index, Scalar>
parse_triplet(const Eigen::Triplet<Scalar> &tt);

std::vector<std::string> copy(const Rcpp::StringVector &r_vec);

void copy(const Rcpp::StringVector &r_vec, std::vector<std::string> &vec);

#endif
