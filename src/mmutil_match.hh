#include <getopt.h>

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#include "hnswlib.h"
#include "mmutil.hh"
#include "mmutil_stat.hh"
#include "progress.hh"
#include "tuple_util.hh"
#include "io.hh"

// [[Rcpp::plugins(openmp)]]

#ifndef MMUTIL_MATCH_HH_
#define MMUTIL_MATCH_HH_

/////////////////////////////////
// k-nearest neighbor matching //
/////////////////////////////////

using KnnAlg = hnswlib::HierarchicalNSW<Scalar>;

struct KNN {
    explicit KNN(const std::size_t _val)
        : val(_val)
    {
    }
    const std::size_t val;
};

struct BILINK {
    explicit BILINK(const std::size_t _val)
        : val(_val)
    {
    }
    const std::size_t val;
};

struct NNLIST {
    explicit NNLIST(const std::size_t _val)
        : val(_val)
    {
    }
    const std::size_t val;
};

using index_triplet_vec = std::vector<std::tuple<Index, Index, Scalar>>;

struct SrcDataT {
    explicit SrcDataT(const Scalar *_data, const Index d, const Index s)
        : data(_data)
        , vecdim(d)
        , vecsize(s)
    {
    }
    const Scalar *data;
    const Index vecdim;
    const Index vecsize;
};

struct TgtDataT {
    explicit TgtDataT(const Scalar *_data, const Index d, const Index s)
        : data(_data)
        , vecdim(d)
        , vecsize(s)
    {
    }
    const Scalar *data;
    const Index vecdim;
    const Index vecsize;
};

///////////////////////////////////
// search over the dense data	 //
// 				 //
// each column = each data point //
///////////////////////////////////

/**
   @param SrcDataT each column = each data point
   @param TgtDataT each column = each data point
   @param KNN number of neighbours
   @param BILINK the size bidirectional list
   @param NNlist the size of neighbouring list
   @param OUT
 */
template <typename VS>
int
search_knn(const SrcDataT _SrcData,       //
           const TgtDataT _TgtData,       //
           const KNN _knn,                //
           const BILINK _bilink,          //
           const NNLIST _nnlist,          //
           const std::size_t NUM_THREADS, //
           index_triplet_vec &out)
{
    ERR_RET(_SrcData.vecdim != _TgtData.vecdim,
            "source and target must have the same dimensionality");

    const std::size_t knn = _knn.val;
    const std::size_t vecdim = _TgtData.vecdim;
    const std::size_t vecsize = _TgtData.vecsize;

    std::size_t param_bilink = _bilink.val;
    std::size_t param_nnlist = _nnlist.val;

    if (param_bilink >= vecdim) {
        WLOG("too big M value: " << param_bilink << " vs. " << vecdim);
        param_bilink = vecdim - 1;
    }

    if (param_bilink < 2) {
        WLOG("too small M value");
        param_bilink = 2;
    }

    if (param_nnlist <= knn) {
        WLOG("too small N value");
        param_nnlist = knn + 1;
    }

    // Construct KnnAlg interface
    VS vecspace(vecdim);
    // hnswlib::InnerProductSpace vecspace(vecdim);
    // hnswlib::L2Space vecspace(vecdim);

    KnnAlg alg(&vecspace, vecsize, param_bilink, param_nnlist);
    alg.ef_ = param_nnlist;

    TLOG("Initializing kNN algorithm");

    {
        const Scalar *mass = _TgtData.data;

        progress_bar_t<Index> prog(vecsize, 1e2);

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
        for (Index i = 0; i < vecsize; ++i) {
#pragma omp critical
            {
                alg.addPoint((void *)(mass + vecdim * i),
                             static_cast<std::size_t>(i));
                prog.update();
                prog(Rcpp::Rcerr);
            }
        }
    }

    ////////////
    // recall //
    ////////////

    {
        const Index N = _SrcData.vecsize;
        TLOG("Finding " << knn << " nearest neighbors for N = " << N);

        const Scalar *mass = _SrcData.data;
        progress_bar_t<Index> prog(_SrcData.vecsize, 1e2);

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
        for (Index i = 0; i < _SrcData.vecsize; ++i) {
#pragma omp critical
            {
                auto pq = alg.searchKnn((void *)(mass + vecdim * i), knn);
                Scalar d = 0;
                std::size_t j;
                while (!pq.empty()) {
                    std::tie(d, j) = pq.top();
                    out.emplace_back(i, j, d);
                    pq.pop();
                }
                prog.update();
                prog(Rcpp::Rcerr);
            }
        }
    }
    TLOG("Done kNN searches");
    return EXIT_SUCCESS;
}

/**
 * @param deg_i number of elements
 * @param dist deg_i-vector for distance
 * @param weights deg_i-vector for weights

 Since the inner-product distance is d(x,y) = (1 - x'y),
 d = 0.5 * (x - y)'(x - y) = 0.5 * (x'x + y'y) - x'y,
 we have Gaussian weight w(x,y) = exp(-lambda * d(x,y))

 */
void normalize_weights(const Index deg_i,
                       std::vector<Scalar> &dist,
                       std::vector<Scalar> &weights);

template <typename TVec, typename SVec>
auto
build_knn_named(const TVec &out_index,     //
                const SVec &col_src_names, //
                const SVec &col_tgt_names)
{
    using RET = std::vector<std::tuple<std::string, std::string, Scalar>>;

    RET out_named;
    out_named.reserve(out_index.size());

    for (auto tt : out_index) {
        Index i, j;
        Scalar d;
        std::tie(i, j, d) = tt;
        out_named.push_back(
            std::make_tuple(col_src_names.at(i), col_tgt_names.at(j), d));
    }

    return out_named;
}

template <typename TVec, typename SVec, typename VVec>
auto
build_knn_named(const TVec &out_index,     //
                const SVec &col_src_names, //
                const SVec &col_tgt_names, //
                const VVec &valid_src,     //
                const VVec &valid_tgt)
{
    using RET = std::vector<std::tuple<std::string, std::string, Scalar>>;

    RET out_named;
    out_named.reserve(out_index.size());

    for (auto tt : out_index) {
        Index i, j;
        Scalar d;
        std::tie(i, j, d) = tt;
        if (valid_src.count(i) > 0 && valid_tgt.count(j) > 0) {
            out_named.push_back(
                std::make_tuple(col_src_names.at(i), col_tgt_names.at(j), d));
        }
    }

    return out_named;
}

//////////////////////
// non-zero columns //
//////////////////////

inline std::tuple<std::unordered_set<Index>, Index>
find_nz_cols(const std::string mtx_file);

///////////////////////////
// non-zero column names //
///////////////////////////

inline std::tuple<std::unordered_set<Index>, // valid
                  Index,                     // #total
                  std::vector<std::string>   // names
                  >
find_nz_col_names(const std::string mtx_file,
                  const std::string col_file,
                  const std::size_t,
                  const char);

//////////////////////
// reciprocal match //
//////////////////////

template <typename T>
inline std::vector<T>
keep_reciprocal_knn(const std::vector<T> &knn_index, bool undirected = false)
{
    // Make sure that we could only consider reciprocal kNN pairs
    std::unordered_map<std::tuple<Index, Index>,
                       short,
                       hash_tuple::hash<std::tuple<Index, Index>>>
        edge_count;

    auto _count = [&edge_count](const auto &tt) {
        Index i, j, temp;
        std::tie(i, j, std::ignore) = parse_triplet(tt);
        if (i == j)
            return;

        if (i > j) {
            temp = i;
            i = j;
            j = temp;
        }

        if (edge_count.count({ i, j }) < 1) {
            edge_count[{ i, j }] = 1;
        } else {
            edge_count[{ i, j }] += 1;
        }
    };

    std::for_each(knn_index.begin(), knn_index.end(), _count);

    auto is_mutual = [&edge_count, &undirected](const auto &tt) {
        Index i, j, temp;
        std::tie(i, j, std::ignore) = parse_triplet(tt);
        if (i == j)
            return false;
        if (i > j) {
            temp = i;
            i = j;
            j = temp;
        }
        if (undirected)
            return (edge_count[{ i, j }] > 1) && (i <= j);
        return (edge_count[{ i, j }] > 1);
    };

    std::vector<T> reciprocal_knn_index;
    reciprocal_knn_index.reserve(knn_index.size());
    std::copy_if(knn_index.begin(),
                 knn_index.end(),
                 std::back_inserter(reciprocal_knn_index),
                 is_mutual);

    return reciprocal_knn_index;
}

#endif
