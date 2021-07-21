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

#ifndef MMUTIL_MATCH_HH_
#define MMUTIL_MATCH_HH_

/////////////////////////////////
// k-nearest neighbor matching //
/////////////////////////////////

using KnnAlg = hnswlib::HierarchicalNSW<float>;

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

using index_triplet_vec = std::vector<std::tuple<Index, Index, float>>;

struct SrcDataT {
    explicit SrcDataT(const float *_data, const Index d, const Index s)
        : data(_data)
        , vecdim(d)
        , vecsize(s)
    {
    }
    const float *data;
    const Index vecdim;
    const Index vecsize;
};

struct TgtDataT {
    explicit TgtDataT(const float *_data, const Index d, const Index s)
        : data(_data)
        , vecdim(d)
        , vecsize(s)
    {
    }
    const float *data;
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
int search_knn(const SrcDataT _SrcData, //
               const TgtDataT _TgtData, //
               const KNN _knn,          //
               const BILINK _bilink,    //
               const NNLIST _nnlist,    //
               index_triplet_vec &out);

/**
 * @param deg_i number of elements
 * @param dist deg_i-vector for distance
 * @param weights deg_i-vector for weights

 Since the inner-product distance is d(x,y) = (1 - x'y),
 d = 0.5 * (x - y)'(x - y) = 0.5 * (x'x + y'y) - x'y,
 we have Gaussian weight w(x,y) = exp(-lambda * d(x,y))

 */
void normalize_weights(const Index deg_i,
                       std::vector<float> &dist,
                       std::vector<float> &weights);

template <typename TVec, typename SVec>
auto
build_knn_named(const TVec &out_index,     //
                const SVec &col_src_names, //
                const SVec &col_tgt_names)
{
    using RET = std::vector<std::tuple<std::string, std::string, float>>;

    RET out_named;
    out_named.reserve(out_index.size());

    for (auto tt : out_index) {
        Index i, j;
        float d;
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
    using RET = std::vector<std::tuple<std::string, std::string, float>>;

    RET out_named;
    out_named.reserve(out_index.size());

    for (auto tt : out_index) {
        Index i, j;
        float d;
        std::tie(i, j, d) = tt;
        if (valid_src.count(i) > 0 && valid_tgt.count(j) > 0) {
            out_named.push_back(
                std::make_tuple(col_src_names.at(i), col_tgt_names.at(j), d));
        }
    }

    return out_named;
}

inline std::tuple<std::unordered_set<Index>, Index>
find_nz_cols(const std::string mtx_file)
{
    col_stat_collector_t collector;
    visit_matrix_market_file(mtx_file, collector);
    std::unordered_set<Index> valid;
    const IntVec &nn = collector.Col_N;
    for (Index j = 0; j < nn.size(); ++j) {
        if (nn(j) > 0.0)
            valid.insert(j);
    }
    const Index N = collector.max_col;
    return std::make_tuple(valid, N); // RVO
}

inline std::tuple<std::unordered_set<Index>, // valid
                  Index,                     // #total
                  std::vector<std::string>   // names
                  >
find_nz_col_names(const std::string mtx_file, const std::string col_file)
{
    using valid_set_t = std::unordered_set<Index>;
    valid_set_t valid;
    Index N;

    std::tie(valid, N) = find_nz_cols(mtx_file);

    std::vector<std::string> col_names;

    if (file_exists(col_file)) {
        CHECK(read_vector_file(col_file, col_names));
        ASSERT(col_names.size() >= N,
               "Not enough # names in `find_nz_col_names`");
    } else {
        for (Index j = 0; j < N; ++j) {
            col_names.push_back(std::to_string(j + 1));
        }
    }

    return std::make_tuple(valid, N, col_names);
}

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

// struct match_options_t {
//     using Str = std::string;
//     typedef enum { UNIFORM, CV, MEAN } sampling_method_t;
//     const std::vector<Str> METHOD_NAMES;
//     match_options_t()
//     {
//         knn = 1;
//         bilink = 10; // 2 ~ 100 (bi-directional link per element)
//         nlist = 10;  // knn ~ N (nearest neighbour)

//         src_mtx = "";
//         src_col = "";
//         tgt_mtx = "";
//         tgt_col = "";
//         out = "output.txt.gz";

//         tau = 1.0;
//         rank = 50;
//         lu_iter = 5;
//         col_norm = 10000;

//         prune_knn = false;
//         check_index = false;
//         raw_scale = false;
//         log_scale = true;
//         row_weight_file = "";

//         initial_sample = 10000;
//         block_size = 10000;

//         sampling_method = UNIFORM;

//         verbose = false;
//     }

//     Str src_mtx;
//     Str src_col;
//     Str tgt_mtx;
//     Str tgt_col;

//     Index knn;
//     Index bilink;
//     Index nlist;
//     Str out;

//     float tau;
//     Index rank;
//     Index lu_iter;
//     float col_norm;

//     bool check_index;
//     bool prune_knn;
//     bool raw_scale;
//     bool log_scale;
//     std::string row_weight_file;

//     Index initial_sample;
//     Index block_size;

//     sampling_method_t sampling_method;

//     bool verbose;

//     void set_sampling_method(const std::string _method)
//     {
//         for (int j = 0; j < METHOD_NAMES.size(); ++j) {
//             if (METHOD_NAMES.at(j) == _method) {
//                 sampling_method = static_cast<sampling_method_t>(j);
//                 break;
//             }
//         }
//     }
// };

// int
// parse_match_options(const int argc,     //
//                     const char *argv[], //
//                     match_options_t &options)
// {
//     const char *_usage =
//         "\n"
//         "[Arguments]\n"
//         "--src_mtx (-s)        : Source MTX file\n"
//         "--src_col (-c)        : Source column file\n"
//         "--tgt_mtx (-t)        : Target MTX file\n"
//         "--tgt_col (-g)        : Target column file\n"
//         "--tgt_dict (-d)       : Target dictionary file\n"
//         "--knn (-k)            : K nearest neighbors (default: 1)\n"
//         "--bilink (-m)         : # of bidirectional links (default: 10)\n"
//         "--nlist (-f)          : # nearest neighbor lists (default: 10)\n"
//         "--col_norm (-C)       : Column normalization (default: 10000)\n"
//         "--row_weight (-w)     : Feature re-weighting (default: none)\n"
//         "--log_scale (-L)      : Data in a log-scale (default: true)\n"
//         "--raw_scale (-R)      : Data in a raw-scale (default: false)\n"
//         "--prune_knn (-P)      : prune kNN graph (reciprocal match)\n"
//         "--out (-o)            : Output file name\n"
//         "\n"
//         "--check_index          : check matrix market index (default:
//         false)\n"
//         "\n"
//         "[Arguments for spectral matching]\n"
//         "--tau (-u)            : Regularization parameter (default: tau =
//         1)\n"
//         "--rank (-r)           : The maximal rank of SVD (default: rank =
//         50)\n"
//         "--lu_iter (-i)           : # of LU iterations (default: lu_iter =
//         5)\n"
//         "--initial_sample (-S) : Nystrom sample size (default: 10000)\n"
//         "--block_size (-B)  : Nystrom batch size (default: 10000)\n"
//         "--sampling_method (-M) : Nystrom sampling method: UNIFORM (default),
//         " "CV, MEAN\n"
//         "\n"
//         "[Details for kNN graph]\n"
//         "\n"
//         "(bilink)\n"
//         "The number of bi-directional links created for every new element\n"
//         "during construction. Reasonable range for M is 2-100. A high M
//         value\n" "works better on datasets with high intrinsic dimensionality
//         and/or\n" "high recall, while a low M value works better for datasets
//         with low\n" "intrinsic dimensionality and/or low recalls.\n"
//         "\n"
//         "(nlist)\n"
//         "The size of the dynamic list for the nearest neighbors (used
//         during\n" "the search). A higher N value leads to more accurate but
//         slower\n" "search. This cannot be set lower than the number of
//         queried nearest\n" "neighbors k. The value ef of can be anything
//         between k and the size of\n" "the dataset.\n"
//         "\n"
//         "[Reference]\n"
//         "Malkov, Yu, and Yashunin. `Efficient and robust approximate
//         nearest\n" "neighbor search using Hierarchical Navigable Small World
//         graphs.`\n"
//         "\n"
//         "preprint:"
//         "https://arxiv.org/abs/1603.09320\n"
//         "\n"
//         "See also:\n"
//         "https://github.com/nmslib/hnswlib\n"
//         "\n";

//     const char *const short_opts = "s:c:t:g:k:m:f:o:u:r:l:w:C:S:B:PILRM:h";

//     const option long_opts[] =
//         { { "src_mtx", required_argument, nullptr, 's' },         //
//           { "src_col", required_argument, nullptr, 'c' },         //
//           { "tgt_mtx", required_argument, nullptr, 't' },         //
//           { "tgt_col", required_argument, nullptr, 'g' },         //
//           { "knn", required_argument, nullptr, 'k' },             //
//           { "bilink", required_argument, nullptr, 'm' },          //
//           { "nlist", required_argument, nullptr, 'f' },           //
//           { "out", required_argument, nullptr, 'o' },             //
//           { "tau", required_argument, nullptr, 'u' },             //
//           { "rank", required_argument, nullptr, 'r' },            //
//           { "lu_iter", required_argument, nullptr, 'l' },         //
//           { "row_weight", required_argument, nullptr, 'w' },      //
//           { "col_norm", required_argument, nullptr, 'C' },        //
//           { "prune_knn", no_argument, nullptr, 'P' },             //
//           { "check_index", no_argument, nullptr, 'I' },           //
//           { "log_scale", no_argument, nullptr, 'L' },             //
//           { "raw_scale", no_argument, nullptr, 'R' },             //
//           { "initial_sample", required_argument, nullptr, 'S' },  //
//           { "block_size", required_argument, nullptr, 'B' },      //
//           { "sampling_method", required_argument, nullptr, 'M' }, //
//           { "help", no_argument, nullptr, 'h' },                  //
//           { nullptr, no_argument, nullptr, 0 } };

//     while (true) {
//         const auto opt = getopt_long(argc,                      //
//                                      const_cast<char **>(argv), //
//                                      short_opts,                //
//                                      long_opts,                 //
//                                      nullptr);

//         if (-1 == opt)
//             break;

//         switch (opt) {
//         case 's':
//             options.src_mtx = std::string(optarg);
//             break;
//         case 't':
//             options.tgt_mtx = std::string(optarg);
//             break;
//         case 'c':
//             options.src_col = std::string(optarg);
//             break;
//         case 'g':
//             options.tgt_col = std::string(optarg);
//             break;
//         case 'o':
//             options.out = std::string(optarg);
//             break;
//         case 'k':
//             options.knn = std::stoi(optarg);
//             break;
//         case 'm':
//             options.bilink = std::stoi(optarg);
//             break;
//         case 'f':
//             options.nlist = std::stoi(optarg);
//             break;
//         case 'u':
//             options.tau = std::stof(optarg);
//             break;
//         case 'C':
//             options.col_norm = std::stof(optarg);
//             break;
//         case 'r':
//             options.rank = std::stoi(optarg);
//             break;
//         case 'l':
//             options.lu_iter = std::stoi(optarg);
//             break;
//         case 'P':
//             options.prune_knn = true;
//             break;
//         case 'w':
//             options.row_weight_file = std::string(optarg);
//             break;
//         case 'S':
//             options.initial_sample = std::stoi(optarg);
//             break;
//         case 'B':
//             options.block_size = std::stoi(optarg);
//             break;
//         case 'I':
//             options.check_index = true;
//             break;
//         case 'L':
//             options.log_scale = true;
//             options.raw_scale = false;
//             break;
//         case 'R':
//             options.log_scale = false;
//             options.raw_scale = true;
//             break;
//         case 'M':
//             options.set_sampling_method(std::string(optarg));
//             break;

//         case 'h': // -h or --help
//         case '?': // Unrecognized option
//             std::cerr << _usage << std::endl;
//             return EXIT_FAILURE;
//         default: //
//                  ;
//         }
//     }

//     return EXIT_SUCCESS;
// }

#endif
