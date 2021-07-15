// #include <getopt.h>

#include "mmutil.hh"
#include "mmutil_io.hh"
#include "mmutil_stat.hh"
#include "mmutil_score.hh"

#ifndef MMUTIL_FILTER_ROW_HH_
#define MMUTIL_FILTER_ROW_HH_

template <typename OPTIONS>
int filter_row_by_score(OPTIONS &options);

struct filter_row_options_t {

    typedef enum { NNZ, MEAN, CV, SD } row_score_t;
    const std::vector<std::string> SCORE_NAMES;

    explicit filter_row_options_t()
        : SCORE_NAMES { "NNZ", "MEAN", "CV", "SD" }
    {
        mtx_file = "";
        row_file = "";
        col_file = "";
        output = "output";
        Ntop = 0;
        cutoff = 0;
        COL_NNZ_CUTOFF = 0;

        row_score = NNZ;
    }

    Index Ntop;
    Scalar cutoff;
    Index COL_NNZ_CUTOFF;

    row_score_t row_score;

    std::string mtx_file;
    std::string row_file;
    std::string col_file;
    std::string output;

    void set_row_score(const std::string _score)
    {
        for (int j = 0; j < SCORE_NAMES.size(); ++j) {
            if (SCORE_NAMES.at(j) == _score) {
                row_score = static_cast<row_score_t>(j);
                break;
            }
        }
    }
};

struct select_cv_t {
    inline Vec operator()(Vec &mu, Vec &sig, Vec &cv, Vec &nnz) { return cv; }
};

struct select_mean_t {
    inline Vec operator()(Vec &mu, Vec &sig, Vec &cv, Vec &nnz) { return mu; }
};

struct select_sd_t {
    inline Vec operator()(Vec &mu, Vec &sig, Vec &cv, Vec &nnz) { return sig; }
};

struct select_nnz_t {
    inline Vec operator()(Vec &mu, Vec &sig, Vec &cv, Vec &nnz) { return nnz; }
};

/**
 * @param select_row_score
 * @param options
 * @return temp_mtx_file
 */
template <typename FUN, typename OPTIONS>
std::string
_filter_row_by_score(FUN select_row_score, OPTIONS &options)
{
    using namespace mmutil::io;
    std::vector<std::string> rows;
    CHECK(read_vector_file(options.row_file, rows));

    Vec mu, sig, cv;
    Index max_row, max_col;
    std::vector<Index> Nvec;

    std::tie(mu, sig, cv, Nvec, max_row, max_col) =
        compute_mtx_row_stat(options.mtx_file);

    Vec nvec = eigen_vector_<Index, Scalar>(Nvec);
    Vec row_scores = select_row_score(mu, sig, cv, nvec);

    using copier_t =
        triplet_copier_remapped_rows_t<obgzf_stream, Index, Scalar>;

    using index_map_t = copier_t::index_map_t;

    const Index nmax =
        (options.Ntop > 0) ? std::min(options.Ntop, max_row) : max_row;

    auto order = eigen_argsort_descending(row_scores);

    TLOG("row scores: " << row_scores(order.at(0)) << " ~ "
                        << row_scores(order.at(order.size() - 1)));

    std::vector<std::string> out_rows;
    out_rows.reserve(nmax);

    Index NNZ = 0;
    Index Nrow = 0;
    index_map_t remap;

    for (Index i = 0; i < nmax; ++i) {
        const Index j = order.at(i);
        const Scalar s = row_scores(j);

        if (s < options.cutoff)
            break;

        out_rows.emplace_back(rows.at(j));
        NNZ += Nvec.at(j);
        remap[j] = i;
        Nrow++;
    }

    TLOG("Filter in " << Nrow << " rows with " << NNZ << " elements");

    if (NNZ < 1)
        ELOG("Found no element to write out");

    std::string temp_mtx_file = options.output + "_temp.mtx.gz";
    std::string output_row_file = options.output + ".rows.gz";

    if (file_exists(temp_mtx_file)) {
        WLOG("Delete this pre-existing temporary file: " << temp_mtx_file);
        std::remove(temp_mtx_file.c_str());
    }

    copier_t copier(temp_mtx_file, remap, NNZ);
    visit_matrix_market_file(options.mtx_file, copier);
    write_vector_file(output_row_file, out_rows);

    return temp_mtx_file;
}

/**
 * @param mtx_temp_file
 * @param options
 */
template <typename OPTIONS>
int
squeeze_columns(const std::string mtx_temp_file, OPTIONS &options)
{
    using namespace mmutil::io;
    std::vector<std::string> cols;
    CHECK(read_vector_file(options.col_file, cols));

    using copier_t =
        triplet_copier_remapped_cols_t<obgzf_stream, Index, Scalar>;

    using index_map_t = copier_t::index_map_t;

    Index max_row, max_col;
    std::vector<Index> nnz_col;

    std::tie(std::ignore, std::ignore, std::ignore, nnz_col, max_row, max_col) =
        compute_mtx_col_stat(mtx_temp_file);

    index_map_t remap;

    Index Ncol = 0;
    Index NNZ = 0;
    std::vector<std::string> out_cols;

    for (Index i = 0; i < max_col; ++i) {
        const Scalar nnz = nnz_col.at(i);
        if (nnz > options.COL_NNZ_CUTOFF) {
            remap[i] = Ncol;
            out_cols.emplace_back(cols.at(i));
            ++Ncol;
            NNZ += nnz_col.at(i);
        }
    }

    TLOG("Found " << Ncol << " columns with the nnz > "
                  << options.COL_NNZ_CUTOFF << " from the total " << max_col
                  << " columns");

    ERR_RET(NNZ < 1, "Found no element to write out");

    std::string output_mtx_file = options.output + ".mtx.gz";

    copier_t copier(output_mtx_file, remap, NNZ);
    visit_matrix_market_file(mtx_temp_file, copier);

    if (file_exists(mtx_temp_file)) {
        std::remove(mtx_temp_file.c_str());
    }

    std::string output_col_file = options.output + ".cols.gz";

    write_vector_file(output_col_file, out_cols);

    return EXIT_SUCCESS;
}

template <typename OPTIONS>
int
filter_row_by_score(OPTIONS &options)
{
    std::string mtx_temp_file;

    switch (options.row_score) {
    case filter_row_options_t::NNZ:
        mtx_temp_file = _filter_row_by_score(select_nnz_t {}, options);
        break;
    case filter_row_options_t::MEAN:
        mtx_temp_file = _filter_row_by_score(select_mean_t {}, options);
        break;
    case filter_row_options_t::SD:
        mtx_temp_file = _filter_row_by_score(select_sd_t {}, options);
        break;
    case filter_row_options_t::CV:
        mtx_temp_file = _filter_row_by_score(select_cv_t {}, options);
        break;
    default:
        break;
    }

    return squeeze_columns(mtx_temp_file, options);
}

// template <typename OPTIONS>
// int
// parse_filter_row_options(const int argc,     //
//                          const char *argv[], //
//                          OPTIONS &options)
// {

//     const char *_usage =
//         "\n"
//         "[Arguments]\n"
//         "--mtx (-m)        : data MTX file (M x N)\n"
//         "--data (-m)       : data MTX file (M x N)\n"
//         "--row (-f)        : row file (M x 1)\n"
//         "--feature (-f)    : row file (M x 1)\n"
//         "--col (-c)        : data column file (N x 1)\n"
//         "--out (-o)        : Output file header\n"
//         "\n"
//         "[Options]\n"
//         "\n"
//         "--score (-S)      : a type of row scores {NNZ, MEAN, CV}\n"
//         "--ntop (-t)       : number of top features\n"
//         "--cutoff (-k)     : cutoff of row-wise scores\n"
//         "--col_cutoff (-C) : column's #non-zero cutoff\n"
//         "\n"
//         "[Output]\n"
//         "${out}.mtx.gz, ${out}.rows.gz, ${out}.cols.gz\n"
//         "\n";

//     const char *const short_opts = "m:c:f:o:t:k:C:S:";

//     const option long_opts[] = {
//         { "mtx", required_argument, nullptr, 'm' },            //
//         { "data", required_argument, nullptr, 'm' },           //
//         { "feature", required_argument, nullptr, 'f' },        //
//         { "row", required_argument, nullptr, 'f' },            //
//         { "col", required_argument, nullptr, 'c' },            //
//         { "out", required_argument, nullptr, 'o' },            //
//         { "output", required_argument, nullptr, 'o' },         //
//         { "ntop", required_argument, nullptr, 't' },           //
//         { "Ntop", required_argument, nullptr, 't' },           //
//         { "nTop", required_argument, nullptr, 't' },           //
//         { "cutoff", required_argument, nullptr, 'k' },         //
//         { "Cutoff", required_argument, nullptr, 'k' },         //
//         { "col_cutoff", required_argument, nullptr, 'C' },     //
//         { "col_nnz_cutoff", required_argument, nullptr, 'C' }, //
//         { "score", required_argument, nullptr, 'S' },          //
//         { nullptr, no_argument, nullptr, 0 }
//     };

//     while (true) {
//         const auto opt = getopt_long(argc,                      //
//                                      const_cast<char **>(argv), //
//                                      short_opts,                //
//                                      long_opts,                 //
//                                      nullptr);

//         if (-1 == opt)
//             break;

//         switch (opt) {
//         case 'm':
//             options.mtx_file = std::string(optarg);
//             break;

//         case 'f':
//             options.row_file = std::string(optarg);
//             break;

//         case 'c':
//             options.col_file = std::string(optarg);
//             break;

//         case 'o':
//             options.output = std::string(optarg);
//             break;

//         case 'S':
//             options.set_row_score(std::string(optarg));
//             break;

//         case 't':
//             options.Ntop = std::stoi(optarg);
//             break;

//         case 'k':
//             options.cutoff = std::stof(optarg);
//             break;

//         case 'C':
//             options.COL_NNZ_CUTOFF = std::stoi(optarg);
//             break;

//         case 'h': // -h or --help
//         case '?': // Unrecognized option
//             std::cerr << _usage << std::endl;
//             return EXIT_FAILURE;
//         default: //
//                  ;
//         }
//     }

//     ERR_RET(!file_exists(options.mtx_file), "No MTX data file");
//     ERR_RET(!file_exists(options.col_file), "No COL data file");
//     ERR_RET(!file_exists(options.row_file), "No ROW data file");

//     ERR_RET(options.Ntop <= 0 && options.cutoff <= 0,
//             "Must have positive Ntop or cutoff value");

//     return EXIT_SUCCESS;
// }

#endif
