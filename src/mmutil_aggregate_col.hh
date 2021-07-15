#include <getopt.h>
#include <unordered_map>

#include "mmutil.hh"
#include "mmutil_io.hh"
#include "mmutil_stat.hh"
#include "mmutil_index.hh"
#include "mmutil_bgzf_util.hh"
#include "mmutil_spectral.hh"
#include "mmutil_match.hh"

#include "io.hh"

#include "progress.hh"
#include "mmutil_pois.hh"

#ifndef MMUTIL_AGGREGATE_COL_HH_
#define MMUTIL_AGGREGATE_COL_HH_

struct aggregate_options_t {

    aggregate_options_t()
    {
        mtx_file = "";
        annot_prob_file = "";
        annot_file = "";
        ind_file = "";
        lab_file = "";

        col_norm = 1000;
        out = "output";
        verbose = false;

        discretize = true;
        normalize = false;
    }

    std::string mtx_file;
    std::string annot_prob_file;
    std::string annot_file;
    std::string col_file;
    std::string ind_file;
    std::string lab_file;
    std::string out;

    bool verbose;
    bool discretize;
    bool normalize;
    Scalar col_norm;
};

template <typename OPTIONS>
int
parse_aggregate_options(const int argc,     //
                        const char *argv[], //
                        OPTIONS &options)
{
    const char *_usage =
        "\n"
        "[Arguments]\n"
        "--mtx (-m)           : data MTX file (M x N)\n"
        "--data (-m)          : data MTX file (M x N)\n"
        "--col (-c)           : data column file (N x 1)\n"
        "--annot (-a)         : annotation/clustering assignment (N x 2)\n"
        "--annot_prob (-A)    : annotation/clustering probability (N x K)\n"
        "--ind (-i)           : N x 1 sample to individual (n)\n"
        "--lab (-l)           : K x 1 annotation label name (e.g., cell type) \n"
        "--out (-o)           : Output file header\n"
        "\n"
        "[Options]\n"
        "--col_norm (-C)      : Column normalization (default: 10000)\n"
        "--normalize (-z)     : Normalize columns (default: false) \n"
        "--discretize (-D)    : Use discretized annotation matrix (default: true)\n"
        "--probabilistic (-P) : Use expected annotation matrix (default: false)\n"
        "\n"
        "[Output]\n"
        "\n"
        "${out}.mean.gz       : (M x n) Mean matrix\n"
        "${out}.sum.gz        : (M x n) Summation matrix\n"
        "${out}.mu.gz         : (M x n) Depth-adjusted Mean matrix\n"
        "${out}.mu_sd.gz      : (M x n) SD of the depth-adjusted mean\n"
        "${out}.mu_cols.gz    : (n x 1) Column names\n"
        "\n";

    const char *const short_opts = "m:c:a:A:i:l:o:C:DPhv";

    const option long_opts[] = {
        { "mtx", required_argument, nullptr, 'm' },        //
        { "data", required_argument, nullptr, 'm' },       //
        { "annot_prob", required_argument, nullptr, 'A' }, //
        { "annot", required_argument, nullptr, 'a' },      //
        { "col", required_argument, nullptr, 'c' },        //
        { "ind", required_argument, nullptr, 'i' },        //
        { "lab", required_argument, nullptr, 'l' },        //
        { "label", required_argument, nullptr, 'l' },      //
        { "out", required_argument, nullptr, 'o' },        //
        { "discretize", no_argument, nullptr, 'D' },       //
        { "probabilistic", no_argument, nullptr, 'P' },    //
        { "col_norm", required_argument, nullptr, 'C' },   //
        { "normalize", no_argument, nullptr, 'z' },        //
        { "verbose", no_argument, nullptr, 'v' },          //
        { nullptr, no_argument, nullptr, 0 }
    };

    while (true) {
        const auto opt = getopt_long(argc,                      //
                                     const_cast<char **>(argv), //
                                     short_opts,                //
                                     long_opts,                 //
                                     nullptr);

        if (-1 == opt)
            break;

        switch (opt) {
        case 'm':
            options.mtx_file = std::string(optarg);
            break;
        case 'A':
            options.annot_prob_file = std::string(optarg);
            break;
        case 'a':
            options.annot_file = std::string(optarg);
            break;
        case 'c':
            options.col_file = std::string(optarg);
            break;
        case 'i':
            options.ind_file = std::string(optarg);
            break;
        case 'l':
            options.lab_file = std::string(optarg);
            break;
        case 'o':
            options.out = std::string(optarg);
            break;

        case 'P':
            options.discretize = false;
            break;

        case 'D':
            options.discretize = true;
            break;

        case 'C':
            options.col_norm = std::stof(optarg);
            break;

        case 'z':
            options.normalize = true;
            break;

        case 'v': // -v or --verbose
            options.verbose = true;
            break;
        case 'h': // -h or --help
        case '?': // Unrecognized option
            std::cerr << _usage << std::endl;
            return EXIT_FAILURE;
        default: //
                 ;
        }
    }

    ERR_RET(!file_exists(options.mtx_file), "No MTX file");
    ERR_RET(!file_exists(options.annot_prob_file) &&
                !file_exists(options.annot_file),
            "No ANNOT or ANNOT_PROB file");
    ERR_RET(!file_exists(options.ind_file), "No IND file");
    ERR_RET(!file_exists(options.lab_file), "No LAB file");

    return EXIT_SUCCESS;
}

template <typename OPTIONS>
int
aggregate_col(const OPTIONS &options)
{

    using namespace mmutil::io;
    using namespace mmutil::index;

    const std::string mtx_file = options.mtx_file;
    const std::string idx_file = options.mtx_file + ".index";
    const std::string annot_prob_file = options.annot_prob_file;
    const std::string annot_file = options.annot_file;
    const std::string col_file = options.col_file;
    const std::string ind_file = options.ind_file;
    const std::string lab_file = options.lab_file;
    const std::string output = options.out;

    //////////////////
    // column names //
    //////////////////

    std::vector<std::string> cols;
    CHECK(read_vector_file(col_file, cols));
    const Index Nsample = cols.size();

    /////////////////
    // label names //
    /////////////////

    std::vector<std::string> lab_name;
    CHECK(read_vector_file(lab_file, lab_name));
    auto lab_position = make_position_dict<std::string, Index>(lab_name);
    const Index K = lab_name.size();

    for (auto j : lab_name) {
        TLOG(j << " " << lab_position[j]);
    }

    ///////////////////////
    // latent annotation //
    ///////////////////////

    TLOG("Reading latent annotations");

    Mat Z;

    if (annot_file.size() > 0) {
        Z.resize(Nsample, K);
        Z.setZero();

        std::unordered_map<std::string, std::string> annot_dict;
        CHECK(read_dict_file(annot_file, annot_dict));
        for (Index j = 0; j < cols.size(); ++j) {
            const std::string &s = cols.at(j);
            if (annot_dict.count(s) > 0) {
                const std::string &t = annot_dict.at(s);
                if (lab_position.count(t) > 0) {
                    const Index k = lab_position.at(t);
                    Z(j, k) = 1.;
                }
            }
        }
    } else if (annot_prob_file.size() > 0) {
        CHECK(read_data_file(annot_prob_file, Z));
    } else {
        return EXIT_FAILURE;
    }

    ASSERT(cols.size() == Z.rows(),
           "column and annotation matrix should match");

    ASSERT(lab_name.size() == Z.cols(),
           "Need the same number of label names for the columns of Z");

    TLOG("Latent membership matrix: " << Z.rows() << " x " << Z.cols());

    ///////////////////////////
    // individual membership //
    ///////////////////////////

    std::vector<std::string> indv_membership;
    indv_membership.reserve(Z.rows());
    CHECK(read_vector_file(ind_file, indv_membership));

    ASSERT(indv_membership.size() == Z.rows(),
           "Individual membership file mismatches with Z: "
               << indv_membership.size() << " vs. " << Z.rows());

    std::vector<std::string> indv_id_name;
    std::vector<Index> indv; // map: col -> indv index

    std::tie(indv, indv_id_name, std::ignore) =
        make_indexed_vector<std::string, Index>(indv_membership);

    auto indv_index_set = make_index_vec_vec(indv);

    const Index Nind = indv_id_name.size();

    TLOG("Identified " << Nind << " individuals");

    ASSERT(Z.rows() == indv.size(), "rows(Z) != rows(indv)");

    TLOG("" << std::endl << Z.transpose() * Mat::Ones(Nsample, 1));

    //////////////////////////////
    // Indexing all the columns //
    //////////////////////////////

    std::vector<Index> mtx_idx_tab;

    if (!file_exists(idx_file)) // if needed
        CHECK(build_mmutil_index(mtx_file, idx_file));

    CHECK(read_mmutil_index(idx_file, mtx_idx_tab));

    CHECK(check_index_tab(mtx_file, mtx_idx_tab));

    mm_info_reader_t info;
    CHECK(mmutil::bgzf::peek_bgzf_header(mtx_file, info));
    const Index D = info.max_row;

    ASSERT(Nsample == info.max_col, "Should have matched .mtx.gz");

    ///////////////////////////
    // For each individual i //
    ///////////////////////////

    Mat out_mu;
    Mat out_mu_sd;
    Mat out_ln_mu;
    Mat out_ln_mu_sd;
    Mat out_mean;
    Mat out_sum;

    Vec out_rho(Nsample);
    out_rho.setOnes();

    using namespace mmutil::index;

    std::vector<std::string> out_col(Nind * K);
    std::fill(out_col.begin(), out_col.end(), "");

    // auto _sqrt = [](const Scalar &x) -> Scalar {
    //     return (x > 0.) ? std::sqrt(x) : 0.;
    // };

    const Scalar a0 = 1e-4, b0 = 1e-4;

    Index s_obs = 0;         // cumulative for obs (be cautious)
    const Scalar eps = 1e-4; //

    for (Index i = 0; i < Nind; ++i) {

#ifdef CPYTHON
        if (PyErr_CheckSignals() != 0) {
            ELOG("Interrupted at Ind = " << (i));
            return EXIT_FAILURE;
        }
#endif

        const std::string indv_name = indv_id_name.at(i);

        TLOG("Reading [" << indv_name << "] " << indv_index_set.at(i).size()
                         << " cells for an individual " << (i + 1) << " / "
                         << Nind);

        // Y: features x columns
        const std::vector<Index> &cols_i = indv_index_set.at(i);

        Mat yy = read_eigen_sparse_subset_col(mtx_file, mtx_idx_tab, cols_i);

        if (options.normalize) {
            normalize_columns(yy);
            yy *= options.col_norm;
            TLOG("Normalized Y")
        }

        const Index D = yy.rows();
        const Index N = yy.cols();

        if (i == 0) {
            out_mu.resize(D, Nind * K);
            out_mu_sd.resize(D, Nind * K);
            out_ln_mu.resize(D, Nind * K);
            out_ln_mu_sd.resize(D, Nind * K);
            out_mean.resize(D, Nind * K);
            out_sum.resize(D, Nind * K);
            out_mu.setZero();
            out_mu_sd.setZero();
            out_mean.setZero();
            out_sum.setZero();
        }

        auto is_nz = [](const Scalar &y) -> Scalar {
            return std::abs(y) > 1e-8 ? 1. : 0.;
        };

        const Index y_nnz = yy.unaryExpr(is_nz).sum();

        Index r, c;
        TLOG(N << " columns [" << yy.minCoeff() << ", " << yy.maxCoeff(&r, &c)
               << "], argmax (" << r << ", " << c << ") NNZ= " << y_nnz
               << ", Tot= " << yy.sum());

        Mat zz_prob = row_sub(Z, indv_index_set.at(i)); //
        zz_prob.transposeInPlace();                     // Z: K x N

        Mat zz(zz_prob.rows(), zz_prob.cols()); // K x N

        if (options.discretize) {
            zz.setZero();
            for (Index j = 0; j < zz_prob.cols(); ++j) {
                Index k;
                zz_prob.col(j).maxCoeff(&k);
                zz(k, j) = 1.0;
            }
            TLOG("Using the discretized Z: " << zz.sum());
        } else {
            zz = zz_prob;
            TLOG("Using the probabilistic Z: " << zz.sum());
        }

        poisson_t pois(yy, zz, a0, b0);
        pois.optimize();
        Mat _mu = pois.mu_DK();
        Mat _mu_sd = pois.mu_sd_DK();
        Mat _ln_mu = pois.ln_mu_DK();
        Mat _ln_mu_sd = pois.ln_mu_sd_DK();
        Mat _rho = pois.rho_N();

        Mat _sum = yy * zz.transpose();            // D x K
        Vec _denom = zz * Mat::Ones(zz.cols(), 1); // K x 1

        for (Index k = 0; k < K; ++k) {
            out_col[s_obs] = indv_name + "_" + lab_name.at(k);

            const Scalar _denom_k = _denom(k);

            if (_denom_k > eps) {
                out_sum.col(s_obs) = _sum.col(k);
                out_mean.col(s_obs) = _sum.col(k) / _denom_k;
                out_mu.col(s_obs) = _mu.col(k);
                out_mu_sd.col(s_obs) = _mu_sd.col(k);
                out_ln_mu.col(s_obs) = _ln_mu.col(k);
                out_ln_mu_sd.col(s_obs) = _ln_mu_sd.col(k);
            }
            ++s_obs;
        }

        for (Index j = 0; j < cols_i.size(); ++j) {
            const Index jj = cols_i[j];
            out_rho(jj) = _rho(j, 0);
        }
    }

    TLOG("Writing down the estimated effects");

    write_vector_file(output + ".mu_cols.gz", out_col);
    write_vector_file(output + ".rho.gz", std_vector(out_rho));

    write_data_file(output + ".ln_mu.gz", out_ln_mu);
    write_data_file(output + ".ln_mu_sd.gz", out_ln_mu_sd);
    write_data_file(output + ".mu.gz", out_mu);
    write_data_file(output + ".mu_sd.gz", out_mu_sd);
    write_data_file(output + ".mean.gz", out_mean);
    write_data_file(output + ".sum.gz", out_sum);

    return EXIT_SUCCESS;
}

#endif
