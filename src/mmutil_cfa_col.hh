#include <getopt.h>
#include <unordered_map>

#include "mmutil.hh"
#include "mmutil_io.hh"
#include "mmutil_stat.hh"
#include "mmutil_index.hh"
#include "mmutil_bgzf_util.hh"
#include "mmutil_spectral.hh"
#include "mmutil_match.hh"
#include "progress.hh"
#include "mmutil_pois.hh"
#include "mmutil_glm.hh"

#ifndef MMUTIL_CFA_COL_HH_
#define MMUTIL_CFA_COL_HH_

struct cfa_options_t {

    cfa_options_t()
    {
        mtx_file = "";
        annot_prob_file = "";
        annot_file = "";
        ind_file = "";
        annot_name_file = "";
        trt_ind_file = "";
        out = "output";

        tau = 1.0;   // Laplacian regularization
        rank = 10;   // SVD rank
        lu_iter = 5; // randomized SVD
        knn = 10;    // k-nearest neighbours
        bilink = 5;  // 2 ~ 100 (bi-directional link per element)
        nlist = 5;   // knn ~ N (nearest neighbour)

        raw_scale = true;
        log_scale = false;

        col_norm = 1000;
        block_size = 5000;

        em_iter = 10;
        em_tol = 1e-4;

        gamma_a0 = 1.0;
        gamma_b0 = 1.0;

        glm_pseudo = 1.0;
        glm_iter = 100;
        glm_reg = 1e-4;
        glm_sd = 1e-2;
        glm_std = true;

        impute_knn = false;

        verbose = false;
        discretize = true;

        nboot = 0;
        nthreads = 8;

        do_internal = false;
        check_index = false;

        // svd_u_file = "";
        // svd_d_file = "";
        svd_v_file = "";
    }

    std::string mtx_file;
    std::string annot_prob_file;
    std::string annot_file;
    std::string col_file;
    std::string ind_file;
    std::string trt_ind_file;
    std::string annot_name_file;
    std::string out;

    // std::string svd_u_file;
    // std::string svd_d_file;
    std::string svd_v_file;

    // SVD and matching
    std::string row_weight_file;

    bool raw_scale;
    bool log_scale;
    bool check_index;

    Scalar tau;
    Index rank;
    Index lu_iter;
    Scalar col_norm;

    Index knn;
    Index bilink;
    Index nlist;

    // SVD
    Index block_size;
    Index em_iter;
    Scalar em_tol;

    // Poisson Gamma parameters
    Scalar gamma_a0;
    Scalar gamma_b0;

    // Poisson GLM parameters
    Scalar glm_pseudo;
    Index glm_iter;
    Index glm_reg;
    Scalar glm_sd;
    bool glm_std;

    bool verbose;

    bool discretize;

    Index nboot;
    Index nthreads;

    bool do_internal;
    bool impute_knn;
};

struct cfa_data_t {

    using vs_type = hnswlib::InnerProductSpace;

    explicit cfa_data_t(const cfa_options_t &options)
        : mtx_file(options.mtx_file)
        , idx_file(options.mtx_file + ".index")
        , annot_prob_file(options.annot_prob_file)
        , annot_file(options.annot_file)
        , col_file(options.col_file)
        , ind_file(options.ind_file)
        , trt_file(options.trt_ind_file)
        , annot_name_file(options.annot_name_file)
        , row_weight_file(options.row_weight_file)
        , output(options.out)
        , glm_feature(options.glm_pseudo)
        , glm_reg(options.glm_reg)
        , glm_sd(options.glm_sd)
        , glm_std(options.glm_std)
        , glm_iter(options.glm_iter)
        , knn(options.knn)
        , impute_knn(options.impute_knn)
        , do_check_index(options.check_index)
        , param_bilink(options.bilink)
        , param_nnlist(options.nlist)
        , n_threads(options.nthreads)
    {
        CHECK(init());

        rank = options.rank;

        if (file_exists(options.svd_v_file)) {
            TLOG("Reusing previous SVD results ...");
            read_data_file(options.svd_v_file, Vt); // Nsample x rank
            Vt.transposeInPlace();
            const Index k = Vt.rows();
            const Index n = Vt.cols();

            ASSERT(Nsample == n, "SVD V file has different sample size");

            if (k > rank) {
                WLOG("Using only the top " << rank << " factors of the V");
                Mat temp = Vt;
                Vt.resize(Nsample, rank);
                for (Index j = 0; j < rank; ++j)
                    Vt.col(j) = temp.col(j);
            }

            if (k < rank) {
                WLOG("Using k=" << k << " factors");
                rank = k;
            }

        } else {
            TLOG("Training SVD for spectral matching ...");
            CHECK(run_svd(options));
        }

        TLOG("Done SVD");

        if (param_bilink >= rank) {
            WLOG("Shrink M value: " << param_bilink << " vs. " << rank);
            param_bilink = rank - 1;
        }

        if (param_bilink < 2) {
            WLOG("too small M value");
            param_bilink = 2;
        }

        if (param_nnlist <= knn) {
            WLOG("too small N value");
            param_nnlist = knn + 1;
        }

        // build_dictionary_by_individual();
        build_dictionary_by_treatment();
        TLOG("Successfully loaded necessary data in the memory");
    }

public:
    struct glm_feature_op_t {
        explicit glm_feature_op_t(const Scalar pseudo)
            : glm_pseudo(pseudo)
        {
        }
        Scalar operator()(const Scalar &x) const
        {
            return fasterlog(x + glm_pseudo);
        }
        const Scalar glm_pseudo;
    };

private:
    int init();

    int run_svd(const cfa_options_t &options);

    void build_dictionary_by_individual();

    void build_dictionary_by_treatment();

public:
    Mat read_y_block(const std::vector<Index> &subcol);

    Mat read_z_block(const std::vector<Index> &subcol);

    Mat read_cf_block(const std::vector<Index> &subcol, bool is_internal);

public:
    const std::vector<Index> &take_cells_in(const Index ii)
    {
        return indv_index_set.at(ii);
    }

    std::string take_ind_name(const Index ii) { return indv_id_name.at(ii); }

    std::string take_annot_name(const Index k) { return annot_name.at(k); }

private:
    Index Nsample;
    Index Nind;
    Index Ntrt;
    Index K;
    Index D;
    Mat Z;
    Vec ww;
    Index rank;
    Mat Vt;

    std::vector<Index> mtx_idx_tab;

    std::vector<std::string> cols;
    std::vector<std::string> annot_name;

public:
    Index num_annot() const { return K; }
    Index num_ind() const { return Nind; }
    Index num_trt() const { return Ntrt; }
    Index num_feature() const { return D; }

public: // const names
    const std::string mtx_file;
    const std::string idx_file;
    const std::string annot_prob_file;
    const std::string annot_file;
    const std::string col_file;
    const std::string ind_file;
    const std::string trt_file;
    const std::string annot_name_file;
    const std::string row_weight_file;
    const std::string output;

    const glm_feature_op_t glm_feature;
    const Index glm_reg;
    const Scalar glm_sd;
    const bool glm_std;
    const Index glm_iter;

private:
    std::size_t knn;
    bool impute_knn;
    bool do_check_index;
    std::size_t param_bilink;
    std::size_t param_nnlist;
    Index n_threads;

private:
    std::vector<std::string> indv_membership; // [N] -> individual
    std::vector<std::string> indv_id_name;    //
    std::vector<Index> indv;                  // map: col -> indv index
    std::vector<std::vector<Index>> indv_index_set;

    std::vector<std::string> trt_membership; // [N] -> treatment
    std::vector<std::string> trt_id_name;    //
    std::vector<Index> trt_map;              // map: col -> trt index
    std::vector<std::vector<Index>> trt_index_set;

    std::vector<std::shared_ptr<hnswlib::InnerProductSpace>> vs_vec_trt;
    std::vector<std::shared_ptr<KnnAlg>> knn_lookup_trt;

    std::vector<std::shared_ptr<hnswlib::InnerProductSpace>> vs_vec_indv;
    std::vector<std::shared_ptr<KnnAlg>> knn_lookup_indv;
};

Mat
cfa_data_t::read_y_block(const std::vector<Index> &subcol)
{
    using namespace mmutil::io;
    return Mat(read_eigen_sparse_subset_col(mtx_file, mtx_idx_tab, subcol));
}

Mat
cfa_data_t::read_z_block(const std::vector<Index> &subcol)
{
    Mat zz = row_sub(Z, subcol); //
    zz.transposeInPlace();       // Z: K x N
    return zz;
};

Mat
cfa_data_t::read_cf_block(const std::vector<Index> &cells_j,
                          bool is_internal = false)
{
    float *mass = Vt.data();
    const Index n_j = cells_j.size();
    Mat y = read_y_block(cells_j);
    Mat y0(D, n_j);

    for (Index jth = 0; jth < n_j; ++jth) {     // For each cell j
        const Index _cell_j = cells_j.at(jth);  //
        const Index tj = trt_map.at(_cell_j);   // Trt group for this cell j
        const Index jj = indv.at(_cell_j);      // Individual for this cell j
        const std::size_t n_j = cells_j.size(); // number of cells

        std::vector<Index> counterfactual_neigh;

#ifdef CPYTHON
        if (PyErr_CheckSignals() != 0) {
            ELOG("Interrupted while working on kNN: j = " << jth);
            std::exit(1);
        }
#endif

        std::vector<Scalar> dist_neigh, weights_neigh;

        ///////////////////////////////////////////////
        // Search neighbours in the other conditions //
        ///////////////////////////////////////////////

        for (Index ti = 0; ti < Ntrt; ++ti) {

            if (!is_internal) { // counterfactual
                if (ti == tj)   // skip the same condition
                    continue;   //
            } else {            // internal... more like null
                if (ti != tj)   // skip the different condition
                    continue;   //
            }

            const std::vector<Index> &cells_i = trt_index_set.at(ti);
            KnnAlg &alg_ti = *knn_lookup_trt[ti].get();
            const std::size_t n_i = cells_i.size();
            const std::size_t nquery = std::min(knn, n_i);

            Index deg_j = 0; // # of neighbours

            auto pq = alg_ti.searchKnn((void *)(mass + rank * _cell_j), nquery);

            while (!pq.empty()) {
                float d = 0;                         // distance
                std::size_t k;                       // local index
                std::tie(d, k) = pq.top();           //
                const Index _cell_i = cells_i.at(k); // global index
                if (_cell_j != _cell_i) {
                    counterfactual_neigh.emplace_back(_cell_i);
                    dist_neigh.emplace_back(d);
                }
                pq.pop();
            }
        }

        ////////////////////////////////////////////////////////
        // Find optimal weights for counterfactual imputation //
        ////////////////////////////////////////////////////////

        if (counterfactual_neigh.size() > 1) {
            Mat yy = y.col(jth);

            if (impute_knn) {
                Index deg_ = counterfactual_neigh.size();
                weights_neigh.resize(deg_);
                normalize_weights(deg_, dist_neigh, weights_neigh);
                Vec w0_ = eigen_vector(weights_neigh);
                Mat y0_ = read_y_block(counterfactual_neigh);
                const Scalar denom = w0_.sum(); // must be > 0

                y0.col(jth) = y0_ * w0_ / denom;

            } else {

                Mat xx =
                    read_y_block(counterfactual_neigh).unaryExpr(glm_feature);
                const bool glm_intercept = true;
                y0.col(jth) = predict_poisson_glm(xx,
                                                  yy,
                                                  glm_iter,
                                                  glm_reg,
                                                  glm_intercept,
                                                  glm_std,
                                                  glm_sd);
            }

        } else if (counterfactual_neigh.size() == 1) {
            y0.col(jth) = read_y_block(counterfactual_neigh).col(0);
        }
    }
    return y0;
}

void
cfa_data_t::build_dictionary_by_treatment()
{

    for (Index tt = 0; tt < Ntrt; ++tt) {

        const Index n_tot = trt_index_set[tt].size();

        vs_vec_trt.emplace_back(std::make_shared<vs_type>(rank));

        vs_type &VS = *vs_vec_trt[tt].get();

        knn_lookup_trt.emplace_back(
            std::make_shared<KnnAlg>(&VS, n_tot, param_bilink, param_nnlist));
    }

    progress_bar_t<Index> prog(Nsample, 1e2);

    for (Index tt = 0; tt < Ntrt; ++tt) {
        const Index n_tot = trt_index_set[tt].size(); // # cells
        KnnAlg &alg = *knn_lookup_trt[tt].get();      // lookup
        float *mass = Vt.data();                      // raw data

#pragma omp parallel for num_threads(n_threads)
        for (Index i = 0; i < n_tot; ++i) {
            const Index cell_j = trt_index_set.at(tt).at(i);
            alg.addPoint((void *)(mass + rank * cell_j), i);
            // prog.update();
            // prog(std::cerr);
        }
        TLOG("Built a lookup for the treatment : " << tt);
    }
}

void
cfa_data_t::build_dictionary_by_individual()
{

    for (Index ii = 0; ii < Nind; ++ii) {
        const Index n_tot = indv_index_set[ii].size();
        vs_vec_indv.emplace_back(std::make_shared<vs_type>(rank));

        vs_type &VS = *vs_vec_indv[ii].get();
        knn_lookup_indv.emplace_back(
            std::make_shared<KnnAlg>(&VS, n_tot, param_bilink, param_nnlist));
    }

    progress_bar_t<Index> prog(Nsample, 1e2);

    for (Index ii = 0; ii < Nind; ++ii) {
        const Index n_tot = indv_index_set[ii].size(); // # cells
        KnnAlg &alg = *knn_lookup_indv[ii].get();      // lookup
        float *mass = Vt.data();                       // raw data

#pragma omp parallel for num_threads(n_threads)
        for (Index i = 0; i < n_tot; ++i) {
            const Index cell_j = indv_index_set.at(ii).at(i);
            alg.addPoint((void *)(mass + rank * cell_j), i);
            // prog.update();
            // prog(std::cerr);
        }
        TLOG("Built a lookup for the individual : " << ii);
    }
}

int
cfa_data_t::run_svd(const cfa_options_t &options)
{
    ////////////////////////////////
    // Learn latent embedding ... //
    ////////////////////////////////
    svd_out_t svd = take_svd_online_em(mtx_file, idx_file, ww, options);
    TLOG("Done online SVD");

    Mat proj = svd.U * svd.D.cwiseInverse().asDiagonal(); // feature x rank
    TLOG("Found projection: " << proj.rows() << " x " << proj.cols());

    rank = proj.cols();

    Vt.resize(rank, Nsample);

    const Index block_size = options.block_size;

    TLOG("Populating projected data...");

    for (Index lb = 0; lb < Nsample; lb += block_size) {
        const Index ub = std::min(Nsample, block_size + lb);

        std::vector<Index> sub_b(ub - lb);
        std::iota(sub_b.begin(), sub_b.end(), lb);

        Mat x0 = read_y_block(sub_b);

        Mat xx = make_normalized_laplacian(x0,
                                           ww,
                                           options.tau,
                                           options.col_norm,
                                           options.log_scale);

        Mat vv = proj.transpose() * xx; // rank x block_size
        normalize_columns(vv);          // to make cosine distance

        for (Index j = 0; j < vv.cols(); ++j) {
            const Index r = sub_b[j];
            Vt.col(r) = vv.col(j);
        }

        if (options.verbose)
            TLOG("Projected on batch [" << lb << ", " << ub << ")");
    }

    return EXIT_SUCCESS;
}

int
cfa_data_t::init()
{

    using namespace mmutil::io;
    using namespace mmutil::index;

    //////////////////
    // column names //
    //////////////////

    CHECK(read_vector_file(col_file, cols));
    Nsample = cols.size();

    //////////////////////////////
    // Indexing all the columns //
    //////////////////////////////

    if (!file_exists(idx_file)) // if needed
        CHECK(build_mmutil_index(mtx_file, idx_file));

    CHECK(read_mmutil_index(idx_file, mtx_idx_tab));

    if (do_check_index)
        CHECK(check_index_tab(mtx_file, mtx_idx_tab));

    mm_info_reader_t info;
    CHECK(mmutil::bgzf::peek_bgzf_header(mtx_file, info));
    D = info.max_row;

    ASSERT(Nsample == info.max_col,
           "Should have matched .mtx.gz, N = " << Nsample << " vs. "
                                               << info.max_col);

    /////////////////
    // label names //
    /////////////////

    CHECK(read_vector_file(annot_name_file, annot_name));
    auto lab_position = make_position_dict<std::string, Index>(annot_name);
    K = annot_name.size();

    ///////////////////////
    // latent annotation //
    ///////////////////////

    TLOG("Reading latent annotations");

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
        ELOG("Unable to read latent annotations");
        return EXIT_FAILURE;
    }

    ASSERT(cols.size() == Z.rows(),
           "column and annotation matrix should match");

    ASSERT(annot_name.size() == Z.cols(),
           "Need the same number of label names for the columns of Z");

    TLOG("Latent membership matrix: " << Z.rows() << " x " << Z.cols());

    ///////////////////////////////////
    // weights for the rows/features //
    ///////////////////////////////////

    Vec weights;
    if (file_exists(row_weight_file)) {
        std::vector<Scalar> _ww;
        CHECK(read_vector_file(row_weight_file, _ww));
        weights = eigen_vector(_ww);
    }

    ww.resize(D);
    ww.setOnes();

    if (weights.size() > 0) {
        ASSERT(weights.rows() == D, "Found invalid weight vector");
        ww = weights;
    }

    ///////////////////////////
    // individual membership //
    ///////////////////////////

    indv_membership.reserve(Z.rows());
    CHECK(read_vector_file(ind_file, indv_membership));

    ASSERT(indv_membership.size() == Nsample,
           "Check the individual membership file: "
               << indv_membership.size() << " vs. expected N = " << Nsample);

    std::tie(indv, indv_id_name, std::ignore) =
        make_indexed_vector<std::string, Index>(indv_membership);

    indv_index_set = make_index_vec_vec(indv);

    Nind = indv_id_name.size();

    TLOG("Identified " << Nind << " individuals");

    ASSERT(Z.rows() == indv.size(), "rows(Z) != rows(indv)");

    ////////////////////////////////////////////
    // case-control-like treatment membership //
    ////////////////////////////////////////////

    trt_membership.reserve(Nsample);

    CHECK(read_vector_file(trt_file, trt_membership));

    ASSERT(trt_membership.size() == Z.rows(),
           "size(Treatment) != row(Z) " << trt_membership.size() << " vs. "
                                        << Z.rows());

    std::tie(trt_map, trt_id_name, std::ignore) =
        make_indexed_vector<std::string, Index>(trt_membership);

    trt_index_set = make_index_vec_vec(trt_map);
    Ntrt = trt_index_set.size();
    TLOG("Identified " << Ntrt << " treatment conditions");

    ASSERT(Ntrt > 1, "Must have more than one treatment conditions");

    return EXIT_SUCCESS;
}

template <typename OPTIONS>
int
run_cfa_col(const OPTIONS &options)
{

    cfa_data_t data(options);

    const Scalar a0 = options.gamma_a0, b0 = options.gamma_b0;

    const Index K = data.num_annot();
    const Index Nind = data.num_ind();
    const Index D = data.num_feature();
    const Index Ntrt = data.num_trt();

    ASSERT_RET(Nind > 1, "Must have at least two individuals");
    ASSERT_RET(Ntrt > 1, "Must have at least two treatment conditions");

    std::vector<std::string> mu_col_names(K * Nind);
    std::fill(mu_col_names.begin(), mu_col_names.end(), "");

    Mat obs_sum(D, K * Nind);

    Mat obs_mu(D, K * Nind);
    Mat obs_mu_sd(D, K * Nind);
    Mat ln_obs_mu(D, K * Nind);
    Mat ln_obs_mu_sd(D, K * Nind);

    Mat cf_mu(D, K * Nind);
    Mat cf_mu_sd(D, K * Nind);
    Mat ln_cf_mu(D, K * Nind);
    Mat ln_cf_mu_sd(D, K * Nind);

    Mat resid_mu(D, K * Nind);
    Mat resid_mu_sd(D, K * Nind);
    Mat ln_resid_mu(D, K * Nind);
    Mat ln_resid_mu_sd(D, K * Nind);

    obs_sum.setZero();
    obs_mu.setZero();
    obs_mu_sd.setZero();
    ln_obs_mu.setZero();
    ln_obs_mu_sd.setZero();

    ln_cf_mu.setZero();
    ln_cf_mu_sd.setZero();
    cf_mu.setZero();
    cf_mu_sd.setZero();

    resid_mu.setZero();
    resid_mu_sd.setZero();
    ln_resid_mu.setZero();
    ln_resid_mu_sd.setZero();

    Mat cf_intern_mu(D, K * Nind);
    Mat cf_intern_mu_sd(D, K * Nind);
    Mat resid_intern_mu(D, K * Nind);
    Mat resid_intern_mu_sd(D, K * Nind);
    Mat ln_resid_intern_mu(D, K * Nind);
    Mat ln_resid_intern_mu_sd(D, K * Nind);

    cf_intern_mu.setZero();
    cf_intern_mu_sd.setZero();
    resid_intern_mu.setZero();
    resid_intern_mu_sd.setZero();
    ln_resid_intern_mu.setZero();
    ln_resid_intern_mu_sd.setZero();

    Mat boot_mean_resid_mu;
    Mat boot_sd_resid_mu;
    Mat boot_mean_ln_resid_mu;
    Mat boot_sd_ln_resid_mu;

    if (options.nboot > 0) {
        boot_mean_resid_mu.resize(D, K * Nind);
        boot_sd_resid_mu.resize(D, K * Nind);
        boot_mean_ln_resid_mu.resize(D, K * Nind);
        boot_sd_ln_resid_mu.resize(D, K * Nind);
    }

    Mat boot_mean_resid_intern_mu(D, K * Nind);
    Mat boot_sd_resid_intern_mu(D, K * Nind);
    Mat boot_mean_ln_resid_intern_mu(D, K * Nind);
    Mat boot_sd_ln_resid_intern_mu(D, K * Nind);

    std::random_device rd;
    std::mt19937 rng(rd());

    running_stat_t<Mat> _resid_mu_boot_i(D, K);
    running_stat_t<Mat> _ln_resid_mu_boot_i(D, K);

    running_stat_t<Mat> _resid_intern_mu_boot_i(D, K);
    running_stat_t<Mat> _ln_resid_intern_mu_boot_i(D, K);

    Index nind_proc = 0;

#pragma omp parallel for num_threads(options.nthreads)
    for (Index ii = 0; ii < Nind; ++ii) {

#ifdef CPYTHON
        if (PyErr_CheckSignals() != 0) {
            ELOG("Interrupted at Ind = " << (ii));
            return EXIT_FAILURE;
        }
#endif

        auto storage_index = [&K, &ii](const Index k) { return K * ii + k; };

        const std::vector<Index> &cells_i = data.take_cells_in(ii);

        TLOG("Creating imputed data by kNN matching [ind="
             << ii << ", #cells=" << cells_i.size() << "]");

        Mat y = data.read_y_block(cells_i);          // D x N
        Mat z = data.read_z_block(cells_i);          // K x N
        Mat y0 = data.read_cf_block(cells_i, false); // D x N

        TLOG("Estimating the model parameters       [ind="
             << ii << ", #cells=" << cells_i.size() << "]");

        {
            ///////////////////////////////////////////////////
            // control cells from different treatment groups //
            ///////////////////////////////////////////////////
            poisson_t pois(y, z, y0, z, a0, b0);
            pois.optimize();

            const Mat cf_mu_i = pois.mu_DK();
            const Mat cf_mu_sd_i = pois.mu_sd_DK();
            const Mat ln_cf_mu_i = pois.ln_mu_DK();
            const Mat ln_cf_mu_sd_i = pois.ln_mu_sd_DK();

            for (Index k = 0; k < K; ++k) {
                const Index s = storage_index(k);
                cf_mu.col(s) = cf_mu_i.col(k);
                cf_mu_sd.col(s) = cf_mu_sd_i.col(k);
                ln_cf_mu.col(s) = ln_cf_mu_i.col(k);
                ln_cf_mu_sd.col(s) = ln_cf_mu_sd_i.col(k);

                const std::string c =
                    data.take_ind_name(ii) + "_" + data.take_annot_name(k);

                mu_col_names[s] = c;
            }

            pois.residual_optimize();

            const Mat resid_mu_i = pois.residual_mu_DK();
            const Mat resid_mu_sd_i = pois.residual_mu_sd_DK();

            const Mat ln_resid_mu_i = pois.ln_residual_mu_DK();
            const Mat ln_resid_mu_sd_i = pois.ln_residual_mu_sd_DK();

            for (Index k = 0; k < K; ++k) {
                const Index s = storage_index(k);
                resid_mu.col(s) = resid_mu_i.col(k);
                resid_mu_sd.col(s) = resid_mu_sd_i.col(k);
                ln_resid_mu.col(s) = ln_resid_mu_i.col(k);
                ln_resid_mu_sd.col(s) = ln_resid_mu_sd_i.col(k);
            }
        }

        std::uniform_int_distribution<> rboot(0, cells_i.size() - 1);

        //////////////////////
        // bootstrapping... //
        //////////////////////

        if (options.nboot > 0) {
            _resid_mu_boot_i.reset();
            _ln_resid_mu_boot_i.reset();

            TLOG("Bootstrapping the model parameters    [ind="
                 << ii << ", #bootstrap=" << options.nboot << "]");
        }

        for (Index bb = 0; bb < options.nboot; ++bb) {

            Mat Yboot(y.rows(), y.cols());
            Mat Zboot(z.rows(), z.cols());
            Mat Y0boot(y0.rows(), y0.cols());

            for (Index j = 0; j < cells_i.size(); ++j) {
                const Index r = rboot(rng);
                Yboot.col(j) = y.col(r);
                Zboot.col(j) = z.col(r);
                Y0boot.col(j) = y0.col(r);
            }

            poisson_t pois(Yboot, Zboot, Y0boot, Zboot, a0, b0);
            pois.optimize();
            pois.residual_optimize();

            _resid_mu_boot_i(pois.residual_mu_DK());
            _ln_resid_mu_boot_i(pois.ln_residual_mu_DK());
        }

        if (options.nboot > 0) {

            Mat _mean = _resid_mu_boot_i.mean();
            Mat _sd = _resid_mu_boot_i.var().cwiseSqrt();

            Mat _ln_mean = _ln_resid_mu_boot_i.mean();
            Mat _ln_sd = _ln_resid_mu_boot_i.var().cwiseSqrt();

            for (Index k = 0; k < K; ++k) {
                const Index s = storage_index(k);
                boot_mean_resid_mu.col(s) = _mean.col(k);
                boot_mean_ln_resid_mu.col(s) = _ln_mean.col(k);
                boot_sd_resid_mu.col(s) = _sd.col(k);
                boot_sd_ln_resid_mu.col(s) = _ln_sd.col(k);
            }
        }

        if (options.do_internal) {

            Mat y0_intern = data.read_cf_block(cells_i, true); // D x N

            ////////////////////////////
            // internal control cells //
            ////////////////////////////
            poisson_t pois(y, z, y0_intern, z, a0, b0);
            pois.optimize();
            const Mat cf_intern_mu_i = pois.mu_DK();
            const Mat cf_intern_mu_sd_i = pois.mu_sd_DK();
            for (Index k = 0; k < K; ++k) {
                const Index s = storage_index(k);
                cf_intern_mu.col(s) = cf_intern_mu_i.col(k);
                cf_intern_mu_sd.col(s) = cf_intern_mu_sd_i.col(k);
            }
            pois.residual_optimize();
            const Mat resid_intern_mu_i = pois.residual_mu_DK();
            const Mat resid_intern_mu_sd_i = pois.residual_mu_sd_DK();
            const Mat ln_resid_intern_mu_i = pois.ln_residual_mu_DK();
            const Mat ln_resid_intern_mu_sd_i = pois.ln_residual_mu_sd_DK();

            for (Index k = 0; k < K; ++k) {
                const Index s = storage_index(k);
                resid_intern_mu.col(s) = resid_intern_mu_i.col(k);
                resid_intern_mu_sd.col(s) = resid_intern_mu_sd_i.col(k);
                ln_resid_intern_mu.col(s) = ln_resid_intern_mu_i.col(k);
                ln_resid_intern_mu_sd.col(s) = ln_resid_intern_mu_sd_i.col(k);
            }

            ///////////////////////////////////////////////////
            // bootstrapping... on the internally-controlled //
            ///////////////////////////////////////////////////

            if (options.nboot > 0) {
                _resid_intern_mu_boot_i.reset();
                _ln_resid_intern_mu_boot_i.reset();
                TLOG("Bootstrapping the model (internal)    [ind="
                     << ii << ", #bootstrap=" << options.nboot << "]");
            }

            for (Index bb = 0; bb < options.nboot; ++bb) {
                Mat Yboot(y.rows(), y.cols());
                Mat Zboot(z.rows(), z.cols());
                Mat Y0boot(y0_intern.rows(), y0_intern.cols());
                for (Index j = 0; j < cells_i.size(); ++j) {
                    const Index r = rboot(rng);
                    Yboot.col(j) = y.col(r);
                    Zboot.col(j) = z.col(r);
                    Y0boot.col(j) = y0_intern.col(r);
                }
                poisson_t pois(Yboot, Zboot, Y0boot, Zboot, a0, b0);
                pois.optimize();
                pois.residual_optimize();
                _resid_intern_mu_boot_i(pois.residual_mu_DK());
                _ln_resid_intern_mu_boot_i(pois.ln_residual_mu_DK());
            }

            if (options.nboot > 0) {
                Mat _mean = _resid_intern_mu_boot_i.mean();
                Mat _sd = _resid_intern_mu_boot_i.var().cwiseSqrt();
                Mat _ln_mean = _ln_resid_intern_mu_boot_i.mean();
                Mat _ln_sd = _ln_resid_intern_mu_boot_i.var().cwiseSqrt();
                for (Index k = 0; k < K; ++k) {
                    const Index s = storage_index(k);
                    boot_mean_resid_intern_mu.col(s) = _mean.col(k);
                    boot_mean_ln_resid_intern_mu.col(s) = _ln_mean.col(k);
                    boot_sd_resid_intern_mu.col(s) = _sd.col(k);
                    boot_sd_ln_resid_intern_mu.col(s) = _ln_sd.col(k);
                }
            }
        }

        {
            ////////////////////////////
            // vanilla quantification //
            ////////////////////////////
            poisson_t pois(y, z, a0, b0);
            pois.optimize();

            const Mat obs_mu_i = pois.mu_DK();
            const Mat obs_mu_sd_i = pois.mu_sd_DK();
            const Mat ln_obs_mu_i = pois.ln_mu_DK();
            const Mat ln_obs_mu_sd_i = pois.ln_mu_sd_DK();

            const Mat obs_sum_i = y * z.transpose();

            for (Index k = 0; k < K; ++k) {
                const Index s = storage_index(k);
                obs_mu.col(s) = obs_mu_i.col(k);
                ln_obs_mu.col(s) = ln_obs_mu_i.col(k);
                obs_mu_sd.col(s) = obs_mu_sd_i.col(k);
                ln_obs_mu_sd.col(s) = ln_obs_mu_sd_i.col(k);
                obs_sum.col(s) = obs_sum_i.col(k);
            }
        }

        TLOG("Number of individuals processed: " << (++nind_proc) << std::endl);
    }

    TLOG("Writing down the results ...");

    write_vector_file(options.out + ".mu_cols.gz", mu_col_names);

    write_data_file(options.out + ".ln_cf_mu.gz", ln_cf_mu);
    write_data_file(options.out + ".ln_cf_mu_sd.gz", ln_cf_mu_sd);
    write_data_file(options.out + ".cf_mu.gz", cf_mu);
    write_data_file(options.out + ".cf_mu_sd.gz", cf_mu_sd);

    write_data_file(options.out + ".ln_obs_mu.gz", ln_obs_mu);
    write_data_file(options.out + ".ln_obs_mu_sd.gz", ln_obs_mu_sd);
    write_data_file(options.out + ".obs_mu.gz", obs_mu);
    write_data_file(options.out + ".obs_mu_sd.gz", obs_mu_sd);
    write_data_file(options.out + ".obs_sum.gz", obs_sum);

    // residual effect

    write_data_file(options.out + ".resid_mu.gz", resid_mu);
    write_data_file(options.out + ".resid_mu_sd.gz", resid_mu_sd);
    write_data_file(options.out + ".ln_resid_mu.gz", ln_resid_mu);
    write_data_file(options.out + ".ln_resid_mu_sd.gz", ln_resid_mu_sd);

    // bootstrapped results
    if (options.nboot > 0) {
        write_data_file(options.out + ".boot_mu.gz", boot_mean_resid_mu);
        write_data_file(options.out + ".boot_ln_mu.gz", boot_mean_ln_resid_mu);
        write_data_file(options.out + ".boot_sd_mu.gz", boot_sd_resid_mu);
        write_data_file(options.out + ".boot_sd_ln_mu.gz", boot_sd_ln_resid_mu);
    }

    ////////////////////////////
    // internal control cells //
    ////////////////////////////

    if (options.do_internal) {

        write_data_file(options.out + ".cf_internal_mu.gz", cf_intern_mu);
        write_data_file(options.out + ".cf_internal_mu_sd.gz", cf_intern_mu_sd);
        write_data_file(options.out + ".resid_internal_mu.gz", resid_intern_mu);
        write_data_file(options.out + ".resid_internal_mu_sd.gz",
                        resid_intern_mu_sd);

        write_data_file(options.out + ".ln_resid_internal_mu.gz",
                        ln_resid_intern_mu);

        write_data_file(options.out + ".ln_resid_internal_mu_sd.gz",
                        ln_resid_intern_mu_sd);

        // bootstrapped results
        if (options.nboot > 0) {
            write_data_file(options.out + ".boot_internal_mu.gz",
                            boot_mean_resid_intern_mu);
            write_data_file(options.out + ".boot_ln_internal_mu.gz",
                            boot_mean_ln_resid_intern_mu);
            write_data_file(options.out + ".boot_sd_internal_mu.gz",
                            boot_sd_resid_intern_mu);
            write_data_file(options.out + ".boot_sd_ln_internal_mu.gz",
                            boot_sd_ln_resid_intern_mu);
        }
    }
    return EXIT_SUCCESS;
}

template <typename OPTIONS>
int
parse_cfa_options(const int argc,     //
                  const char *argv[], //
                  OPTIONS &options)
{
    const char *_usage =
        "\n"
        "[Arguments]\n"
        "--mtx (-m)                  : data MTX file (M x N)\n"
        "--data (-m)                 : data MTX file (M x N)\n"
        "--col (-c)                  : data column file (N x 1)\n"
        "--annot (-a)                : annotation/clustering assignment (N x 2)\n"
        "--annot_prob (-A)           : annotation/clustering probability (N x K)\n"
        "--ind (-i)                  : N x 1 sample to individual (n)\n"
        "--trt (-t)                  : N x 1 sample to case-control membership/probability\n"
        "--lab (-l)                  : K x 1 annotation label name (e.g., cell type) \n"
        "--out (-o)                  : Output file header\n"
        "\n"
        "[Options]\n"
        "\n"
        "--svd_v (-V)                : SVD V file (N x K)\n"
        "--col_norm (-C)             : Column normalization (default: 10000)\n"
        "\n"
        "--discretize (-D)           : Use discretized annotation matrix (default: true)\n"
        "--probabilistic (-P)        : Use expected annotation matrix (default: false)\n"
        "\n"
        "--gamma_a0                  : prior for gamma distribution(a0,b0) (default: 1)\n"
        "--gamma_b0                  : prior for gamma distribution(a0,b0) (default: 1)\n"
        "--impute_knn                : impute by kNN, useful for large values (default: false)\n"
        "--glm_pseudo                : pseudocount for GLM features (default: 1)\n"
        "--glm_reg                   : Regularization parameter for GLM fitting (default: 1e-2)\n"
        "--glm_iter                  : Maximum number of iterations for GLM fitting (default: 100)\n"
        "\n"
        "--do_internal               : Calibration of the baseline by internal matching (default: false)\n"
        "--check_index               : Check matrix market index (default: false)\n"
        "\n"
        "[Matching options]\n"
        "\n"
        "--knn (-k)                  : k nearest neighbours (default: 10)\n"
        "--bilink (-b)               : # of bidirectional links (default: 5)\n"
        "--nlist (-n)                : # nearest neighbor lists (default: 5)\n"
        "\n"
        "--rank (-r)                 : # of SVD factors (default: rank = 50)\n"
        "--lu_iter (-u)              : # of LU iterations (default: iter = 5)\n"
        "--row_weight (-w)           : Feature re-weighting (default: none)\n"
        "\n"
        "--log_scale (-L)            : Data in a log-scale (default: false)\n"
        "--raw_scale (-R)            : Data in a raw-scale (default: true)\n"
        "\n"
        "--nboot (-B)                : # of bootstrap (default: 0)\n"
        "--nthreads (-T)             : # of threads (default: 8)\n"
        "--em_iter (-E)              : # of EM iterations for SVD (default: 10)\n"
        "\n"
        "[Output]\n"
        "\n"
        "${out}.ln_obs_mu.gz         : (M x n) Log Mean of observed matrix\n"
        "${out}.obs_mu.gz            : (M x n) Mean of observed matrix\n"
        "${out}.ln_cf_mu.gz          : (M x n) Log Confounding factors matrix\n"
        "${out}.cf_mu.gz             : (M x n) Confounding factors matrix\n"
        "${out}.resid_mu.gz          : (M x n) Mean after adjusting confounders\n"
        "${out}.ln_resid_mu.gz       : (M x n) Log mean after adjusting confounders\n"
        "${out}.mu_col.gz            : (n x 1) column names\n"
        "\n"
        "[Details for kNN graph]\n"
        "\n"
        "(bilink)\n"
        "The number of bi-directional links created for every new element\n"
        "during construction. Reasonable range for M is 2-100. A high M value\n"
        "works better on datasets with high intrinsic dimensionality and/or\n"
        "high recall, while a low M value works better for datasets with low\n"
        "intrinsic dimensionality and/or low recalls.\n"
        "\n"
        "(nlist)\n"
        "The size of the dynamic list for the nearest neighbors (used during\n"
        "the search). A higher N value leads to more accurate but slower\n"
        "search. This cannot be set lower than the number of queried nearest\n"
        "neighbors k. The value ef of can be anything between k and the size of\n"
        "the dataset.\n"
        "\n"
        "[Reference]\n"
        "Malkov, Yu, and Yashunin. `Efficient and robust approximate nearest\n"
        "neighbor search using Hierarchical Navigable Small World graphs.`\n"
        "\n"
        "preprint:"
        "https://arxiv.org/abs/1603.09320\n"
        "\n"
        "See also:\n"
        "https://github.com/nmslib/hnswlib\n"
        "\n";

    const char *const short_opts =
        "m:V:c:a:A:i:l:t:o:HLRS:r:u:w:g:G:BDPC:k:B:T:E:b:n:hzvIN0:1:p:e:g:";

    const option long_opts[] = {
        { "mtx", required_argument, nullptr, 'm' },        //
        { "data", required_argument, nullptr, 'm' },       //
        { "svd_v", required_argument, nullptr, 'V' },      //
        { "svd_v_file", required_argument, nullptr, 'V' }, //
        { "annot_prob", required_argument, nullptr, 'A' }, //
        { "annot", required_argument, nullptr, 'a' },      //
        { "col", required_argument, nullptr, 'c' },        //
        { "ind", required_argument, nullptr, 'i' },        //
        { "trt", required_argument, nullptr, 't' },        //
        { "trt_ind", required_argument, nullptr, 't' },    //
        { "lab", required_argument, nullptr, 'l' },        //
        { "label", required_argument, nullptr, 'l' },      //
        { "out", required_argument, nullptr, 'o' },        //
        { "check_index", no_argument, nullptr, 'H' },      //
        { "log_scale", no_argument, nullptr, 'L' },        //
        { "raw_scale", no_argument, nullptr, 'R' },        //
        { "block_size", required_argument, nullptr, 'S' }, //
        { "rank", required_argument, nullptr, 'r' },       //
        { "lu_iter", required_argument, nullptr, 'u' },    //
        { "row_weight", required_argument, nullptr, 'w' }, //
        { "discretize", no_argument, nullptr, 'D' },       //
        { "probabilistic", no_argument, nullptr, 'P' },    //
        { "col_norm", required_argument, nullptr, 'C' },   //
        { "knn", required_argument, nullptr, 'k' },        //
        { "bilink", required_argument, nullptr, 'b' },     //
        { "nlist", required_argument, nullptr, 'n' },      //
        { "a0", required_argument, nullptr, '0' },         //
        { "b0", required_argument, nullptr, '1' },         //
        { "gamma_a0", required_argument, nullptr, '0' },   //
        { "gamma_b0", required_argument, nullptr, '1' },   //
        { "glm_pseudo", required_argument, nullptr, 'p' }, //
        { "glm_iter", required_argument, nullptr, 'e' },   //
        { "glm_reg", required_argument, nullptr, 'g' },    //
        { "verbose", no_argument, nullptr, 'v' },          //
        { "do_internal", no_argument, nullptr, 'I' },      //
        { "impute_knn", no_argument, nullptr, 'N' },       //
        { "nboot", required_argument, nullptr, 'B' },      //
        { "num_boot", required_argument, nullptr, 'B' },   //
        { "bootstrap", required_argument, nullptr, 'B' },  //
        { "nthreads", required_argument, nullptr, 'T' },   //
        { "em_iter", required_argument, nullptr, 'E' },    //
        { "emiter", required_argument, nullptr, 'E' },     //
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
        case 'V':
            options.svd_v_file = std::string(optarg);
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
        case 't':
            options.trt_ind_file = std::string(optarg);
            break;
        case 'l':
            options.annot_name_file = std::string(optarg);
            break;
        case 'o':
            options.out = std::string(optarg);
            break;

        case 'r':
            options.rank = std::stoi(optarg);
            break;

        case 'u':
            options.lu_iter = std::stoi(optarg);
            break;

        case 'w':
            options.row_weight_file = std::string(optarg);
            break;

        case 'k':
            options.knn = std::stoi(optarg);
            break;

        case 'H':
            options.check_index = true;
            break;

        case 'L':
            options.log_scale = true;
            options.raw_scale = false;
            break;

        case 'R':
            options.log_scale = false;
            options.raw_scale = true;
            break;

        case 'P':
            options.discretize = false;
            break;

        case 'C':
            options.col_norm = std::stof(optarg);
            break;

        case 'D':
            options.discretize = true;
            break;

        case 'S':
            options.block_size = std::stoi(optarg);
            break;

        case 'B':
            options.nboot = std::stoi(optarg);
            break;

        case 'T':
            options.nthreads = std::stoi(optarg);
            break;

        case 'E':
            options.em_iter = std::stoi(optarg);
            break;

        case 'b':
            options.bilink = std::stoi(optarg);
            break;

        case 'n':
            options.nlist = std::stoi(optarg);
            break;

        case 'p':
            options.glm_pseudo = std::stof(optarg);
            break;

        case 'e':
            options.glm_iter = std::stoi(optarg);
            break;

        case 'g':
            options.glm_reg = std::stof(optarg);
            break;

        case '0':
            options.gamma_a0 = std::stof(optarg);
            break;

        case '1':
            options.gamma_b0 = std::stof(optarg);
            break;

        case 'v': // -v or --verbose
            options.verbose = true;
            break;

        case 'I':
            options.do_internal = true;
            break;

        case 'N':
            options.impute_knn = true;
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
    ERR_RET(!file_exists(options.col_file), "No COL file");
    ERR_RET(!file_exists(options.annot_name_file), "No LAB file");

    ERR_RET(options.rank < 1, "Too small rank");

    return EXIT_SUCCESS;
}

///////////////////////////////////////////////
// Estimate sequencing depth given mu matrix //
///////////////////////////////////////////////

struct cfa_depth_finder_t {

    explicit cfa_depth_finder_t(const Mat &_mu,
                                const Mat &_zz,
                                const std::vector<Index> &_indv,
                                const Scalar a0,
                                const Scalar b0)
        : Mu(_mu)
        , Z(_zz)
        , indv(_indv)
        , D(Mu.rows())
        , K(Z.cols())
        , Nsample(Z.rows())
        , opt_op(a0, b0)
    {
        max_row = 0;
        max_col = 0;
        max_elem = 0;
    }

    void set_file(BGZF *_fp) { fp = _fp; }

    void eval_after_header(const Index r, const Index c, const Index e)
    {
        std::tie(max_row, max_col, max_elem) = std::make_tuple(r, c, e);
        num_vec.resize(max_col, 1);
        denom_vec.resize(max_col, 1);
        num_vec.setZero();
        denom_vec.setZero();
    }

    void eval(const Index row, const Index col, const Scalar weight)
    {

        if (row < max_row && col < max_col) {

            const Index ii = indv.at(col);
            Scalar num = 0., denom = 0.;

            num += weight;

            for (Index k = 0; k < K; ++k) { // [ii * K, (ii+1)*K)
                const Index j = ii * K + k;
                const Scalar z_k = Z(col, k);
                denom += Mu(row, j) * z_k;
            }

            num_vec(col) += num;
            denom_vec(col) += denom;
        }
#ifdef DEBUG
        else {
            TLOG("[" << row << ", " << col << ", " << weight << "]");
            TLOG(max_row << " x " << max_col);
        }
#endif
    }

    void eval_end_of_file() { }

    Vec estimate_depth() { return num_vec.binaryExpr(denom_vec, opt_op); }

    const Mat &Mu;                  // D x (K * Nind)
    const Mat &Z;                   // Nsample x K
    const std::vector<Index> &indv; // Nsample x 1
    const Index D;
    const Index K;
    const Index Nsample;

private:
    BGZF *fp;

    Index max_row;
    Index max_col;
    Index max_elem;

    Vec num_vec;
    Vec denom_vec;

private:
    poisson_t::rate_op_t opt_op;
};

////////////////////////////////
// Adjust confounding factors //
////////////////////////////////

struct cfa_normalizer_t {

    explicit cfa_normalizer_t(const Mat &_mu,
                              const Mat &_zz,
                              const Vec &_rho,
                              const std::vector<Index> &_indv,
                              const std::string _outfile)
        : Mu(_mu)
        , Z(_zz)
        , rho(_rho)
        , indv(_indv)
        , D(Mu.rows())
        , K(Z.cols())
        , Nsample(Z.rows())
        , outfile(_outfile)
    {
        ASSERT(Z.rows() == indv.size(),
               "Needs the annotation and membership for each column");
    }

    void set_file(BGZF *_fp) { fp = _fp; }

    void eval_after_header(const Index r, const Index c, const Index e)
    {
        std::tie(max_row, max_col, max_elem) = std::make_tuple(r, c, e);
        // confirm that the sizes are compatible
        ASSERT(D == r, "dimensionality should match");
        ASSERT(Nsample == c, "sample size should match");
        ofs.open(outfile.c_str(), std::ios::out);
        ofs << "%%MatrixMarket matrix coordinate integer general" << std::endl;
        ofs << max_row << FS << max_col << FS << max_elem << std::endl;
        elem_check = 0;
    }

    void eval(const Index row, const Index col, const Scalar weight)
    {
        const Index ii = indv.at(col);

        Scalar denom = 0.;

        for (Index k = 0; k < K; ++k) { // [ii * K, (ii+1)*K)
            const Index j = ii * K + k;
            const Scalar z_k = Z(col, k);
            denom += Mu(row, j) * z_k;
        }

        // weight <- weight / denom;
        if (row < max_row && col < max_col) {
            const Index i = row + 1; // fix zero-based to one-based
            const Index j = col + 1; // fix zero-based to one-based

            if (denom > 0. && rho(col) > 0.) {
                const Scalar new_weight = weight / denom / rho(col);
                ofs << i << FS << j << FS << new_weight << std::endl;
            } else {
                ofs << i << FS << j << FS << weight << std::endl;
            }
            elem_check++;
        }
    }

    void eval_end_of_file()
    {
        ofs.close();
        ASSERT(max_elem == elem_check, "Failed to write all the elements");
    }

    const Mat &Mu;                  // D x (K * Nind)
    const Mat &Z;                   // Nsample x K
    const Vec &rho;                 // Nsample x 1
    const std::vector<Index> &indv; // Nsample x 1
    const Index D;
    const Index K;
    const Index Nsample;

    const std::string outfile;

private:
    obgzf_stream ofs;
    BGZF *fp;
    Index max_row;
    Index max_col;
    Index max_elem;
    Index elem_check;
    static constexpr char FS = ' ';
};

#endif
