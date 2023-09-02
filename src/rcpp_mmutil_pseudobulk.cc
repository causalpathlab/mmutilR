#include "mmutil.hh"
#include "mmutil_io.hh"
#include "mmutil_stat.hh"
#include "mmutil_index.hh"
#include "mmutil_bgzf_util.hh"
#include "mmutil_spectral.hh"
#include "mmutil_match.hh"
#include "mmutil_pois.hh"
#include "mmutil_glm.hh"

#include "mmutil_cocoa.hh"
#include "mmutil_cocoa_paired.hh"

// [[Rcpp::plugins(openmp)]]

#include "io.hh"
#include "std_util.hh"

//' Create pseudo-bulk data for pairwise comparisons
//'
//' @param mtx_file data file
//' @param row_file row file
//' @param col_file column file
//' @param r_indv membership for the cells (\code{r_cols})
//' @param r_V SVD factors
//' @param r_cols cell (col) names (if we want to take a subset)
//' @param r_annot label annotation for the (\code{r_cols})
//' @param r_lab_name label names (default: everything in \code{r_annot})
//' @param r_annot_mat label annotation matrix (cell x type) (default: NULL)
//' @param a0 hyperparameter for gamma(a0, b0) (default: 1)
//' @param b0 hyperparameter for gamma(a0, b0) (default: 1)
//' @param eps small number (default: 1e-8)
//' @param knn_cell k-NN matching between cells
//' @param knn_indv k-NN matching between individuals
//' @param KNN_BILINK num. of bidirectional links (default: 10)
//' @param KNN_NNLIST num. of nearest neighbor lists (default: 10)
//' @param NUM_THREADS number of threads for multi-core processing
//' @param IMPUTE_BY_KNN imputation by kNN alone (default: FALSE)
//'
//' @return a list of inference results
//'
// [[Rcpp::export]]
Rcpp::List
rcpp_mmutil_aggregate_pairwise(
    const std::string mtx_file,
    const std::string row_file,
    const std::string col_file,
    Rcpp::Nullable<Rcpp::StringVector> r_indv,
    Rcpp::Nullable<Rcpp::NumericMatrix> r_V,
    Rcpp::Nullable<Rcpp::StringVector> r_cols = R_NilValue,
    Rcpp::Nullable<Rcpp::StringVector> r_annot = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericMatrix> r_annot_mat = R_NilValue,
    Rcpp::Nullable<Rcpp::StringVector> r_lab_name = R_NilValue,
    const double a0 = 1.0,
    const double b0 = 1.0,
    const double eps = 1e-8,
    const std::size_t knn_cell = 10,
    const std::size_t knn_indv = 1,
    const std::size_t KNN_BILINK = 10,
    const std::size_t KNN_NNLIST = 10,
    const std::size_t NUM_THREADS = 1,
    const bool IMPUTE_BY_KNN = false)
{

    Eigen::initParallel();

    CHK_RETL(mmutil::bgzf::convert_bgzip(mtx_file));

    std::vector<std::string> mtx_cols;
    CHK_RETL(read_vector_file(col_file, mtx_cols));

    //////////////////////////////
    // check column annotations //
    //////////////////////////////

    std::vector<std::string> cols;
    std::vector<std::string> indv;
    std::vector<std::string> annot;
    std::vector<std::string> lab_name;

    if (r_cols.isNotNull()) {
        copy(Rcpp::StringVector(r_cols), cols);
    } else {
        cols.reserve(mtx_cols.size());
        std::copy(std::begin(mtx_cols),
                  std::end(mtx_cols),
                  std::back_inserter(cols));
    }

    copy(Rcpp::StringVector(r_indv), indv);

    ASSERT_RETL(cols.size() == indv.size(), "|cols| != |indv|");

    ////////////////////////
    // universal position //
    ////////////////////////

    const Index Nsample = mtx_cols.size();
    auto mtx_pos = make_position_dict<std::string, Index>(mtx_cols);

    ///////////////////////////
    // cell-type annotation  //
    ///////////////////////////

    Mat Z; // latent membership matrix

    if (r_annot_mat.isNotNull()) { // reading latent membership matrix
        Z = Rcpp::as<Mat>(Rcpp::NumericMatrix(r_annot_mat));
        if (r_lab_name.isNotNull()) {
            copy(Rcpp::StringVector(r_lab_name), lab_name);
            ASSERT_RETL(lab_name.size() == Z.cols(), "|annot| != |Z's cols|");
        } else {
            for (auto k = 0; k < Z.cols(); ++k)
                lab_name.push_back("ct" + std::to_string(k + 1));
        }

    } else { // reading latent membership assignment vector

        if (r_annot.isNotNull()) {
            copy(Rcpp::StringVector(r_annot), annot);
        } else {
            const std::string _lab = "ct1"; // default: just one cell type
            annot.resize(cols.size());      //
            std::fill(std::begin(annot), std::end(annot), _lab);
        }

        if (r_lab_name.isNotNull()) {
            copy(Rcpp::StringVector(r_lab_name), lab_name);
        } else {
            make_unique(annot, lab_name);
        }

        const Index K = lab_name.size();
        Z.resize(Nsample, K);
        Z.setZero();

        ASSERT_RETL(cols.size() == annot.size(), "|cols| != |annot|");

        auto lab_pos = make_position_dict<std::string, Index>(lab_name);

        for (Index j = 0; j < cols.size(); ++j) {
            if (mtx_pos.count(cols[j]) > 0) {
                const Index i = mtx_pos[cols[j]];
                if (lab_pos.count(annot[j]) > 0) {
                    const Index k = lab_pos[annot[j]];
                    Z(i, k) = 1.;
                }
            }
        }
    }

    TLOG("" << std::endl << Z.transpose() * Mat::Ones(Nsample, 1));

    //////////////////////////////
    // Indexing all the columns //
    //////////////////////////////

    const std::string idx_file = mtx_file + ".index";

    std::vector<Index> mtx_idx_tab;

    if (!file_exists(idx_file)) // if needed
        CHK_RETL(mmutil::index::build_mmutil_index(mtx_file, idx_file));

    CHK_RETL(mmutil::index::read_mmutil_index(idx_file, mtx_idx_tab));

    CHK_RETL(mmutil::index::check_index_tab(mtx_file, mtx_idx_tab));

    mmutil::io::mm_info_reader_t info;
    CHK_RETL(mmutil::bgzf::peek_bgzf_header(mtx_file, info));

    ASSERT_RETL(Nsample == info.max_col, "Should have matched .mtx.gz");

    ///////////////
    // data type //
    ///////////////

    paired_data_t pdata(mtx_file,
                        mtx_idx_tab,
                        mtx_cols,
                        cols,
                        KNN(knn_cell),
                        KNN(knn_indv),
                        BILINK(KNN_BILINK),
                        NNLIST(KNN_NNLIST));

    pdata.set_individual_info(indv);

    const Index Nind = pdata.num_individuals();
    ASSERT_RETL(Nind > 1, "at least two individuals");

    CHK_RETL(pdata.build_dictionary(Rcpp::NumericMatrix(r_V), NUM_THREADS));

    TLOG("Identified " << Nind << " individuals");
    TLOG("" << std::endl << Z.transpose() * Mat::Ones(Nsample, 1));

    ///////////////////////////
    // For each individual i //
    ///////////////////////////

    const Index D = info.max_row;
    const Index K = Z.cols();

    auto indv_pairs = pdata.match_individuals();
    const Index Npairs = indv_pairs.size();

    std::vector<std::string> out_col_names(Npairs * K);
    out_col_names.reserve(Npairs * K);
    std::fill(out_col_names.begin(), out_col_names.end(), "");

    TLOG("Total " << out_col_names.size() << " pairs");

    Mat delta(D, K * Npairs);
    Mat ln_delta(D, K * Npairs);
    delta.setZero();
    ln_delta.setZero();

    Mat mu(D, K * Npairs);
    Mat ln_mu(D, K * Npairs);
    mu.setZero();
    ln_mu.setZero();

    Mat covar(pdata.rank(), K * Npairs);
    covar.setZero();

    Mat delta_sd(D, K * Npairs);
    Mat ln_delta_sd(D, K * Npairs);
    delta_sd.setZero();
    ln_delta_sd.setZero();

    Mat mu_sd(D, K * Npairs);
    Mat ln_mu_sd(D, K * Npairs);
    mu_sd.setZero();
    ln_mu_sd.setZero();

    Mat delta_sum(D, K * Npairs);
    Mat delta_mean(D, K * Npairs);
    Mat nobs(D, K * Npairs);
    delta_sum.setZero();
    delta_mean.setZero();
    nobs.setZero();

    is_positive_op<Mat> is_positive; // check nnz > 0 between pairs

    Index npair_proc = 0;
    TLOG("Processed: " << npair_proc);

    using RowVec = Eigen::internal::plain_row_type<Mat>::type;

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
    for (Index pi = 0; pi < indv_pairs.size(); ++pi) {
        auto pp = indv_pairs.at(pi);
        Index ii = std::get<0>(pp), jj = std::get<1>(pp);
        const std::string indv_name_ii = pdata.indv_name(ii);
        const std::string indv_name_jj = pdata.indv_name(jj);
        TLOG("compare " << indv_name_ii << " vs. " << indv_name_jj);

        Mat yy = pdata.read_block(ii);                            // D x N
        Mat zz = row_sub(Z, pdata.cell_indexes(ii));              // N x K
        zz.transposeInPlace();                                    // Z: K x N
        Mat y0 = pdata.read_matched_block(ii, jj, IMPUTE_BY_KNN); // D x N

        Mat y_sum = yy * zz.transpose();   // D x K
        Mat y0_sum = y0 * zz.transpose();  // D x K
        Mat delta_sum_ij = y_sum - y0_sum; // difference
        Mat delta_mean_ij = (delta_sum_ij.array().rowwise() /
                             zz.transpose().colwise().sum().array());
        Mat nobs_ij =
            y_sum.unaryExpr(is_positive) + y0_sum.unaryExpr(is_positive);

        Vec u_ij = pdata.read_matched_covar(ii, jj);

        TLOG("Estimating [" << ii << ", #cells=" << yy.cols() << "] vs. "
                            << "Estimating [" << jj << ", #cells=" << y0.cols()
                            << "]");

        poisson_t pois(yy, zz, y0, zz, a0, b0);
        pois.optimize();
        const Mat mu_ij = pois.mu_DK();
        const Mat mu_sd_ij = pois.mu_sd_DK();
        const Mat ln_mu_ij = pois.ln_mu_DK();
        const Mat ln_mu_sd_ij = pois.ln_mu_sd_DK();

        pois.residual_optimize();
        const Mat delta_ij = pois.residual_mu_DK();
        const Mat delta_sd_ij = pois.residual_mu_sd_DK();
        const Mat ln_delta_ij = pois.ln_residual_mu_DK();
        const Mat ln_delta_sd_ij = pois.ln_residual_mu_sd_DK();

        // auto storage_index = [&K, &pi](const Index k) { return K * pi + k; };
        for (Index k = 0; k < K; ++k) {
            const Index s = K * pi + k;
            delta.col(s) = delta_ij.col(k);
            delta_sd.col(s) = delta_sd_ij.col(k);
            ln_delta.col(s) = ln_delta_ij.col(k);
            ln_delta_sd.col(s) = ln_delta_sd_ij.col(k);

            mu.col(s) = mu_ij.col(k);
            mu_sd.col(s) = mu_sd_ij.col(k);
            ln_mu.col(s) = ln_mu_ij.col(k);
            ln_mu_sd.col(s) = ln_mu_sd_ij.col(k);

            delta_sum.col(s) = delta_sum_ij.col(k);
            delta_mean.col(s) = delta_mean_ij.col(k);
            nobs.col(s) = nobs_ij.col(k);

            covar.col(s) = u_ij;

            const std::string pair_name =
                indv_name_ii + "_" + indv_name_jj + "_" + lab_name.at(k);
            out_col_names[s] = pair_name;
        }

        TLOG("Processed: " << (++npair_proc) << " / " << Npairs << std::endl);
    }

    TLOG("Writing down the estimated effects");

    /////////////////////////////////
    // output individual-level KNN //
    /////////////////////////////////

    const std::size_t Nout = indv_pairs.size();

    Rcpp::IntegerVector obs_index(Nout, NA_INTEGER);
    Rcpp::IntegerVector matched_index(Nout, NA_INTEGER);
    Rcpp::NumericVector weight_vec(Nout, NA_REAL);
    Rcpp::NumericVector distance_vec(Nout, NA_REAL);
    Rcpp::StringVector obs_name(Nout, "");
    Rcpp::StringVector matched_name(Nout, "");

    for (std::size_t pi = 0; pi < Nout; ++pi) {
        auto pp = indv_pairs.at(pi);
        const Index ii = std::get<0>(pp), jj = std::get<1>(pp);
        const Scalar w = std::get<2>(pp), d = std::get<3>(pp);
        const std::string indv_name_ii = pdata.indv_name(ii);
        const std::string indv_name_jj = pdata.indv_name(jj);

        obs_index[pi] = ii + 1;
        matched_index[pi] = jj + 1;
        obs_name[pi] = indv_name_ii;
        matched_name[pi] = indv_name_jj;
        weight_vec[pi] = w;
        distance_vec[pi] = d;
    }

    /////////////////////
    // output matrices //
    /////////////////////

    std::vector<std::string> out_row_names;
    CHK_RETL(read_vector_file(row_file, out_row_names));

    auto named_mat = [&out_row_names, &out_col_names](const Mat &xx) {
        Rcpp::NumericMatrix x = Rcpp::wrap(xx);
        if (xx.rows() == out_row_names.size() &&
            xx.cols() == out_col_names.size()) {
            Rcpp::rownames(x) = Rcpp::wrap(out_row_names);
            Rcpp::colnames(x) = Rcpp::wrap(out_col_names);
        }
        return x;
    };

    auto col_named_mat = [&out_row_names, &out_col_names](const Mat &xx) {
        Rcpp::NumericMatrix x = Rcpp::wrap(xx);
        if (xx.cols() == out_col_names.size()) {
            Rcpp::colnames(x) = Rcpp::wrap(out_col_names);
        }
        return x;
    };

    Rcpp::List _knn =
        Rcpp::List::create(Rcpp::_["obs.index"] = obs_index,
                           Rcpp::_["matched.index"] = matched_index,
                           Rcpp::_["weight"] = weight_vec,
                           Rcpp::_["dist"] = distance_vec,
                           Rcpp::_["obs.name"] = obs_name,
                           Rcpp::_["matched.name"] = matched_name);

    const Mat Vind = pdata.export_covar_indv();

    return Rcpp::List::create(Rcpp::_["delta"] = named_mat(delta),
                              Rcpp::_["sum.delta"] = named_mat(delta_sum),
                              Rcpp::_["mean.delta"] = named_mat(delta_mean),
                              Rcpp::_["nobs"] = named_mat(nobs),
                              Rcpp::_["delta.sd"] = named_mat(delta_sd),
                              Rcpp::_["ln.delta"] = named_mat(ln_delta),
                              Rcpp::_["ln.delta.sd"] = named_mat(ln_delta_sd),
                              Rcpp::_["mu"] = named_mat(mu),
                              Rcpp::_["mu.sd"] = named_mat(mu_sd),
                              Rcpp::_["ln.mu"] = named_mat(ln_mu),
                              Rcpp::_["ln.mu.sd"] = named_mat(ln_mu_sd),
                              Rcpp::_["knn"] = _knn,
                              Rcpp::_["covar.matched"] = col_named_mat(covar),
                              Rcpp::_["covar.ind"] = Vind);
}

//' Create pseudo-bulk data by aggregating columns
//'
//' @param mtx_file data file
//' @param row_file row file
//' @param col_file column file
//' @param r_cols cell (col) names
//' @param r_indv membership for the cells (\code{r_cols})
//' @param r_annot label annotation for the (\code{r_cols})
//' @param r_lab_name label names (default: everything in \code{r_annot})
//' @param r_annot_mat label annotation matrix (cell x type) (default: NULL)
//' @param r_trt treatment assignment (default: NULL)
//' @param r_V SVD factors (default: NULL)
//' @param a0 hyperparameter for gamma(a0, b0) (default: 1)
//' @param b0 hyperparameter for gamma(a0, b0) (default: 1)
//' @param eps small number (default: 1e-8)
//' @param knn k-NN matching
//' @param KNN_BILINK num. of bidirectional links (default: 10)
//' @param KNN_NNLIST num. of nearest neighbor lists (default: 10)
//' @param NUM_THREADS number of threads for multi-core processing
//' @param IMPUTE_BY_KNN imputation by kNN alone (default: FALSE)
//'
//' @return a list of inference results
//'
// [[Rcpp::export]]
Rcpp::List
rcpp_mmutil_aggregate(
    const std::string mtx_file,
    const std::string row_file,
    const std::string col_file,
    Rcpp::Nullable<Rcpp::StringVector> r_cols = R_NilValue,
    Rcpp::Nullable<Rcpp::StringVector> r_indv = R_NilValue,
    Rcpp::Nullable<Rcpp::StringVector> r_annot = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericMatrix> r_annot_mat = R_NilValue,
    Rcpp::Nullable<Rcpp::StringVector> r_lab_name = R_NilValue,
    Rcpp::Nullable<Rcpp::StringVector> r_trt = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericMatrix> r_V = R_NilValue,
    const double a0 = 1.0,
    const double b0 = 1.0,
    const double eps = 1e-8,
    const std::size_t knn = 10,
    const std::size_t KNN_BILINK = 10,
    const std::size_t KNN_NNLIST = 10,
    const std::size_t NUM_THREADS = 1,
    const bool IMPUTE_BY_KNN = false)
{
    Eigen::initParallel();
    CHK_RETL(mmutil::bgzf::convert_bgzip(mtx_file));

    std::vector<std::string> mtx_cols;
    CHK_RETL(read_vector_file(col_file, mtx_cols));

    //////////////////////////////
    // check column annotations //
    //////////////////////////////

    std::vector<std::string> cols;
    std::vector<std::string> indv;
    std::vector<std::string> annot;
    std::vector<std::string> lab_name;

    if (r_cols.isNotNull()) {
        copy(Rcpp::StringVector(r_cols), cols);
    } else {
        cols.reserve(mtx_cols.size());
        std::copy(std::begin(mtx_cols),
                  std::end(mtx_cols),
                  std::back_inserter(cols));
    }

    if (r_indv.isNotNull()) {
        copy(Rcpp::StringVector(r_indv), indv);
    } else {
        indv.resize(mtx_cols.size());
        std::string _indv = "ind1";
        std::fill(std::begin(indv), std::end(indv), _indv);
    }

    ASSERT_RETL(cols.size() == indv.size(), "|cols| != |indv|");

    ////////////////////////
    // universal position //
    ////////////////////////

    const Index Nsample = mtx_cols.size();
    auto mtx_pos = make_position_dict<std::string, Index>(mtx_cols);

    ///////////////////////////
    // Individual membership //
    ///////////////////////////

    std::vector<std::string> indv_membership(Nsample);
    std::fill(std::begin(indv_membership), std::end(indv_membership), "?");

    TLOG("Allocating cells to individuals");

    for (Index j = 0; j < cols.size(); ++j) {
        if (mtx_pos.count(cols[j]) > 0) {
            const Index i = mtx_pos[cols[j]];
            indv_membership[i] = indv[j];
        }
    }

    std::vector<std::string> indv_id_name;
    std::vector<Index> indv_map; // map: col -> indv index

    std::tie(indv_map, indv_id_name, std::ignore) =
        make_indexed_vector<std::string, Index>(indv_membership);

    auto indv_index_set = make_index_vec_vec(indv_map);

    const Index Nind = indv_id_name.size();

    TLOG("Identified " << Nind << " individuals");

    ///////////////////////////
    // cell-type annotation  //
    ///////////////////////////

    Mat Z; // latent membership matrix

    if (r_annot_mat.isNotNull()) { // reading latent membership matrix
        Z = Rcpp::as<Mat>(Rcpp::NumericMatrix(r_annot_mat));
        if (r_lab_name.isNotNull()) {
            copy(Rcpp::StringVector(r_lab_name), lab_name);
            ASSERT_RETL(lab_name.size() == Z.cols(), "|annot| != |Z's cols|");
        } else {
            for (auto k = 0; k < Z.cols(); ++k)
                lab_name.push_back("ct" + std::to_string(k + 1));
        }

    } else { // reading latent membership assignment vector

        if (r_annot.isNotNull()) {
            copy(Rcpp::StringVector(r_annot), annot);
        } else {
            const std::string _lab = "ct1"; // default: just one cell type
            annot.resize(cols.size());      //
            std::fill(std::begin(annot), std::end(annot), _lab);
        }

        if (r_lab_name.isNotNull()) {
            copy(Rcpp::StringVector(r_lab_name), lab_name);
        } else {
            make_unique(annot, lab_name);
        }

        const Index K = lab_name.size();
        Z.resize(Nsample, K);
        Z.setZero();

        ASSERT_RETL(cols.size() == annot.size(), "|cols| != |annot|");

        auto lab_pos = make_position_dict<std::string, Index>(lab_name);

        for (Index j = 0; j < cols.size(); ++j) {
            if (mtx_pos.count(cols[j]) > 0) {
                const Index i = mtx_pos[cols[j]];
                if (lab_pos.count(annot[j]) > 0) {
                    const Index k = lab_pos[annot[j]];
                    Z(i, k) = 1.;
                }
            }
        }
    }

    TLOG("" << std::endl << Z.transpose() * Mat::Ones(Nsample, 1));

    //////////////////////////////
    // Indexing all the columns //
    //////////////////////////////

    const std::string idx_file = mtx_file + ".index";

    std::vector<Index> mtx_idx_tab;

    if (!file_exists(idx_file)) // if needed
        CHK_RETL(mmutil::index::build_mmutil_index(mtx_file, idx_file));

    CHK_RETL(mmutil::index::read_mmutil_index(idx_file, mtx_idx_tab));

    CHK_RETL(mmutil::index::check_index_tab(mtx_file, mtx_idx_tab));

    mmutil::io::mm_info_reader_t info;
    CHK_RETL(mmutil::bgzf::peek_bgzf_header(mtx_file, info));

    ASSERT_RETL(Nsample == info.max_col, "Should have matched .mtx.gz");

    ////////////////////////////////////////////////////////
    // A wrapper data structure to retrieve matched cells //
    ////////////////////////////////////////////////////////

    bool do_cocoa = false;

    matched_data_t matched_data(mtx_file,
                                mtx_idx_tab,
                                mtx_cols,
                                cols,
                                KNN(knn),
                                BILINK(KNN_BILINK),
                                NNLIST(KNN_NNLIST));

    if (r_trt.isNotNull()) {
        std::vector<std::string> trt = copy(Rcpp::StringVector(r_trt));
        matched_data.set_treatment_info(trt);
    }

    ////////////////////////////////////////
    // build dictionaries for fast lookup //
    ////////////////////////////////////////

    if (r_V.isNotNull()) {
        CHK_RETL(matched_data.build_dictionary(Rcpp::NumericMatrix(r_V),
                                               NUM_THREADS));
        do_cocoa = true;
    } else {
        if (r_trt.isNotNull()) {
            WLOG("For counterfactual analysis, we need covariate matrix r_V");
        }
    }

    const Index Ntrt = matched_data.num_treatment();

    if (do_cocoa) {
        ASSERT_RETL(Nind > 1, "at least two individuals");
    }

    ///////////////////////////
    // For each individual i //
    ///////////////////////////

    const Index D = info.max_row;
    const Index K = Z.cols();
    std::vector<std::string> out_col_names(K * Nind);
    std::fill(out_col_names.begin(), out_col_names.end(), "");

    Mat obs_sum(D, K * Nind);
    Mat obs_mean(D, K * Nind);

    Mat obs_mu(D, K * Nind);
    Mat obs_mu_sd(D, K * Nind);
    Mat obs_ln_mu(D, K * Nind);
    Mat obs_ln_mu_sd(D, K * Nind);

    obs_sum.setZero();
    obs_mean.setZero();
    obs_mu.setZero();
    obs_mu_sd.setZero();
    obs_ln_mu.setZero();
    obs_ln_mu_sd.setZero();

    Mat cf_mu;
    Mat cf_mu_sd;
    Mat cf_ln_mu;
    Mat cf_ln_mu_sd;

    Mat resid_mu;
    Mat resid_mu_sd;
    Mat resid_ln_mu;
    Mat resid_ln_mu_sd;

    if (do_cocoa) {
        cf_mu.resize(D, K * Nind);
        cf_mu_sd.resize(D, K * Nind);
        cf_ln_mu.resize(D, K * Nind);
        cf_ln_mu_sd.resize(D, K * Nind);

        resid_mu.resize(D, K * Nind);
        resid_mu_sd.resize(D, K * Nind);
        resid_ln_mu.resize(D, K * Nind);
        resid_ln_mu_sd.resize(D, K * Nind);

        cf_ln_mu.setZero();
        cf_ln_mu_sd.setZero();
        cf_mu.setZero();
        cf_mu_sd.setZero();

        resid_mu.setZero();
        resid_mu_sd.setZero();
        resid_ln_mu.setZero();
        resid_ln_mu_sd.setZero();
    }

    Index nind_proc = 0;

    Vec obs_rho(Nsample);

    using namespace mmutil::io;

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
    for (Index ii = 0; ii < Nind; ++ii) {
        auto storage_index = [&K, &ii](const Index k) { return K * ii + k; };
        const std::string indv_name = indv_id_name.at(ii);

        // Y: features x columns
        const std::vector<Index> &cols_i = indv_index_set.at(ii);
        Mat yy = read_eigen_sparse_subset_col(mtx_file, mtx_idx_tab, cols_i);
        Mat zz = row_sub(Z, cols_i); //
        zz.transposeInPlace();       // Z: K x N

        TLOG("Estimating [ind=" << ii << ", #cells=" << cols_i.size() << "]");

        ///////////////////////////////////////////////////
        // control cells from different treatment groups //
        ///////////////////////////////////////////////////

        if (do_cocoa) {
            Mat y0 = matched_data.read_cf_block(cols_i, IMPUTE_BY_KNN);
            poisson_t pois(yy, zz, y0, zz, a0, b0);
            pois.optimize();

            const Mat cf_mu_i = pois.mu_DK();
            const Mat cf_mu_sd_i = pois.mu_sd_DK();
            const Mat cf_ln_mu_i = pois.ln_mu_DK();
            const Mat cf_ln_mu_sd_i = pois.ln_mu_sd_DK();

            for (Index k = 0; k < K; ++k) {
                const Index s = storage_index(k);
                cf_mu.col(s) = cf_mu_i.col(k);
                cf_mu_sd.col(s) = cf_mu_sd_i.col(k);
                cf_ln_mu.col(s) = cf_ln_mu_i.col(k);
                cf_ln_mu_sd.col(s) = cf_ln_mu_sd_i.col(k);
            }

            pois.residual_optimize();

            const Mat resid_mu_i = pois.residual_mu_DK();
            const Mat resid_mu_sd_i = pois.residual_mu_sd_DK();

            const Mat resid_ln_mu_i = pois.ln_residual_mu_DK();
            const Mat resid_ln_mu_sd_i = pois.ln_residual_mu_sd_DK();

            for (Index k = 0; k < K; ++k) {
                const Index s = storage_index(k);
                resid_mu.col(s) = resid_mu_i.col(k);
                resid_mu_sd.col(s) = resid_mu_sd_i.col(k);
                resid_ln_mu.col(s) = resid_ln_mu_i.col(k);
                resid_ln_mu_sd.col(s) = resid_ln_mu_sd_i.col(k);
            }
        }

        /////////////////////////
        // vanilla aggregation //
        /////////////////////////

        {
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

                const Index s = storage_index(k);
                const std::string sname = indv_name + "_" + lab_name.at(k);
                out_col_names[s] = sname;

                const Scalar _denom_k = _denom(k);

                if (_denom_k > eps) {
                    obs_sum.col(s) = _sum.col(k);
                    obs_mean.col(s) = _sum.col(k) / _denom_k;
                    obs_mu.col(s) = _mu.col(k);
                    obs_mu_sd.col(s) = _mu_sd.col(k);
                    obs_ln_mu.col(s) = _ln_mu.col(k);
                    obs_ln_mu_sd.col(s) = _ln_mu_sd.col(k);
                }
            }

            for (Index j = 0; j < cols_i.size(); ++j) {
                const Index jj = cols_i[j];
                obs_rho(jj) = _rho(j, 0);
            }
        }

        TLOG("Processed: " << (++nind_proc) << std::endl);
    }

    TLOG("Writing down the estimated effects");

    /////////////////////
    // output matrices //
    /////////////////////

    std::vector<std::string> out_row_names;
    CHK_RETL(read_vector_file(row_file, out_row_names));

    auto named_mat = [&out_row_names, &out_col_names](const Mat &xx) {
        Rcpp::NumericMatrix x = Rcpp::wrap(xx);
        if (xx.rows() == out_row_names.size() &&
            xx.cols() == out_col_names.size()) {
            Rcpp::rownames(x) = Rcpp::wrap(out_row_names);
            Rcpp::colnames(x) = Rcpp::wrap(out_col_names);
        }
        return x;
    };

    return Rcpp::List::create(Rcpp::_["rho"] = obs_rho,
                              Rcpp::_["mean"] = named_mat(obs_mean),
                              Rcpp::_["sum"] = named_mat(obs_sum),
                              Rcpp::_["mu"] = named_mat(obs_mu),
                              Rcpp::_["mu.sd"] = named_mat(obs_mu_sd),
                              Rcpp::_["ln.mu"] = named_mat(obs_ln_mu),
                              Rcpp::_["ln.mu.sd"] = named_mat(obs_ln_mu_sd),
                              Rcpp::_["cf.mu"] = named_mat(cf_mu),
                              Rcpp::_["cf.mu.sd"] = named_mat(cf_mu_sd),
                              Rcpp::_["cf.ln.mu"] = named_mat(cf_ln_mu),
                              Rcpp::_["cf.ln.mu.sd"] = named_mat(cf_ln_mu_sd),
                              Rcpp::_["resid.mu"] = named_mat(resid_mu),
                              Rcpp::_["resid.mu.sd"] = named_mat(resid_mu_sd),
                              Rcpp::_["resid.ln.mu"] = named_mat(resid_ln_mu),
                              Rcpp::_["resid.ln.mu.sd"] =
                                  named_mat(resid_ln_mu_sd));
}
