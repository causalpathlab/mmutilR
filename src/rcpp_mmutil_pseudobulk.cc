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

// [[Rcpp::plugins(openmp)]]

#include "io.hh"
#include "std_util.hh"

//' Create pseudo-bulk data by aggregating columns
//'
//' @param mtx_file data file
//' @param row_file row file
//' @param col_file column file
//' @param r_cols column names
//' @param r_indv individual annotation
//' @param r_annot label annotation
//' @param r_lab_name label names
//' @param r_trt treatment assignment (default: NULL)
//' @param r_V SVD factors (default: NULL)
//' @param a0 hyperparameter for gamma(a0, b0) (default: 1e-4)
//' @param b0 hyperparameter for gamma(a0, b0) (default: 1e-4)
//' @param eps small number (default: 1e-8)
//' @param knn k-NN matching
//' @param KNN_BILINK # of bidirectional links (default: 10)
//' @param KNN_NNLIST # nearest neighbor lists (default: 10)
//' @param NUM_THREADS number of threads for multi-core processing
//' @param IMPUTE_BY_KNN imputation by kNN alone (default: false)
//'
//' @return a list of inference results
//'
//' @examples
//' options(stringsAsFactors = FALSE)
//' ## combine two different mu matrices
//' rr <- rgamma(1000, 1, 1) # 1000 cells
//' mm.1 <- matrix(rgamma(100 * 3, 1, 1), 100, 3)
//' mm.1[1:10, ] <- rgamma(5, 1, .1)
//' mm.2 <- matrix(rgamma(100 * 3, 1, 1), 100, 3)
//' mm.2[11:20, ] <- rgamma(5, 1, .1)
//' mm <- cbind(mm.1, mm.2)
//' dat <- mmutilR::rcpp_mmutil_simulate_poisson(mm, rr, "sim_test")
//' rows <- read.table(dat$row)$V1
//' cols <- read.table(dat$col)$V1
//' ## marker feature
//' markers <- list(
//'   annot.1 = list(
//'     ct1 = rows[1:10],
//'     ct2 = rows[11:20]
//'   )
//' )
//' ## annotation on the MTX file
//' out <- mmutilR::rcpp_mmutil_annotate_columns(
//'        dat$row, dat$col,
//'        mtx_file = dat$mtx, pos_labels = markers)
//' annot <- out$annotation
//' ## prepare column to individual
//' .ind <- read.table(dat$indv, col.names = c("col", "ind"))
//' .annot.ind <- .ind$ind[match(annot$col, .ind$col)]
//' ## aggregate
//' agg <- mmutilR::rcpp_mmutil_aggregate(dat$mtx, dat$row, dat$col,
//'         annot$col, .annot.ind, annot$argmax, c("ct1", "ct2"))
//' ## show average marker features
//' print(round(agg$mean[1:20, ]))
//' unlink(list.files(pattern = "sim_test"))
//' ## Case-control simulation
//' .sim <- mmutilR::simulate_gamma_glm()
//' .dat <- mmutilR::rcpp_mmutil_simulate_poisson(.sim$obs.mu,
//'                                              .sim$rho,
//'                                              "sim_test")
//' ## find column-wise annotation
//' .annot <- read.table(.dat$indv,
//'                      col.names = c("col", "ind"))
//' .annot$trt <- .sim$W[match(.annot$ind, 1:length(.sim$W))]
//' .annot$ct <- "ct1"
//' ## simple PCA
//' .pca <- mmutilR::rcpp_mmutil_pca(.dat$mtx, 10)
//' .agg <- mmutilR::rcpp_mmutil_aggregate(mtx_file = .dat$mtx,
//'                                        row_file = .dat$row,
//'                                        col_file = .dat$col,
//'                                        r_cols = .annot$col,
//'                                        r_indv = .annot$ind,
//'                                        r_annot = .annot$ct,
//'                                        r_lab_name = "ct1",
//'                                        r_trt = .annot$trt,
//'                                        r_V = .pca$V,
//'                                        knn = 50,
//'                                        IMPUTE_BY_KNN = TRUE)
//' par(mfrow=c(1,3))
//' for(k in sample(.sim$causal, 3)) {
//'     y0 <- .agg$resid.mu[k, .sim$W == 0]
//'     y1 <- .agg$resid.mu[k, .sim$W == 1]
//'     boxplot(y0, y1)
//' }
//' ## clean up temp directory
//' unlink(list.files(pattern = "sim_test"))
//'
// [[Rcpp::export]]
Rcpp::List
rcpp_mmutil_aggregate(const std::string mtx_file,
                      const std::string row_file,
                      const std::string col_file,
                      Rcpp::StringVector r_cols,
                      Rcpp::StringVector r_indv,
                      Rcpp::StringVector r_annot,
                      Rcpp::StringVector r_lab_name,
                      Rcpp::Nullable<Rcpp::StringVector> r_trt = R_NilValue,
                      Rcpp::Nullable<Rcpp::NumericMatrix> r_V = R_NilValue,
                      const double a0 = 1e-4,
                      const double b0 = 1e-4,
                      const double eps = 1e-8,
                      const std::size_t knn = 10,
                      const std::size_t KNN_BILINK = 10,
                      const std::size_t KNN_NNLIST = 10,
                      const std::size_t NUM_THREADS = 1,
                      const bool IMPUTE_BY_KNN = false)
{

    CHECK(mmutil::bgzf::convert_bgzip(mtx_file));

    //////////////////////////////
    // check column annotations //
    //////////////////////////////

    std::vector<std::string> cols = copy(r_cols);
    std::vector<std::string> indv = copy(r_indv);
    std::vector<std::string> annot = copy(r_annot);
    std::vector<std::string> lab_name = copy(r_lab_name);

    const Index K = lab_name.size();

    ASSERT(cols.size() == indv.size(), "|cols| != |indv|");
    ASSERT(cols.size() == annot.size(), "|cols| != |annot|");

    std::vector<std::string> mtx_cols;
    CHECK(read_vector_file(col_file, mtx_cols));
    const Index Nsample = mtx_cols.size();
    auto mtx_pos = make_position_dict<std::string, Index>(mtx_cols);
    auto lab_pos = make_position_dict<std::string, Index>(lab_name);

    /////////////////////////////////////////////////
    // latent annotation and individual membership //
    /////////////////////////////////////////////////

    Mat Z(Nsample, K);
    Z.setZero();

    std::vector<std::string> indv_membership(Nsample);
    std::fill(std::begin(indv_membership), std::end(indv_membership), "?");

    TLOG("Reading latent & indv annotations");

    for (Index j = 0; j < cols.size(); ++j) {
        if (mtx_pos.count(cols[j]) > 0) {
            const Index i = mtx_pos[cols[j]];
            if (lab_pos.count(annot[j]) > 0) {
                const Index k = lab_pos[annot[j]];
                Z(i, k) = 1.;
            }
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

    TLOG("" << std::endl << Z.transpose() * Mat::Ones(Nsample, 1));

    //////////////////////////////
    // Indexing all the columns //
    //////////////////////////////

    const std::string idx_file = mtx_file + ".index";

    std::vector<Index> mtx_idx_tab;

    if (!file_exists(idx_file)) // if needed
        CHECK(mmutil::index::build_mmutil_index(mtx_file, idx_file));

    CHECK(mmutil::index::read_mmutil_index(idx_file, mtx_idx_tab));

    CHECK(mmutil::index::check_index_tab(mtx_file, mtx_idx_tab));

    mmutil::io::mm_info_reader_t info;
    CHECK(mmutil::bgzf::peek_bgzf_header(mtx_file, info));

    ASSERT(Nsample == info.max_col, "Should have matched .mtx.gz");

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
        CHECK(matched_data.build_dictionary(Rcpp::NumericMatrix(r_V),
                                            NUM_THREADS));
        do_cocoa = true;
    }

    const Index Ntrt = matched_data.num_treatment();

    ASSERT(Nind > 1, "Must have at least two individuals");

    ///////////////////////////
    // For each individual i //
    ///////////////////////////

    const Index D = info.max_row;

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

                out_col_names[s] = indv_name + "_" + lab_name.at(k);

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
    std::vector<std::string> out_row_names;
    CHECK(read_vector_file(row_file, out_row_names));

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
