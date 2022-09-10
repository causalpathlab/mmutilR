// [[Rcpp::plugins(openmp)]]

#include "mmutil.hh"
#include "mmutil_spectral.hh"
#include "mmutil_match.hh"
#include "mmutil_util.hh"
#include "mmutil_index.hh"
#include "progress.hh"
#include "mmutil_bbknn.hh"

//' Approximate PCA
//'
//' @param mtx_file data file (feature x n)
//' @param RANK SVD rank
//' @param TAKE_LN take log(1 + x) trans or not
//' @param TAU regularization parameter (default = 1)
//' @param COL_NORM column normalization
//' @param EM_ITER EM iteration for factorization (default: 10)
//' @param EM_TOL EM convergence (default: 1e-4)
//' @param LU_ITER LU iteration
//' @param row_weight_file row-wise weight file
//' @param NUM_THREADS number of threads for multi-core processing
//' @param BLOCK_SIZE number of columns per block
//'
//' @return a list of (1) U (2) D (3) V
//'
//' @examples
//' ## Generate some data
//' set.seed(1)
//' rr <- rgamma(1000, 1, 1) # 1000 cells
//' mm <- matrix(rgamma(100 * 3, 1, 1), 100, 3)
//' .dat <- mmutilR::rcpp_mmutil_simulate_poisson(mm, rr, "sim_test")
//' .pc <- mmutilR::rcpp_mmutil_pca(.dat$mtx, 3, TAKE_LN = FALSE)
//' .ind <- read.table(.dat$indv)
//' .col <- unlist(read.table(.dat$col))
//' .ind <- .ind[match(.col, .ind$V1), ]
//' plot(.pc$V[, 1], .pc$V[, 2], col = .ind$V2,
//'      xlab = "PC1", ylab = "PC2")
//' plot(.pc$V[, 2], .pc$V[, 3], col = .ind$V2,
//'      xlab = "PC2", ylab = "PC3")
//' ## clean up temp directory
//' unlink(list.files(pattern = "sim_test"))
//'
// [[Rcpp::export]]
Rcpp::List
rcpp_mmutil_pca(const std::string mtx_file,
                const std::size_t RANK,
                const bool TAKE_LN = true,
                const double TAU = 1.,
                const double COL_NORM = 1e4,
                const std::size_t EM_ITER = 0,
                const double EM_TOL = 1e-4,
                const std::size_t KNN_BILINK = 10,
                const std::size_t KNN_NNLIST = 10,
                const std::size_t LU_ITER = 5,
                const std::string row_weight_file = "",
                const std::size_t NUM_THREADS = 1,
                const std::size_t BLOCK_SIZE = 10000)
{

    spectral_options_t options;

    /////////////////////
    // fill in options //
    /////////////////////

    options.lu_iter = LU_ITER;
    options.tau = TAU;
    options.log_scale = TAKE_LN;
    options.col_norm = COL_NORM;
    options.rank = RANK;
    options.em_iter = EM_ITER;
    options.em_tol = EM_TOL;
    options.block_size = BLOCK_SIZE;

    ////////////////////////////////////////
    // Indexing all the columns if needed //
    ////////////////////////////////////////

    const std::string idx_file = mtx_file + ".index";

    if (!file_exists(idx_file)) // if needed
        CHECK(mmutil::index::build_mmutil_index(mtx_file, idx_file));

    mmutil::index::mm_info_reader_t info;
    CHECK(mmutil::bgzf::peek_bgzf_header(mtx_file, info));
    const Index D = info.max_row;
    const Index Nsample = info.max_col;

    TLOG("info: " << info.max_row << ", " << info.max_col << " --> "
                  << info.max_elem);

    Vec weights;
    if (file_exists(row_weight_file)) {
        std::vector<Scalar> ww;
        CHECK(read_vector_file(row_weight_file, ww));
        weights = eigen_vector(ww);
    }

    Vec ww(D, 1);
    ww.setOnes();

    if (weights.size() > 0) {
        ASSERT_RETL(weights.rows() == D, "Found invalid weight vector");
        ww = weights;
    }

    svd_out_t svd;
    if (EM_ITER > 0) {
        svd = take_svd_online_em(mtx_file, idx_file, ww, options, NUM_THREADS);
    } else {
        svd = take_svd_online(mtx_file, idx_file, ww, options, NUM_THREADS);
    }

    return Rcpp::List::create(Rcpp::_["U"] = svd.U,
                              Rcpp::_["D"] = svd.D,
                              Rcpp::_["V"] = svd.V);
}

//' BBKNN(Batch-balancing kNN)-adjusted PCA
//'
//' @param mtx_file data file (feature x n)
//' @param r_batches batch names (n x 1)
//' @param knn kNN parameter k
//' @param RANK SVD rank
//' @param TAKE_LN take log(1 + x) trans or not
//' @param TAU regularization parameter (default = 1)
//' @param COL_NORM column normalization
//' @param EM_ITER EM iteration for factorization (default: 10)
//' @param EM_TOL EM convergence (default: 1e-4)
//' @param LU_ITER LU iteration
//' @param KNN_BILINK num. of bidirectional links (default: 10)
//' @param KNN_NNLIST num. of nearest neighbor lists (default: 10)
//' @param row_weight_file row-wise weight file
//' @param NUM_THREADS number of threads for multi-core processing
//' @param BLOCK_SIZE number of columns per block
//'
//' @return a list of (1) factors.adjusted (2) U (3) D (4) V
//'
//' @examples
//' ## Generate some data
//' set.seed(1)
//' .sim <- mmutilR::simulate_gamma_glm(nind = 5, ncell.ind = 1000)
//' .dat <- mmutilR::rcpp_mmutil_simulate_poisson(.sim$obs.mu,
//'                                               .sim$rho,
//'                                               "sim_test")
//' .ind <- read.table(.dat$indv)
//' .col <- unlist(read.table(.dat$col))
//' .ind <- .ind[match(.col, .ind$V1), ]
//' .bbknn <- mmutilR::rcpp_mmutil_bbknn_pca(.dat$mtx,
//'                   r_batches = .ind$V2,
//'                   knn = 10, RANK = 3, TAKE_LN = TRUE)
//' plot(.bbknn$V[, 1], .bbknn$V[, 2], col = .ind$V2,
//'      xlab = "PC1", ylab = "PC2", main = "no BBKNN")
//' plot(.bbknn$factors.adjusted[, 1], .bbknn$factors.adjusted[, 2],
//'      col = .ind$V2,
//'      xlab = "PC1 (BBKNN)", ylab = "PC2 (BBKNN)")
//' ## clean up temp directory
//' unlink(list.files(pattern = "sim_test"))
//'
// [[Rcpp::export]]
Rcpp::List
rcpp_mmutil_bbknn_pca(const std::string mtx_file,
                      const Rcpp::StringVector &r_batches,
                      const std::size_t knn,
                      const std::size_t RANK,
                      const bool TAKE_LN = true,
                      const double TAU = 1.,
                      const double COL_NORM = 1e4,
                      const std::size_t EM_ITER = 0,
                      const double EM_TOL = 1e-4,
                      const std::size_t KNN_BILINK = 10,
                      const std::size_t KNN_NNLIST = 10,
                      const std::size_t LU_ITER = 5,
                      const std::string row_weight_file = "",
                      const std::size_t NUM_THREADS = 1,
                      const std::size_t BLOCK_SIZE = 10000)
{

    std::vector<std::string> batch_membership(r_batches.begin(),
                                              r_batches.end());

    ////////////////////////////////////////
    // Indexing all the columns if needed //
    ////////////////////////////////////////

    const std::string idx_file = mtx_file + ".index";

    if (!file_exists(idx_file)) // if needed
        CHECK(mmutil::index::build_mmutil_index(mtx_file, idx_file));

    mmutil::index::mm_info_reader_t info;
    CHECK(mmutil::bgzf::peek_bgzf_header(mtx_file, info));
    const Index D = info.max_row;
    const Index Nsample = info.max_col;

    TLOG("info: " << info.max_row << ", " << info.max_col << " -> "
                  << info.max_elem);

    //////////////////////
    // batch membership //
    //////////////////////

    ASSERT_RETL(batch_membership.size() == Nsample,
                "This batch membership vector mismatches with the mtx data");

    std::vector<std::string> batch_id_name;
    std::vector<Index> batch; // map: col -> batch index

    std::tie(batch, batch_id_name, std::ignore) =
        make_indexed_vector<std::string, Index>(batch_membership);

    auto batch_index_set = make_index_vec_vec(batch);

    ASSERT_RETL(batch.size() >= Nsample,
                "Need batch membership for each column");
    const Index Nbatch = batch_id_name.size();
    TLOG("Identified " << Nbatch << " batches");

    ///////////////////////////////////
    // weights for the rows/features //
    ///////////////////////////////////

    Vec weights;
    if (file_exists(row_weight_file)) {
        std::vector<Scalar> ww;
        CHECK(read_vector_file(row_weight_file, ww));
        weights = eigen_vector(ww);
    }

    Vec ww(D, 1);
    ww.setOnes();

    if (weights.size() > 0) {
        ASSERT_RETL(weights.rows() == D, "Found invalid weight vector");
        ww = weights;
    }

    /////////////////////
    // fill in options //
    /////////////////////

    spectral_options_t options;

    options.lu_iter = LU_ITER;
    options.tau = TAU;
    options.log_scale = TAKE_LN;
    options.col_norm = COL_NORM;
    options.rank = RANK;
    options.em_iter = EM_ITER;
    options.em_tol = EM_TOL;
    options.block_size = BLOCK_SIZE;

    ////////////////////////////////
    // Learn latent embedding ... //
    ////////////////////////////////

    TLOG("Training SVD for spectral matching ...");
    svd_out_t svd;
    if (EM_ITER > 0) {
        svd = take_svd_online_em(mtx_file, idx_file, ww, options, NUM_THREADS);
    } else {
        svd = take_svd_online(mtx_file, idx_file, ww, options, NUM_THREADS);
    }

    std::vector<std::tuple<Index, Index, Scalar, Scalar>> knn_index;
    CHECK(build_bbknn(svd,
                      batch_index_set,
                      knn,
                      knn_index,
                      KNN_BILINK,
                      KNN_NNLIST,
                      NUM_THREADS));

    SpMat W = build_eigen_sparse(knn_index, Nsample, Nsample);
    TLOG("A weighted adjacency matrix W");

    /////////////////////////////
    // adjusting spectral data //
    /////////////////////////////

    Mat Vorg = svd.V.transpose();
    Mat Vadj = Vorg;
    Mat Delta_feature(D, Nbatch);
    Mat Delta_factor(svd.V.cols(), Nbatch);
    Delta_feature.setZero();
    Delta_factor.setZero();

    Mat proj(svd.U.rows(), svd.U.cols());
    proj = svd.U * svd.D.cwiseInverse().asDiagonal(); // feature x rank
    TLOG("Found projection: " << proj.rows() << " x " << proj.cols());

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
    for (Index aa = 1; aa < Nbatch; ++aa) {
        const auto &batch_a = batch_index_set.at(aa);
        const Index nn_a = batch_a.size();

        Mat delta_a(svd.V.rows(), 1);
        delta_a.setZero();
        Scalar num_a = 0.;

        for (Index a_k = 0; a_k < nn_a; ++a_k) {
            const Index j = batch_a.at(a_k);

            for (SpMat::InnerIterator it(W, j); it; ++it) {

                const Index i = it.col();

                if (batch.at(i) < aa) { // mingle toward the previous ones

                    const Scalar wji = it.value();

                    delta_a += wji * (Vorg.col(j) - Vadj.col(i));
                    num_a += wji;
                }
            }
        }

        delta_a /= std::max(num_a, (Scalar)1.0);

        for (Index a_k = 0; a_k < nn_a; ++a_k) {
            const Index j = batch_a.at(a_k);
            Vadj.col(j) = Vadj.col(j) - delta_a;
        }

        Delta_factor.col(aa) = delta_a;
        Delta_feature.col(aa) = proj * delta_a;
    }

    TLOG("Writing down the results...");

    Vadj.transposeInPlace();
    Vorg.transposeInPlace();

    const std::size_t nout = knn_index.size();

    Rcpp::IntegerVector src_index(nout, NA_INTEGER);
    Rcpp::IntegerVector tgt_index(nout, NA_INTEGER);
    Rcpp::NumericVector weight_vec(nout, NA_REAL);
    Rcpp::NumericVector distance_vec(nout, NA_REAL);
    Rcpp::StringVector src_batch(nout, "");
    Rcpp::StringVector tgt_batch(nout, "");

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
    for (std::size_t i = 0; i < knn_index.size(); ++i) {
        auto &tt = knn_index.at(i);
        src_index[i] = std::get<0>(tt) + 1;
        tgt_index[i] = std::get<1>(tt) + 1;
        weight_vec[i] = std::get<2>(tt);
        distance_vec[i] = std::get<3>(tt);
        src_batch[i] = batch_id_name.at(batch.at(std::get<0>(tt)));
        tgt_batch[i] = batch_id_name.at(batch.at(std::get<1>(tt)));
    }

    return Rcpp::List::create(Rcpp::_["factors.adjusted"] = Vadj,
                              Rcpp::_["U"] = svd.U,
                              Rcpp::_["D"] = svd.D,
                              Rcpp::_["V"] = Vorg,
                              Rcpp::_["delta.samples"] = Delta_factor,
                              Rcpp::_["delta.features"] = Delta_feature,
                              Rcpp::_["knn"] =
                                  Rcpp::List::create(Rcpp::_["src.index"] =
                                                         src_index,
                                                     Rcpp::_["tgt.index"] =
                                                         tgt_index,
                                                     Rcpp::_["weight"] =
                                                         weight_vec,
                                                     Rcpp::_["dist"] =
                                                         distance_vec,
                                                     Rcpp::_["src.batch"] =
                                                         src_batch,
                                                     Rcpp::_["tgt.batch"] =
                                                         tgt_batch));
}
