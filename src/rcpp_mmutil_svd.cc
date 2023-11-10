// [[Rcpp::plugins(openmp)]]

#include "mmutil.hh"
#include "mmutil_spectral.hh"
#include "mmutil_match.hh"
#include "mmutil_util.hh"
#include "mmutil_index.hh"
#include "progress.hh"
#include "mmutil_bbknn.hh"

//' Approximate SVD
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
//' .pc <- mmutilR::rcpp_mmutil_svd(.dat$mtx, 3, TAKE_LN = FALSE)
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
rcpp_mmutil_svd(const std::string mtx_file,
                const std::size_t RANK,
                const bool TAKE_LN = true,
                const double TAU = 1.,
                const double COL_NORM = 1e4,
                const std::size_t EM_ITER = 0,
                const double EM_TOL = 1e-4,
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
