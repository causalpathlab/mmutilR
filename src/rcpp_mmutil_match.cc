#include "mmutil_match.hh"
#include "mmutil_spectral.hh"
#include "mmutil_util.hh"
#include "mmutil_index.hh"

//' Match the columns of two MTX files
//'
//' @param src_mtx source data file
//' @param tgt_mtx target data file
//' @param knn k-nearest neighbour
//' @param RANK SVD rank
//' @param TAKE_LN take log(1 + x) trans or not
//' @param TAU regularization parameter (default = 1)
//' @param COL_NORM column normalization (default: 1e4)
//' @param EM_ITER EM iteration for factorization (default: 10)
//' @param EM_TOL EM convergence (default: 1e-4)
//' @param LU_ITER LU iteration (default: 5)
//' @param KNN_BILINK # of bidirectional links (default: 10)
//' @param KNN_NNLIST # nearest neighbor lists (default: 10)
//' @param row_weight_file row-wise weight file
//'
//' @return a list of source, target, distance
//'
//' @examples
//' ## Generate some data
//' rr <- rgamma(100, 1, 6) # 100 cells
//' mm <- matrix(rgamma(100 * 3, 1, 1), 100, 3)
//' dat <- mmutilR::rcpp_simulate_poisson_data(mm, rr, "sim_test")
//' .matched <- mmutilR::rcpp_mmutil_match_files(dat$mtx, dat$mtx,
//'                                              knn=1, RANK=5)
//' ## Do they match well?
//' mean(.matched$src.index == .matched$tgt.index)
//' summary(.matched$dist)
//' ## clean up temp directory
//' unlink(list.files(pattern = "sim_test"))
//'
// [[Rcpp::export]]
Rcpp::List
rcpp_mmutil_match_files(const std::string src_mtx,
                        const std::string tgt_mtx,
                        const std::size_t knn,
                        const std::size_t RANK,
                        const bool TAKE_LN = true,
                        const double TAU = 1.,
                        const double COL_NORM = 1e4,
                        const std::size_t EM_ITER = 10,
                        const double EM_TOL = 1e-4,
                        const std::size_t LU_ITER = 5,
                        const std::size_t KNN_BILINK = 10,
                        const std::size_t KNN_NNLIST = 10,
                        const std::string row_weight_file = "")
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

    //////////////////////
    // check input data //
    //////////////////////

    ASSERT(file_exists(src_mtx), "No source data file");
    ASSERT(file_exists(tgt_mtx), "No target data file");

    CHECK(mmutil::bgzf::convert_bgzip(src_mtx));
    CHECK(mmutil::bgzf::convert_bgzip(tgt_mtx));
    CHECK(mmutil::index::build_mmutil_index(src_mtx, src_mtx + ".index"));
    CHECK(mmutil::index::build_mmutil_index(tgt_mtx, tgt_mtx + ".index"));

    mmutil::index::mm_info_reader_t info;
    CHECK(mmutil::bgzf::peek_bgzf_header(tgt_mtx, info));
    const Index D = info.max_row;
    const Index Nsample = info.max_col;

    ///////////////////////////////////////////////
    // step 1. learn spectral on the target data //
    ///////////////////////////////////////////////

    Vec weights;
    if (file_exists(row_weight_file)) {
        std::vector<Scalar> ww;
        CHECK(read_vector_file(row_weight_file, ww));
        weights = eigen_vector(ww);
    }

    Vec ww(D, 1);
    ww.setOnes();

    if (weights.size() > 0) {
        ASSERT(weights.rows() == D, "Found invalid weight vector");
        ww = weights;
    }

    svd_out_t svd = take_svd_online_em(tgt_mtx, ww, options);

    Mat dict = svd.U; // feature x factor
    Mat d = svd.D;    // singular values
    Mat tgt = svd.V;  // sample x factor

    TLOG("Target matrix: " << tgt.rows() << " x " << tgt.cols());

    /////////////////////////////////////////////////////
    // step 2. project source data onto the same space //
    /////////////////////////////////////////////////////

    const Mat proj = dict * d.cwiseInverse().asDiagonal(); // feature x rank
    Mat src = take_proj_online(src_mtx, weights, proj, options);

    TLOG("Source matrix: " << src.rows() << " x " << src.cols());

    //////////////////////////////
    // step 3. search kNN pairs //
    //////////////////////////////

    ASSERT(src.cols() == tgt.cols(),
           "Found different number of spectral features:"
               << src.cols() << " vs. " << tgt.cols());

    src.transposeInPlace(); // rank x samples
    tgt.transposeInPlace(); // rank x samples
    normalize_columns(src); // For cosine distance
    normalize_columns(tgt); //

    std::vector<std::tuple<Index, Index, Scalar>> out_index;

    TLOG("Running kNN search ...");

    const std::size_t rank = src.rows();

    std::size_t param_bilink = KNN_BILINK;
    std::size_t param_nnlist = KNN_NNLIST;

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

    CHECK(search_knn(SrcDataT(src.data(), src.rows(), src.cols()),
                     TgtDataT(tgt.data(), tgt.rows(), tgt.cols()),
                     KNN(knn),
                     BILINK(param_bilink),
                     NNLIST(param_nnlist),
                     out_index));

    TLOG("Done");

    Rcpp::NumericVector src_index;
    Rcpp::NumericVector tgt_index;
    Rcpp::NumericVector dist_vec;
    for (auto tt : out_index) {
        src_index.push_back(std::get<0>(tt) + 1);
        tgt_index.push_back(std::get<1>(tt) + 1);
        dist_vec.push_back(std::get<2>(tt));
    }

    return Rcpp::List::create(Rcpp::_["src.index"] = src_index,
                              Rcpp::_["tgt.index"] = tgt_index,
                              Rcpp::_["dist"] = dist_vec);
}
