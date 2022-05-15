#include "mmutil_match.hh"
#include "mmutil_spectral.hh"
#include "mmutil_util.hh"
#include "mmutil_index.hh"

// [[Rcpp::plugins(openmp)]]

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
//' @param NUM_THREADS number of threads for multi-core processing
//' @param BLOCK_SIZE number of columns per block
//'
//' @return a list of source, target, distance
//'
//' @examples
//' ## Generate some data
//' rr <- rgamma(100, 1, 6) # 100 cells
//' mm <- matrix(rgamma(100 * 3, 1, 1), 100, 3)
//' dat <- mmutilR::rcpp_mmutil_simulate_poisson(mm, rr, "sim_test")
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

    //////////////////////
    // check input data //
    //////////////////////

    ASSERT_RETL(file_exists(src_mtx), "No source data file");
    ASSERT_RETL(file_exists(tgt_mtx), "No target data file");

    CHK_RETL(mmutil::bgzf::convert_bgzip(src_mtx));
    CHK_RETL(mmutil::bgzf::convert_bgzip(tgt_mtx));
    CHK_RETL(mmutil::index::build_mmutil_index(src_mtx, src_mtx + ".index"));
    CHK_RETL(mmutil::index::build_mmutil_index(tgt_mtx, tgt_mtx + ".index"));

    mmutil::index::mm_info_reader_t info;
    CHK_RETL(mmutil::bgzf::peek_bgzf_header(tgt_mtx, info));
    const Index D = info.max_row;
    const Index Nsample = info.max_col;

    ///////////////////////////////////////////////
    // step 1. learn spectral on the target data //
    ///////////////////////////////////////////////

    Vec weights;
    if (file_exists(row_weight_file)) {
        std::vector<Scalar> ww;
        CHK_RETL(read_vector_file(row_weight_file, ww));
        weights = eigen_vector(ww);
    }

    Vec ww(D, 1);
    ww.setOnes();

    if (weights.size() > 0) {
        ASSERT_RETL(weights.rows() == D, "Found invalid weight vector");
        ww = weights;
    }

    svd_out_t svd = take_svd_online_em(tgt_mtx, ww, options, NUM_THREADS);

    Mat dict = svd.U; // feature x factor
    Mat d = svd.D;    // singular values
    Mat tgt = svd.V;  // sample x factor

    TLOG("Target matrix: " << tgt.rows() << " x " << tgt.cols());

    /////////////////////////////////////////////////////
    // step 2. project source data onto the same space //
    /////////////////////////////////////////////////////

    const Mat proj = dict * d.cwiseInverse().asDiagonal(); // feature x rank
    Mat src = take_proj_online(src_mtx, weights, proj, options, NUM_THREADS);

    TLOG("Source matrix: " << src.rows() << " x " << src.cols());

    //////////////////////////////
    // step 3. search kNN pairs //
    //////////////////////////////

    ASSERT_RETL(src.cols() == tgt.cols(),
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
        param_bilink = rank - 1;
        TLOG("Shrink M value: " << param_bilink << " < " << rank);
    }

    if (param_bilink < 2) {
        param_bilink = 2;
        TLOG("Increase small M value: " << param_bilink);
    }

    if (param_nnlist <= knn) {
        param_nnlist = knn + 1;
        WLOG("Increase small N value: " << param_nnlist);
    }

    CHK_RETL(search_knn<hnswlib::InnerProductSpace>(SrcDataT(src.data(),
                                                             src.rows(),
                                                             src.cols()),
                                                    TgtDataT(tgt.data(),
                                                             tgt.rows(),
                                                             tgt.cols()),
                                                    KNN(knn),
                                                    BILINK(param_bilink),
                                                    NNLIST(param_nnlist),
                                                    NUM_THREADS,
                                                    out_index));

    TLOG("Done kNN searches");

    const std::size_t nout = out_index.size();

    Rcpp::IntegerVector src_index(nout, NA_INTEGER);
    Rcpp::IntegerVector tgt_index(nout, NA_INTEGER);
    Rcpp::NumericVector dist_vec(nout, NA_REAL);

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
    for (std::size_t i = 0; i < out_index.size(); ++i) {
        auto &tt = out_index.at(i);
        src_index[i] = std::get<0>(tt) + 1;
        tgt_index[i] = std::get<1>(tt) + 1;
        dist_vec[i] = std::get<2>(tt);
    }

    return Rcpp::List::create(Rcpp::_["src.index"] = src_index,
                              Rcpp::_["tgt.index"] = tgt_index,
                              Rcpp::_["dist"] = dist_vec);
}
