// [[Rcpp::plugins(openmp)]]

#include "mmutil.hh"
#include "mmutil_spectral.hh"
#include "mmutil_match.hh"
#include "mmutil_util.hh"
#include "mmutil_index.hh"
#include "progress.hh"
#include "mmutil_bbknn.hh"

//' BBKNN(Batch-balancing kNN) adjustment of SVD factors
//'
//' @param r_svd_v (n x L) n number of data points
//' @param r_svd_u (m x L) m number of features (default: NULL)
//' @param r_svd_d (L x 1) singular values (default: NULL)
//' @param r_batches batch names (n x 1)
//' @param knn kNN parameter k
//' @param RECIPROCAL_MATCH do reciprocal match (default: TRUE)
//' @param KNN_BILINK num. of bidirectional links (default: 10)
//' @param KNN_NNLIST num. of nearest neighbor lists (default: 10)
//' @param NUM_THREADS number of threads for multi-core processing
//' @param USE_SINGULAR_VALUES Weight factors by the corresponding SVs
//'
//' @return a list of (1) factors.adjusted (2) D (3) V (4) knn
//'
//' @details
//'
//' Build batch-balancing kNN graph based on (V * D) or V data.
//'
//'
// [[Rcpp::export]]
Rcpp::List
rcpp_mmutil_bbknn(
    const Rcpp::NumericMatrix &r_svd_v,
    const Rcpp::Nullable<Rcpp::NumericMatrix> r_svd_u = R_NilValue,
    const Rcpp::Nullable<Rcpp::NumericMatrix> r_svd_d = R_NilValue,
    const Rcpp::Nullable<Rcpp::StringVector> r_batches = R_NilValue,
    const std::size_t knn = 10,
    const bool RECIPROCAL_MATCH = true,
    const std::size_t KNN_BILINK = 10,
    const std::size_t KNN_NNLIST = 10,
    const std::size_t NUM_THREADS = 1,
    const bool USE_SINGULAR_VALUES = false)
{

    const Mat svd_v = Rcpp::as<Mat>(r_svd_v);

    TLOG("feature matrix: " << svd_v.rows() << " x " << svd_v.cols());

    const Mat svd_d =
        (r_svd_d.isNotNull() ? Rcpp::as<Eigen::MatrixXf>(r_svd_d) :
                               Mat::Ones(svd_v.cols(), 1));

    const Mat svd_u =
        (r_svd_u.isNotNull() ? Rcpp::as<Eigen::MatrixXf>(r_svd_u) : Mat());

    const Index Nsample = svd_v.rows();

    std::vector<std::string> batch_membership;

    if (r_batches.isNotNull()) {
        Rcpp::StringVector _batches(r_batches);
        for (auto r : _batches) {
            batch_membership.emplace_back(r);
        }
    } else {
        for (Index j = 0; j < Nsample; ++j)
            batch_membership.emplace_back("no_batch");
    }

    //////////////////////
    // batch membership //
    //////////////////////

    ASSERT_RETL(batch_membership.size() == Nsample,
                "This batch membership vector mismatches with the mtx data");

    ASSERT_RETL(svd_v.cols() == svd_d.size(), "should have the same rank");

    std::vector<std::string> batch_id_name;
    std::vector<Index> batch; // map: col -> batch index

    std::tie(batch, batch_id_name, std::ignore) =
        make_indexed_vector<std::string, Index>(batch_membership);

    auto batch_index_set = make_index_vec_vec(batch);

    ASSERT_RETL(batch.size() >= Nsample,
                "Need batch membership for each column");
    const Index Nbatch = batch_id_name.size();
    TLOG("Identified " << Nbatch << " batches");

    std::vector<std::tuple<Index, Index, Scalar, Scalar>> knn_index;
    Mat VD_rank_sample;
    if (USE_SINGULAR_VALUES) {
        VD_rank_sample = (svd_v * svd_d.asDiagonal()).transpose();
    } else {
        VD_rank_sample = svd_v.transpose();
    }

    CHECK(build_bbknn(VD_rank_sample,
                      batch_index_set,
                      knn,
                      knn_index,
                      RECIPROCAL_MATCH,
                      KNN_BILINK,
                      KNN_NNLIST,
                      NUM_THREADS));

    SpMat W = build_eigen_sparse(knn_index, Nsample, Nsample);

    TLOG("Built a weighted adjacency matrix W");

    /////////////////////////////
    // adjusting spectral data //
    /////////////////////////////

    Mat VDorg = VD_rank_sample;
    Mat VDadj = VDorg;

    // Sort batch indexes in descending order of the sizes
    std::vector<Index> batch_size;
    for (Index a = 0; a < Nbatch; ++a) {
        batch_size.emplace_back(batch_index_set.at(a).size());
    }
    std::vector<Index> batch_order = std_argsort(batch_size);
    std::vector<Index> batch_rank(Nbatch);
    for (Index a = 0; a < Nbatch; ++a) {
        batch_rank[batch_order.at(a)] = a;
    }

    if (Nbatch > 1) {
        TLOG("Batch adjustment order: ");
        for (auto k : batch_order) {
            TLOG(k << " [" << batch_id_name.at(k)
                   << "] N=" << batch_size.at(k));
        }
    }

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
    for (Index a = 1; a < Nbatch; ++a) {

        const Index aa = batch_order.at(a);

        const auto &batch_a = batch_index_set.at(aa);

        const Index nn_a = batch_a.size();

        Mat delta_a(VDorg.rows(), 1);
        delta_a.setZero();
        Scalar num_a = 0.;

        for (Index a_k = 0; a_k < nn_a; ++a_k) {

            const Index j = batch_a.at(a_k);

            for (SpMat::InnerIterator it(W, j); it; ++it) {

                const Index i = it.index();  // other cell index
                const Index b = batch.at(i); // its batch
                if (batch_rank.at(b) < a) {  // mingle toward the previous ones
                    const Scalar wji = it.value();
                    delta_a += wji * (VDorg.col(j) - VDadj.col(i));
                    num_a += wji;
                }
            }
        }

        delta_a /= std::max(num_a, (Scalar)1.0);

        for (Index a_k = 0; a_k < nn_a; ++a_k) {
            const Index j = batch_a.at(a_k);
            VDadj.col(j) = VDadj.col(j) - delta_a;
        }
    }

    TLOG("Writing down the results...");

    VDadj.transposeInPlace();
    VDorg.transposeInPlace();

    Rcpp::List knn_out;
    {

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

        knn_out = Rcpp::List::create(Rcpp::_["src.index"] = src_index,
                                     Rcpp::_["tgt.index"] = tgt_index,
                                     Rcpp::_["weight"] = weight_vec,
                                     Rcpp::_["dist"] = distance_vec,
                                     Rcpp::_["src.batch"] = src_batch,
                                     Rcpp::_["tgt.batch"] = tgt_batch);
    }

    Rcpp::List knn_adj_out;
    if (Nbatch <= 1) {

        TLOG("No need to recompute the kNN graph.");

    } else {

        TLOG("Building kNN graph after adjusting batch effects.");

        Mat X = VDadj.transpose();
        std::size_t param_bilink = KNN_BILINK;
        std::size_t param_nnlist = KNN_NNLIST;

        if (param_bilink >= X.rows()) {
            WLOG("Shrink M value: " << param_bilink << " vs. " << X.rows());
            param_bilink = X.rows() - 1;
        }

        if (param_bilink < 2) {
            WLOG("too small M value");
            param_bilink = 2;
        }

        if (param_nnlist <= knn) {
            WLOG("too small N value");
            param_nnlist = knn + 1;
        }

        std::vector<std::tuple<Index, Index, Scalar>> backbone;
        std::vector<std::tuple<Index, Index, Scalar, Scalar>> knn_adj_index;

        CHECK(
            search_knn<hnswlib::L2Space>(SrcDataT(X.data(), X.rows(), X.cols()),
                                         TgtDataT(X.data(), X.rows(), X.cols()),
                                         KNN(knn),
                                         BILINK(param_bilink),
                                         NNLIST(param_nnlist),
                                         NUM_THREADS,
                                         backbone));

        auto backbone_rec = keep_reciprocal_knn(backbone);

        reweight_knn_graph(X, backbone_rec, knn, knn_adj_index, NUM_THREADS);

        const std::size_t nout = knn_adj_index.size();

        Rcpp::IntegerVector src_adj_index(nout, NA_INTEGER);
        Rcpp::IntegerVector tgt_adj_index(nout, NA_INTEGER);
        Rcpp::NumericVector dist_adj_vec(nout, NA_REAL);
        Rcpp::NumericVector weight_adj_vec(nout, NA_REAL);

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
        for (std::size_t i = 0; i < knn_adj_index.size(); ++i) {
            auto &tt = knn_adj_index.at(i);
            src_adj_index[i] = std::get<0>(tt) + 1;
            tgt_adj_index[i] = std::get<1>(tt) + 1;
            weight_adj_vec[i] = std::get<2>(tt);
            dist_adj_vec[i] = std::get<3>(tt);
        }

        knn_adj_out = Rcpp::List::create(Rcpp::_["src.index"] = src_adj_index,
                                         Rcpp::_["tgt.index"] = tgt_adj_index,
                                         Rcpp::_["weight"] = weight_adj_vec,
                                         Rcpp::_["dist"] = dist_adj_vec);
    }

    return Rcpp::List::create(Rcpp::_["factors.adjusted"] = VDadj,
                              Rcpp::_["U"] = svd_u,
                              Rcpp::_["D"] = svd_d,
                              Rcpp::_["V"] = svd_v,
                              Rcpp::_["knn"] = knn_out,
                              Rcpp::_["knn.adj"] = knn_adj_out);
}

//' BBKNN(Batch-balancing kNN)-adjusted SVD
//'
//' @param mtx_file data file (feature x n)
//' @param r_batches batch names (n x 1)
//' @param knn kNN parameter k
//' @param RANK SVD rank
//' @param RECIPROCAL_MATCH do reciprocal match (default: TRUE)
//' @param TAKE_LN take log(1 + x) trans or not
//' @param TAU regularization parameter (default: 1)
//' @param COL_NORM column normalization
//' @param EM_ITER EM iteration for factorization (default: 0)
//' @param EM_TOL EM convergence (default: 1e-4)
//' @param LU_ITER LU iteration
//' @param KNN_BILINK num. of bidirectional links (default: 10)
//' @param KNN_NNLIST num. of nearest neighbor lists (default: 10)
//' @param row_weight_file row-wise weight file
//' @param NUM_THREADS number of threads for multi-core processing
//' @param BLOCK_SIZE number of columns per block
//' @param USE_SINGULAR_VALUES Weight factors by the corresponding SVs
//'
//' @return a list of (1) factors.adjusted (2) D (3) V (4) knn
//'
// [[Rcpp::export]]
Rcpp::List
rcpp_mmutil_bbknn_mtx(const std::string mtx_file,
                      const Rcpp::StringVector &r_batches,
                      const std::size_t knn,
                      const std::size_t RANK,
                      const bool RECIPROCAL_MATCH = true,
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
                      const std::size_t BLOCK_SIZE = 10000,
                      const bool USE_SINGULAR_VALUES = false)
{

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

    //////////////////////////
    // BBKNN and adjustment //
    //////////////////////////

    return rcpp_mmutil_bbknn(Rcpp::wrap(svd.V),
                             Rcpp::wrap(svd.U),
                             Rcpp::wrap(svd.D),
                             r_batches,
                             knn,
                             RECIPROCAL_MATCH,
                             KNN_BILINK,
                             KNN_NNLIST,
                             NUM_THREADS,
                             USE_SINGULAR_VALUES);
}
