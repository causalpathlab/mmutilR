#include "mmutil.hh"
#include "mmutil_spectral.hh"
#include "mmutil_match.hh"
#include "mmutil_util.hh"
#include "mmutil_index.hh"
#include "progress.hh"

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
                const std::size_t EM_ITER = 10,
                const double EM_TOL = 1e-4,
                const std::size_t KNN_BILINK = 10,
                const std::size_t KNN_NNLIST = 10,
                const std::size_t LU_ITER = 5,
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

    ////////////////////////////////////////
    // Indexing all the columns if needed //
    ////////////////////////////////////////

    const std::string idx_file = mtx_file + ".index";

    std::vector<Index> mtx_idx_tab;

    if (!file_exists(idx_file)) // if needed
        CHECK(mmutil::index::build_mmutil_index(mtx_file, idx_file));

    CHECK(mmutil::index::read_mmutil_index(idx_file, mtx_idx_tab));

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
        ASSERT(weights.rows() == D, "Found invalid weight vector");
        ww = weights;
    }

    svd_out_t svd = take_svd_online_em(mtx_file, ww, options);

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
//' @param KNN_BILINK # of bidirectional links (default: 10)
//' @param KNN_NNLIST # nearest neighbor lists (default: 10)
//' @param row_weight_file row-wise weight file
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
                      const std::size_t EM_ITER = 10,
                      const double EM_TOL = 1e-4,
                      const std::size_t KNN_BILINK = 10,
                      const std::size_t KNN_NNLIST = 10,
                      const std::size_t LU_ITER = 5,
                      const std::string row_weight_file = "")
{

    ////////////////////////////////////////
    // Indexing all the columns if needed //
    ////////////////////////////////////////

    const std::string idx_file = mtx_file + ".index";

    std::vector<Index> mtx_idx_tab;

    if (!file_exists(idx_file)) // if needed
        CHECK(mmutil::index::build_mmutil_index(mtx_file, idx_file));

    CHECK(mmutil::index::read_mmutil_index(idx_file, mtx_idx_tab));

    mmutil::index::mm_info_reader_t info;
    CHECK(mmutil::bgzf::peek_bgzf_header(mtx_file, info));
    const Index D = info.max_row;
    const Index Nsample = info.max_col;

    TLOG("info: " << info.max_row << ", " << info.max_col << " --> "
                  << info.max_elem);

    //////////////////////
    // batch membership //
    //////////////////////

    std::vector<std::string> batch_membership(r_batches.begin(),
                                              r_batches.end());

    ASSERT(batch_membership.size() == Nsample,
           "This batch membership vector mismatches with the mtx data");

    std::vector<std::string> batch_id_name;
    std::vector<Index> batch; // map: col -> batch index

    std::tie(batch, batch_id_name, std::ignore) =
        make_indexed_vector<std::string, Index>(batch_membership);

    auto batch_index_set = make_index_vec_vec(batch);

    ASSERT(batch.size() >= Nsample, "Need batch membership for each column");
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
        ASSERT(weights.rows() == D, "Found invalid weight vector");
        ww = weights;
    }

    Mat proj;

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

    ////////////////////////////////
    // Learn latent embedding ... //
    ////////////////////////////////

    TLOG("Training SVD for spectral matching ...");
    svd_out_t svd = take_svd_online_em(mtx_file, idx_file, ww, options);
    proj.resize(svd.U.rows(), svd.U.cols());
    proj = svd.U * svd.D.cwiseInverse().asDiagonal(); // feature x rank
    TLOG("Found projection: " << proj.rows() << " x " << proj.cols());

    ////////////////////////////////
    // adjust matching parameters //
    ////////////////////////////////

    const std::size_t param_rank = proj.cols();
    const std::size_t knn_min = 1;
    const std::size_t knn_max = 1000;
    const std::size_t param_knn = std::max(knn_min, std::min(knn, knn_max));

    std::size_t param_bilink = KNN_BILINK;
    std::size_t param_nnlist = KNN_NNLIST;

    if (param_bilink >= param_rank) {
        WLOG("Shrink M value: " << param_bilink << " vs. " << param_rank);
        param_bilink = param_rank - 1;
    }

    if (param_bilink < 2) {
        WLOG("too small M value");
        param_bilink = 2;
    }

    if (param_nnlist <= param_knn) {
        WLOG("too small N value");
        param_nnlist = param_knn + 1;
    }

    /** Take a block of Y matrix
     * @param subcol
     */
    auto read_y_block = [&](std::vector<Index> &subcol) -> Mat {
        using namespace mmutil::io;
        return Mat(read_eigen_sparse_subset_col(mtx_file, mtx_idx_tab, subcol));
    };

    /** Take spectral data for a particular treatment group "k"
     * @param k type index
     */
    auto build_spectral_data = [&](const Index k) -> Mat {
        std::vector<Index> &col_k = batch_index_set[k];
        const Index Nk = col_k.size();
        const Index block_size = options.block_size;
        const Index rank = proj.cols();

        Mat ret(param_rank, Nk);
        ret.setZero();

        Index r = 0;
        for (Index lb = 0; lb < Nk; lb += block_size) {
            const Index ub = std::min(Nk, block_size + lb);

            std::vector<Index> subcol_k(ub - lb);
            std::copy(col_k.begin() + lb, col_k.begin() + ub, subcol_k.begin());

            Mat x0 = read_y_block(subcol_k);

            Mat xx = make_normalized_laplacian(x0,
                                               ww,
                                               options.tau,
                                               options.col_norm,
                                               options.log_scale);

            Mat vv = proj.transpose() * xx; // rank x block_size

            for (Index j = 0; j < vv.cols(); ++j) {
                ret.col(r) = vv.col(j);
                ++r;
            }
        }
        return ret;
    };

    /////////////////////////////////////////
    // construct dictionary for each batch //
    /////////////////////////////////////////

    std::vector<std::shared_ptr<hnswlib::InnerProductSpace>> vs_vec;
    std::vector<std::shared_ptr<KnnAlg>> knn_lookup_vec;
    Mat V(param_rank, Nsample);
    Mat Vorg(param_rank, Nsample);

    for (Index bb = 0; bb < Nbatch; ++bb) {
        const Index n_tot = batch_index_set[bb].size();

        using vs_type = hnswlib::InnerProductSpace;

        vs_vec.push_back(std::make_shared<vs_type>(param_rank));

        vs_type &VS = *vs_vec[vs_vec.size() - 1].get();
        knn_lookup_vec.push_back(
            std::make_shared<KnnAlg>(&VS, n_tot, param_bilink, param_nnlist));
    }

    {
        progress_bar_t<Index> prog(Nsample, 1e2);

        for (Index bb = 0; bb < Nbatch; ++bb) {
            const Index n_tot = batch_index_set[bb].size();
            KnnAlg &alg = *knn_lookup_vec[bb].get();

            const auto &bset = batch_index_set.at(bb);

            Mat dat = build_spectral_data(bb);  // Take the
            for (Index i = 0; i < n_tot; ++i) { // original
                const Index j = bset.at(i);     //
                Vorg.col(j) = dat.col(i);       // SVD
            }                                   // results

            normalize_columns(dat);   // cosine distance
            float *mass = dat.data(); // adding data points
            for (Index i = 0; i < n_tot; ++i) {
                alg.addPoint((void *)(mass + param_rank * i), i);
                const Index j = bset.at(i);
                V.col(j) = dat.col(i);
                prog.update();
                prog(Rcpp::Rcerr);
            }
        }
    }

    TLOG("Built the dictionaries for fast look-ups");

    ///////////////////////////////////////////////////
    // step 1: build mutual kNN graph across batches //
    ///////////////////////////////////////////////////
    std::vector<std::tuple<Index, Index, Scalar>> backbone;

    {
        float *mass = V.data();
        progress_bar_t<Index> prog(Nsample, 1e2);

        for (Index j = 0; j < Nsample; ++j) {

            for (Index bb = 0; bb < Nbatch; ++bb) {
                KnnAlg &alg = *knn_lookup_vec[bb].get();
                const std::size_t nn_b = batch_index_set.at(bb).size();
                std::size_t nquery = (std::min(param_knn, nn_b) / Nbatch);
                if (nquery < 1)
                    nquery = 1;

                auto pq =
                    alg.searchKnn((void *)(mass + param_rank * j), nquery);

                while (!pq.empty()) {
                    float d = 0;
                    std::size_t k;
                    std::tie(d, k) = pq.top();
                    Index i = batch_index_set.at(bb).at(k);
                    if (i != j) {
                        backbone.emplace_back(j, i, 1.0);
                    }
                    pq.pop();
                }
            }
            prog.update();
            prog(std::cerr);
        }

        keep_reciprocal_knn(backbone);
    }

    TLOG("Constructed kNN graph backbone");

    ///////////////////////////////////
    // step2: calibrate edge weights //
    ///////////////////////////////////
    std::vector<std::tuple<Index, Index, Scalar>> knn_index;
    knn_index.clear();

    {
        const SpMat B = build_eigen_sparse(backbone, Nsample, Nsample);

        std::vector<Scalar> dist_j(param_knn);
        std::vector<Scalar> weights_j(param_knn);
        std::vector<Index> neigh_j(param_knn);

        progress_bar_t<Index> prog(B.outerSize(), 1e2);

        for (Index j = 0; j < B.outerSize(); ++j) {

            Index deg_j = 0;
            for (SpMat::InnerIterator it(B, j); it; ++it) {
                Index k = it.col();
                dist_j[deg_j] = V.col(k).cwiseProduct(V.col(j)).sum();
                neigh_j[deg_j] = k;
                ++deg_j;
                if (deg_j >= param_knn)
                    break;
            }

            normalize_weights(deg_j, dist_j, weights_j);

            for (Index i = 0; i < deg_j; ++i) {
                const Index k = neigh_j[i];
                const Scalar w = weights_j[i];

                knn_index.emplace_back(j, k, w);
            }

            prog.update();
            prog(std::cerr);
        }
    }

    TLOG("Adjusted kNN weights");

    SpMat W = build_eigen_sparse(knn_index, Nsample, Nsample);

    TLOG("A weighted adjacency matrix W");

    ////////////////////////////////////
    // step3: adjusting spectral data //
    ////////////////////////////////////

    Mat Vadj = Vorg;
    Mat Delta_feature(D, Nbatch);
    Mat Delta_factor(param_rank, Nbatch);
    Delta_feature.setZero();
    Delta_factor.setZero();

    for (Index aa = 1; aa < Nbatch; ++aa) {
        const auto &batch_a = batch_index_set.at(aa);
        const Index nn_a = batch_a.size();

        Mat delta_a(V.rows(), 1);
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

    return Rcpp::List::create(Rcpp::_["factors.adjusted"] = Vadj,
                              Rcpp::_["U"] = svd.U,
                              Rcpp::_["D"] = svd.D,
                              Rcpp::_["V"] = Vorg,
                              Rcpp::_["delta.samples"] = Delta_factor,
                              Rcpp::_["delta.features"] = Delta_feature);
}
