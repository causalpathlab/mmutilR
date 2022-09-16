#include "rcpp_mmutil_network.hh"

//' Clustering columns of the network mtx file (feature incidence matrix)
//'
//' @param mtx_file data file (feature x edge)
//' @param row_file row file (feature x 1)
//' @param col_file col file (sample x 1)
//' @param output a file header for result/temporary files
//' @param nnz_cutoff Only consider edge with NNZ >= nnz_cutoff (default: 1)
//'
// [[Rcpp::export]]
Rcpp::List
rcpp_mmutil_network_edge_cluster(const std::string mtx_file,
                                 const std::string row_file,
                                 const std::string col_file,
                                 const std::string output,
                                 const std::size_t num_clust = 10,
                                 const std::size_t num_gibbs = 100,
                                 const std::size_t num_burnin = 100,
                                 const std::size_t nnz_cutoff = 1,
                                 const float A0 = 1.,
                                 const float B0 = 1.,
                                 const float Dir0 = 1.,
                                 const bool verbose = true)
{

    ASSERT_RETL(file_exists(mtx_file), "missing the MTX file");
    ASSERT_RETL(file_exists(row_file), "missing the ROW file");
    ASSERT_RETL(file_exists(col_file), "missing the COL file");

    mmutil::index::mm_info_reader_t org_info;
    CHK_RETL(mmutil::bgzf::peek_bgzf_header(mtx_file, org_info));

    // squeeze out meaningless columns from the incidence matrices
    const std::string data_ = output + "-squeeze";
    const std::string squeeze_mtx_file = data_ + ".mtx.gz";
    const std::string squeeze_row_file = data_ + ".rows.gz";
    const std::string squeeze_col_file = data_ + ".cols.gz";

    auto rm = [](const std::string f) {
        if (file_exists(f)) {
            remove_file(f);
        }
    };

    rm(squeeze_mtx_file);
    rm(squeeze_mtx_file + ".index");
    rm(squeeze_row_file);
    rm(squeeze_col_file);

    CHK_RETL(filter_col_by_nnz(nnz_cutoff, mtx_file, col_file, data_));
    copy_file(row_file, squeeze_row_file);

    mmutil::index::mm_info_reader_t info;
    CHK_RETL(mmutil::bgzf::peek_bgzf_header(squeeze_mtx_file, info));

    TLOG("Squeeze: " << org_info.max_col << " ==> " << info.max_col
                     << " columns");

    const Index N = info.max_col;
    const Index D = info.max_row;
    const Index K = std::min((Index)num_clust, N);

    const std::string idx_file = squeeze_mtx_file + ".index";
    rm(idx_file);

    CHK_RETL(mmutil::index::build_mmutil_index(squeeze_mtx_file, idx_file));

    std::vector<Index> idx_tab;
    CHK_RETL(mmutil::index::read_mmutil_index(idx_file, idx_tab));

    TLOG("Read the index file: " << idx_file);

    /////////////////////////////////////
    // Finite mixture model estimation //
    /////////////////////////////////////

    using F = mmutil::cluster::poisson_component_t;

    mmutil::cluster::dim_t dim(D);
    mmutil::cluster::pseudo_count_t a0(A0);
    mmutil::cluster::pseudo_count_t b0(B0);

    TLOG("Initializing " << K << " mixture components");

    std::vector<F> components;
    components.reserve(K);
    for (Index k = 0; k < K; ++k) {
        components.emplace_back(F(dim, a0, b0));
    }

    auto read_point = [&squeeze_mtx_file, &idx_tab](const Index e) -> SpMat {
        using namespace mmutil::io;
        SpMat ret =
            read_eigen_sparse_subset_col(squeeze_mtx_file, idx_tab, { e })
                .transpose();

        return ret;
    };

    discrete_sampler_t sampler_k(K);
    std::vector<Scalar> clust_size(K);
    std::fill(clust_size.begin(), clust_size.end(), 0);

    std::vector<Index> membership =
        mmutil::cluster::random_membership(mmutil::cluster::num_clust_t(K),
                                           mmutil::cluster::num_sample_t(N));

    for (Index j = 0; j < N; ++j) {
        const Index k = membership[j];
        membership[j] = k;
        clust_size[k]++;
        const SpMat xj = read_point(j);
        components[k].add_point(xj);
    }

    TLOG("Randomly initialized...");

    const Scalar maxHist = 20;

    if (verbose) {
        mmutil::cluster::print_histogram(clust_size, Rcpp::Rcerr, maxHist);
    }

    Vec mass(K);
    Scalar dirichlet0 = Dir0 / ((Scalar)K);

    Mat _parameter(D, K);
    running_stat_t<Mat> parameter_stat(D, K);

    TLOG("Start Gibbs sampling...");

    auto update_membership = [&](const Index j, auto &update_fun) {
        const Index k_prev = membership[j];
        SpMat xj = read_point(j);
        clust_size[k_prev]--;
        components[k_prev].remove_point(xj);

        mass.setZero();
        for (Index k = 0; k < K; ++k) {
            mass(k) += fasterlog(dirichlet0 + clust_size[k]);
            mass(k) += components.at(k).log_predictive(xj);
        }

        const Index k_new = update_fun(mass);
        membership[j] = k_new;
        clust_size[k_new]++;
        components[k_new].add_point(xj);
    };

    auto update_gibbs = [&sampler_k](const Vec &mass) -> Index {
        return sampler_k(mass);
    };

    auto update_argmax = [](const Vec &mass) -> Index {
        Index ret;
        mass.maxCoeff(&ret);
        return ret;
    };

    std::vector<Index> _n(N);
    std::iota(_n.begin(), _n.end(), 0);
    std::random_device rd;
    std::mt19937 rng(rd());

    for (Index iter = 0; iter < (num_gibbs + num_burnin); ++iter) {

        std::shuffle(_n.begin(), _n.end(), rng);

        for (Index _j = 0; _j < N; ++_j) {
            const Index j = _n[_j];
            update_membership(j, update_gibbs);
        }

        if (verbose) {
            mmutil::cluster::print_histogram(clust_size, Rcpp::Rcerr, maxHist);
        }

        TLOG("Gibbs [" << std::setw(10) << iter << "]");

        if (iter > num_burnin) {
            for (Index k = 0; k < K; ++k) {
                _parameter.col(k) = components[k].MLE();
            }
            parameter_stat(_parameter);
        }
    }

    TLOG("Freeze membership by taking greedy argmax steps");

    for (Index _j = 0; _j < N; ++_j) {
        const Index j = _n[_j];
        update_membership(j, update_argmax);
    }

    for (Index k = 0; k < K; ++k) {
        _parameter.col(k) = components[k].MLE();
    }

    TLOG("Done clustering");

    Rcpp::List param_out =
        Rcpp::List::create(Rcpp::_["mean"] = parameter_stat.mean(),
                           Rcpp::_["sd"] = parameter_stat.var().cwiseSqrt(),
                           Rcpp::_["argmax"] = _parameter);

    Rcpp::List data_out = Rcpp::List::create(Rcpp::_["mtx"] = squeeze_mtx_file,
                                             Rcpp::_["row"] = squeeze_row_file,
                                             Rcpp::_["col"] = squeeze_col_file);

    // convert zero-based to one-based
    auto add_one = [](const Index x) -> Index { return x + 1; };

    std::transform(membership.begin(),
                   membership.end(),
                   membership.begin(),
                   add_one);

    return Rcpp::List::create(Rcpp::_["membership"] = membership,
                              Rcpp::_["data"] = data_out,
                              Rcpp::_["param"] = param_out);
}

//' Construct a kNN cell-cell interaction network and identify gene topics
//'
//' @param mtx_file data file (feature x n)
//' @param knn kNN parameter
//' @param output a file header for resulting files
//' @param CUTOFF expression present/absent call cutoff (default: 1e-2)
//' @param WEIGHTED keep weights for edges (default: FALSE)
//' @param MAXW maximum weight (default: 1)
//' @param r_batches batch info (default: NULL)
//' @param r_U SVD for kNN network construction (X = UDV')
//' @param r_D SVD for kNN network construction (X = UDV')
//' @param r_V SVD for kNN network construction (X = UDV')
//' @param RANK SVD rank
//' @param TAKE_LN take log(1 + x) trans or not
//' @param TAU regularization parameter (default = 1)
//' @param COL_NORM column normalization for SVD
//' @param EM_ITER EM iteration for factorization (default: 10)
//' @param EM_TOL EM convergence (default: 1e-4)
//' @param LU_ITER LU iteration
//' @param KNN_BILINK num. of bidirectional links (default: 10)
//' @param KNN_NNLIST num. of nearest neighbor lists (default: 10)
//' @param row_weight_file row-wise weight file
//' @param NUM_THREADS number of threads for multi-core processing
//'
//' @return feature.incidence, sample.incidence, edges, adjacency matrix files
//'
// [[Rcpp::export]]
Rcpp::List
rcpp_mmutil_network_topic_data(
    const std::string mtx_file,
    const std::size_t knn,
    const std::string output,
    const float CUTOFF = 1e-2,
    const bool WEIGHTED = false,
    const float MAXW = 1,
    const std::string col_file = "",
    const std::string row_file = "",
    Rcpp::Nullable<const Rcpp::StringVector> r_batches = R_NilValue,
    Rcpp::Nullable<const Rcpp::NumericMatrix> r_U = R_NilValue,
    Rcpp::Nullable<const Rcpp::NumericMatrix> r_D = R_NilValue,
    Rcpp::Nullable<const Rcpp::NumericMatrix> r_V = R_NilValue,
    const std::size_t RANK = 0,
    const bool TAKE_LN = true,
    const double TAU = 1.,
    const double COL_NORM = 1e4,
    const std::size_t EM_ITER = 0,
    const double EM_TOL = 1e-4,
    const std::size_t KNN_BILINK = 10,
    const std::size_t KNN_NNLIST = 10,
    const std::size_t LU_ITER = 5,
    const std::string row_weight_file = "",
    const std::size_t NUM_THREADS = 1)
{

    mmutil::index::mm_info_reader_t info;
    CHK_RETL(mmutil::bgzf::peek_bgzf_header(mtx_file, info));

    TLOG("info: " << info.max_row << " x " << info.max_col
                  << " (NNZ=" << info.max_elem << ")");

    const Index D = info.max_row;
    const Index Nsample = info.max_col;

    std::vector<std::string> rows;
    if (file_exists(row_file)) {
        read_vector_file(row_file, rows);
        ASSERT_RETL(rows.size() == D,
                    "The sample size does not match with the row name file.");
    } else {
        for (Index j = 0; j < D; ++j)
            rows.push_back(std::to_string(j + 1));
    }

    std::vector<std::string> cols;
    if (file_exists(col_file)) {
        read_vector_file(col_file, cols);
        ASSERT_RETL(
            cols.size() == Nsample,
            "The sample size does not match with the column name file.");
    } else {
        for (Index j = 0; j < Nsample; ++j)
            cols.push_back(std::to_string(j + 1));
    }

    const std::string idx_file = mtx_file + ".index";

    if (!file_exists(idx_file)) // if needed
        CHK_RETL(mmutil::index::build_mmutil_index(mtx_file, idx_file));

    /////////////////////////////////////////
    // Step 1. build a weighted kNN matrix //
    /////////////////////////////////////////

    std::vector<std::string> batch_membership;
    if (r_batches.isNotNull()) {
        batch_membership = copy(Rcpp::StringVector(r_batches));
        ASSERT_RETL(
            batch_membership.size() == Nsample,
            "This batch membership vector mismatches with the mtx data");
    } else {
        batch_membership.resize(Nsample);
        std::fill(batch_membership.begin(), batch_membership.end(), "0");
    }

    std::vector<std::string> batch_id_name;
    std::vector<Index> batch; // map: col -> batch index

    std::tie(batch, batch_id_name, std::ignore) =
        make_indexed_vector<std::string, Index>(batch_membership);

    auto batch_index_set = make_index_vec_vec(batch);

    ASSERT_RETL(batch.size() >= Nsample,
                "Need batch membership for each column");
    const Index Nbatch = batch_id_name.size();
    TLOG("Identified " << Nbatch << " batches");

    svd_out_t svd;

    if (r_U.isNotNull() && r_U.isNotNull() && r_U.isNotNull()) {

        svd.U = Rcpp::as<Mat>(Rcpp::NumericMatrix(r_U));
        svd.D = Rcpp::as<Mat>(Rcpp::NumericMatrix(r_D));
        svd.V = Rcpp::as<Mat>(Rcpp::NumericMatrix(r_V));

        ASSERT_RETL(svd.U.rows() == D, "# rows of U != " << D);
        ASSERT_RETL(svd.V.rows() == Nsample, "# rows of V != " << Nsample);
        ASSERT_RETL(svd.D.rows() == svd.U.cols(),
                    "SVD U and D are not compatible");
        ASSERT_RETL(svd.D.rows() == svd.V.cols(),
                    "SVD V and D are not compatible");

    } else {
        ///////////////////////////////////
        // weights for the rows/features //
        ///////////////////////////////////

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
        if (EM_ITER > 0) {
            svd = take_svd_online_em(mtx_file, idx_file, ww, options);
        } else {
            svd = take_svd_online(mtx_file, idx_file, ww, options);
        }
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

    /////////////////////////////////////
    // symmetrize the adjacency matrix //
    /////////////////////////////////////

    SpMat Wt = W.transpose();
    SpMat Wsym = W * .5 + Wt * .5;

    TLOG("A weighted adjacency matrix W");

    /////////////////////////////////////////////
    // Step 2. build a sparse incidence matrix //
    /////////////////////////////////////////////

    const std::string out_feat_inc = output + "_feat_inc.mtx.gz";
    const std::string out_pair_names = output + ".pairs.gz";
    const std::string out_samp_inc = output + "_samp_inc.mtx.gz";
    const std::string out_samp_adj = output + "_samp_adj.mtx.gz";

    auto rm_mtx = [](const std::string mtx) {
        if (file_exists(mtx)) {
            WLOG("Removing the existing mtx file: " << mtx);
            remove_file(mtx);
        }
        if (file_exists(mtx + ".index")) {
            WLOG("Removing the existing index file: " << mtx + ".index");
            remove_file(mtx + ".index");
        }
    };

    auto rm = [](const std::string ff) {
        if (file_exists(ff))
            remove_file(ff);
    };

    rm_mtx(out_feat_inc);
    rm_mtx(out_samp_inc);
    rm_mtx(out_samp_adj);

    //////////////////////////////////////
    // feature (row/gene) x edge matrix //
    //////////////////////////////////////

    {
        using writer_t = feature_incidence_writer_t<obgzf_stream>;
        const std::string temp_file = out_feat_inc + "_temp";
        rm(temp_file);
        writer_t writer(mtx_file, temp_file, CUTOFF, WEIGHTED, MAXW);
        visit_sparse_matrix(W, writer);

        obgzf_stream ofs(out_feat_inc.c_str(), std::ios::out);

        ofs << "%%MatrixMarket matrix coordinate integer general" << std::endl;
        ofs << writer.max_row() << " " << writer.max_col() << " "
            << writer.max_elem() << std::endl;

        std::string line;

        ibgzf_stream ifs(temp_file.c_str(), std::ios::in);
        while (std::getline(ifs, line)) {
            ofs << line << std::endl;
        }

        ofs.close();
        ifs.close();
        remove_file(temp_file);
        TLOG("Wrote feature incidence matrix: " << out_feat_inc);
    }

    {
        using writer_t = sample_pair_writer_t<obgzf_stream>;
        writer_t writer(out_pair_names, cols);
        visit_sparse_matrix(W, writer);
        TLOG("Wrote pair names: " << out_pair_names);
    }

    ///////////////////////////////////
    // vertex (sample) x edge matrix //
    ///////////////////////////////////

    {
        using writer_t = sample_incidence_writer_t<obgzf_stream>;
        writer_t writer(out_samp_inc);
        visit_sparse_matrix(Wsym, writer);
        TLOG("Wrote sample incidence matrix: " << out_samp_inc);
    }

    {
        using writer_t = sample_adjacency_writer_t<obgzf_stream>;
        writer_t writer(out_samp_adj);
        visit_sparse_matrix(Wsym, writer);
        TLOG("Wrote sample adjacency matrix: " << out_samp_adj);
    }

    write_vector_file(output + ".features.gz", rows);
    write_vector_file(output + ".samples.gz", cols);

    TLOG("Done");
    return Rcpp::List::create(Rcpp::_["feature.incidence"] = out_feat_inc,
                              Rcpp::_["pairs"] = output + ".pairs.gz",
                              Rcpp::_["sample.incidence"] = out_samp_inc,
                              Rcpp::_["sample.adjacency"] = out_samp_adj,
                              Rcpp::_["features"] = output + ".features.gz",
                              Rcpp::_["samples"] = output + ".samples.gz");
}
