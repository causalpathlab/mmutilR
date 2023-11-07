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
