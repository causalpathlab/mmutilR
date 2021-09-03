#include "mmutil.hh"
#include "mmutil_io.hh"
#include "mmutil_stat.hh"
#include "mmutil_index.hh"
#include "mmutil_bgzf_util.hh"
#include "mmutil_spectral.hh"
#include "mmutil_match.hh"
#include "mmutil_velocity.hh"

// [[Rcpp::plugins(openmp)]]

#include "io.hh"
#include "std_util.hh"

//' Compute RNA velocity comparing the spliced and unspliced
//' at the pseudo-bulk level (individual and cell type)
//'
//' @param spliced_mtx_file spliced data file
//' @param unspliced_mtx_file unspliced data file
//' @param spliced_col_file column file for the spliced mtx
//' @param unspliced_col_file column file for the unspliced
//' @param row_file row file (shared)
//' @param col_file column file (shared)
//' @param r_cols cell (col) names
//' @param r_indv membership for the cells (\code{r_cols})
//' @param r_annot label annotation for the (\code{r_cols})
//' @param r_lab_name label names (default: everything in \code{r_annot})
//' @param a0 hyperparameter for gamma(a0, b0) (default: 1)
//' @param b0 hyperparameter for gamma(a0, b0) (default: 1)
//' @param MAX_ITER maximum iteration for the delta estimation
//' @param TOL tolerance level for convergence test
//' @param NUM_THREADS number of threads (useful for many individuals)
//'
//' @return a list of inference results
//'
//' @examples
//'
//' options(stringsAsFactors = FALSE)
//' set.seed(1)
//' nn <- 3000
//' rr <- rgamma(nn, 6.25, 6.25) # 1000 cells
//' uu <- matrix(rgamma(100 * 3, 1, 1), 100, 3)
//' dd <- matrix(rgamma(100 * 3, 1, 1/10), 100, 3)
//' ss <- uu / (dd + 1e-2)
//' ind <- sample(3, nn, replace=TRUE)
//'
//' spliced <- mmutilR::rcpp_mmutil_simulate_poisson(ss, rr,
//'                                                  "sim_test_raw_spliced",
//'                                                  r_indv = ind)
//'
//' unspliced <- mmutilR::rcpp_mmutil_simulate_poisson(uu, rr,
//'                                                    "sim_test_raw_unspliced",
//'                                                    r_indv = ind)
//'
//' .col <- sort(intersect(read.table(spliced$col)$V1,
//'                        read.table(unspliced$col)$V1))
//'
//' spliced <- mmutilR::rcpp_mmutil_copy_selected_columns(
//'                         spliced$mtx,
//'                         spliced$row,
//'                         spliced$col,
//'                         .col,
//'                         "sim_test_spliced")
//'
//' unspliced <- mmutilR::rcpp_mmutil_copy_selected_columns(
//'                         unspliced$mtx,
//'                         unspliced$row,
//'                         unspliced$col,
//'                         .col,
//'                         "sim_test_unspliced")
//'
//' .out <- mmutilR::rcpp_mmutil_aggregate_velocity(
//'                      spliced$mtx,
//'                      unspliced$mtx,
//'                      spliced$row,
//'                      spliced$col,
//'                      r_col = .col,
//'                      r_indv = ind[.col],
//'                      a0 = 1, b0 = 1)
//'
//' .agg.u <- mmutilR::rcpp_mmutil_aggregate(
//'                        unspliced$mtx,
//'                        unspliced$row,
//'                        unspliced$col,
//'                        r_col = .col,
//'                        r_indv = ind[.col],
//'                        a0 = 1, b0 = 1)
//'
//' .agg.s <- mmutilR::rcpp_mmutil_aggregate(
//'                        spliced$mtx,
//'                        spliced$row,
//'                        spliced$col,
//'                        r_col = .col,
//'                        r_indv = ind[.col],
//'                        a0 = 1, b0 = 1)
//'
//' par(mfrow=c(1, ncol(.out$delta)))
//' for(k in 1:ncol(.out$delta)){
//'     plot(.agg.u$mu[,k]/.agg.s$mu[,k],
//'          .out$delta[,k],
//'          log = "xy",
//'          pch = 1,
//'          ylab = "predicted",
//'          xlab = "true")
//'     abline(a=0, b=1, col=3)
//' }
//'
//'
//' ## clean up temp directory
//' unlink(list.files(pattern = "sim_test"))
//'
// [[Rcpp::export]]
Rcpp::List
rcpp_mmutil_aggregate_velocity(
    const std::string spliced_mtx_file,
    const std::string unspliced_mtx_file,
    const std::string row_file,
    const std::string col_file,
    Rcpp::Nullable<Rcpp::StringVector> r_cols = R_NilValue,
    Rcpp::Nullable<Rcpp::StringVector> r_indv = R_NilValue,
    Rcpp::Nullable<Rcpp::StringVector> r_annot = R_NilValue,
    Rcpp::Nullable<Rcpp::StringVector> r_lab_name = R_NilValue,
    const float a0 = 1.0,
    const float b0 = 1.0,
    const std::size_t MAX_ITER = 100,
    const float TOL = 1e-4,
    const std::size_t NUM_THREADS = 1)
{

    CHK_RETL(mmutil::bgzf::convert_bgzip(spliced_mtx_file));
    CHK_RETL(mmutil::bgzf::convert_bgzip(unspliced_mtx_file));

    std::vector<std::string> mtx_cols;
    CHK_RETL(read_vector_file(col_file, mtx_cols));

    const Index Nsample = mtx_cols.size();

    mmutil::io::mm_info_reader_t s_info;
    CHK_RETL(mmutil::bgzf::peek_bgzf_header(spliced_mtx_file, s_info));

    ASSERT_RETL(Nsample == s_info.max_col,
                "Should have matched spliced .mtx.gz");

    mmutil::io::mm_info_reader_t u_info;
    CHK_RETL(mmutil::bgzf::peek_bgzf_header(unspliced_mtx_file, u_info));

    ASSERT_RETL(Nsample == u_info.max_col,
                "Should have matched unspliced .mtx.gz");

    ASSERT_RETL(
        s_info.max_row == u_info.max_row,
        "The spliced and unspliced must have been on the same features.")

    const Index Ngene = s_info.max_row;

    std::vector<std::string> out_row_names;
    out_row_names.reserve(Ngene);
    if (file_exists(row_file)) {
        CHK_RETL(read_vector_file(row_file, out_row_names));
        ASSERT_RETL(Ngene == out_row_names.size(), "invalid row file");
    } else {
        for (Index g = 0; g < Ngene; ++g)
            out_row_names.emplace_back(std::to_string(g + 1));
    }

    //////////////////////////////
    // check column annotations //
    //////////////////////////////

    std::vector<std::string> cols;
    std::vector<std::string> indv;
    std::vector<std::string> annot;
    std::vector<std::string> lab_name;

    if (r_cols.isNotNull() && r_indv.isNotNull()) {
        copy(Rcpp::StringVector(r_cols), cols);
        copy(Rcpp::StringVector(r_indv), indv);
    } else {
        cols.reserve(mtx_cols.size());
        std::copy(std::begin(mtx_cols),
                  std::end(mtx_cols),
                  std::back_inserter(cols));
        indv.resize(mtx_cols.size());
        std::string _indv = "ind1";
        std::fill(std::begin(indv), std::end(indv), _indv);
    }

    if (r_annot.isNotNull()) {
        copy(Rcpp::StringVector(r_annot), annot);
    } else {
        const std::string _lab = "ct1";
        annot.resize(cols.size());
        std::fill(std::begin(annot), std::end(annot), _lab);
    }

    if (r_lab_name.isNotNull()) {
        copy(Rcpp::StringVector(r_lab_name), lab_name);
    } else {
        make_unique(annot, lab_name);
    }

    const Index K = lab_name.size();

    ASSERT_RETL(cols.size() == indv.size(), "|cols| != |indv|");
    ASSERT_RETL(cols.size() == annot.size(), "|cols| != |annot|");

    ////////////////////////
    // universal position //
    ////////////////////////

    auto mtx_pos = make_position_dict<std::string, Index>(mtx_cols);
    auto lab_pos = make_position_dict<std::string, Index>(lab_name);

    /////////////////////////////////////////////////
    // latent annotation and individual membership //
    /////////////////////////////////////////////////

    Mat Ctot(Nsample, K);
    Ctot.setZero();

    std::vector<std::string> indv_membership(Nsample);
    std::fill(std::begin(indv_membership), std::end(indv_membership), "?");

    TLOG("Reading latent & indv annotations");

    for (Index j = 0; j < cols.size(); ++j) {
        if (mtx_pos.count(cols[j]) > 0) {
            const Index i = mtx_pos[cols[j]];
            if (lab_pos.count(annot[j]) > 0) {
                const Index k = lab_pos[annot[j]];
                Ctot(i, k) = 1.;
            }
            indv_membership[i] = indv[j];
        }
    }

    Ctot.transposeInPlace(); // type x sample

    std::vector<std::string> indv_id_name;
    std::vector<Index> indv_map; // map: col -> indv index

    std::tie(indv_map, indv_id_name, std::ignore) =
        make_indexed_vector<std::string, Index>(indv_membership);

    auto indv_index_set = make_index_vec_vec(indv_map);

    const Index Nind = indv_id_name.size();

    TLOG("Identified " << Nind << " individuals, " << K << " labels");

    //////////////////////////////
    // Indexing all the columns //
    //////////////////////////////

    std::vector<Index> spliced_idx_tab;
    std::vector<Index> unspliced_idx_tab;

    {
        using namespace mmutil::index;
        std::string _ifile = spliced_mtx_file + ".index";

        if (!file_exists(_ifile)) // if needed
            CHK_RETL(build_mmutil_index(spliced_mtx_file, _ifile));

        CHK_RETL(read_mmutil_index(_ifile, spliced_idx_tab));
        CHK_RETL(check_index_tab(spliced_mtx_file, spliced_idx_tab));
    }

    {
        using namespace mmutil::index;
        std::string _ifile = unspliced_mtx_file + ".index";

        if (!file_exists(_ifile)) // if needed
            CHK_RETL(build_mmutil_index(unspliced_mtx_file, _ifile));

        CHK_RETL(read_mmutil_index(_ifile, unspliced_idx_tab));
        CHK_RETL(check_index_tab(unspliced_mtx_file, unspliced_idx_tab));
    }

    /////////////////////////
    // for each individual //
    /////////////////////////

    std::vector<std::string> out_col_names(K * Nind);
    std::fill(out_col_names.begin(), out_col_names.end(), "");

    Mat out_delta(Ngene, K * Nind);
    Mat out_sd_delta(Ngene, K * Nind);
    Mat out_ln_delta(Ngene, K * Nind);
    Mat out_sd_ln_delta(Ngene, K * Nind);

    out_delta.setZero();
    out_sd_delta.setZero();
    out_ln_delta.setZero();
    out_sd_ln_delta.setZero();

    Index nind_proc = 0;

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
    for (Index ii = 0; ii < Nind; ++ii) {

        using namespace mmutil::io;
        using namespace mmutil::velocity;

        auto storage_index = [&K, &ii](const Index k) { return K * ii + k; };
        const std::string indv_name = indv_id_name.at(ii);
        const std::vector<Index> &cols_i = indv_index_set.at(ii);

        ///////////////////////////////////
        // initialize UC and PhiC matrix //
        ///////////////////////////////////

        const Scalar nj = static_cast<Scalar>(cols_i.size());
        const float eps = static_cast<float>(1. / nj);

        aggregated_delta_model_t model(NGENES { Ngene },
                                       NTYPES { K },
                                       A0 { a0 },
                                       B0 { b0 },
                                       EPS { eps });

        // TODO
        // SpMat S = read_eigen_sparse_subset_col(spliced_mtx_file,
        //                                        spliced_idx_tab,
        //                                        cols_i);

        // SpMat U = read_eigen_sparse_subset_col(unspliced_mtx_file,
        //                                        unspliced_idx_tab,
        //                                        cols_i);

        data_loader_t loader(spliced_mtx_file,
                             unspliced_mtx_file,
                             spliced_idx_tab,
                             unspliced_idx_tab,
                             Ngene);

        for (auto j : cols_i) {
            loader.read(j);
            model.add_stat(loader.unspliced(), loader.spliced(), Ctot.col(j));
        }

        TLOG("Initialized " << model.nsample() << " observations");

        for (std::size_t iter = 0; iter < MAX_ITER; ++iter) {

            ///////////////////////////////
            // global step: update delta //
            ///////////////////////////////

            model.update_delta_stat();

            const Scalar diff = model.update_diff();
            if (diff < TOL) {
                TLOG("Converged: " << diff);
                break;
            }

            TLOG("[" << (iter + 1) << "/" << MAX_ITER << "] " << diff);

            ////////////////////////////
            // local step: update phi //
            ////////////////////////////

            for (auto j : cols_i) {
                loader.read(j);
                model.update_phi_stat(loader.unspliced(),
                                      loader.spliced(),
                                      Ctot.col(j));
            }

            // TLOG("Re-calibrated " << model.nsample() << " observations");
        }

        Mat _delta = model.get_delta();
        Mat _sd_delta = model.get_sd_delta();
        Mat _ln_delta = model.get_ln_delta();
        Mat _sd_ln_delta = model.get_sd_ln_delta();

        for (Index k = 0; k < K; ++k) {
            const Index s = storage_index(k);
            out_col_names[s] = indv_name + "_" + lab_name.at(k);
            out_delta.col(s) = _delta.col(k);
            out_sd_delta.col(s) = _sd_delta.col(k);
            out_ln_delta.col(s) = _ln_delta.col(k);
            out_sd_ln_delta.col(s) = _sd_ln_delta.col(k);
        }

        TLOG("Processed: " << (++nind_proc) << std::endl);
    } // for each individual

    ////////////////////
    // output results //
    ////////////////////

    TLOG("Writing down the estimated effects");

    auto named_mat = [&out_row_names, &out_col_names](const Mat &xx) {
        Rcpp::NumericMatrix x = Rcpp::wrap(xx);
        if (xx.rows() == out_row_names.size() &&
            xx.cols() == out_col_names.size()) {
            Rcpp::rownames(x) = Rcpp::wrap(out_row_names);
            Rcpp::colnames(x) = Rcpp::wrap(out_col_names);
        }
        return x;
    };

    return Rcpp::List::create(Rcpp::_["delta"] = named_mat(out_delta),
                              Rcpp::_["delta.sd"] = named_mat(out_sd_delta),
                              Rcpp::_["ln.delta"] = named_mat(out_ln_delta),
                              Rcpp::_["ln.delta.sd"] =
                                  named_mat(out_sd_ln_delta));
}
