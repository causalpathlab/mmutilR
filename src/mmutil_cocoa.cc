#include "mmutil_cocoa.hh"

Mat
matched_data_t::read_cf_block(const idx_vec_t &cells_j, bool impute_knn)
{
    float *mass = Vt.data();
    const Index n_j = cells_j.size();
    Mat y = read_block(cells_j);
    Mat y0(D, n_j);
    y0.setZero();

    const std::size_t rank = Vt.rows();

    for (Index jth = 0; jth < n_j; ++jth) {     // For each cell j
        const Index _cell_j = cells_j.at(jth);  //
        const Index tj = trt_map.at(_cell_j);   // Trt group for this cell j
        const std::size_t n_j = cells_j.size(); // number of cells

        idx_vec_t counterfactual_neigh;
        num_vec_t dist_neigh, weights_neigh;

        ///////////////////////////////////////////////
        // Search neighbours in the other conditions //
        ///////////////////////////////////////////////

        for (Index ti = 0; ti < Ntrt; ++ti) {

            if (ti == tj) // skip the same condition
                continue; //

            const idx_vec_t &cells_i = trt_index_set.at(ti);
            KnnAlg &alg_ti = *knn_lookup_trt[ti].get();
            const std::size_t n_i = cells_i.size();
            const std::size_t nquery = std::min(knn, n_i);

            Index deg_j = 0; // # of neighbours

            auto pq = alg_ti.searchKnn((void *)(mass + rank * _cell_j), nquery);

            while (!pq.empty()) {
                float d = 0;                         // distance
                std::size_t k;                       // local index
                std::tie(d, k) = pq.top();           //
                const Index _cell_i = cells_i.at(k); // global index
                if (_cell_j != _cell_i) {
                    counterfactual_neigh.emplace_back(_cell_i);
                    dist_neigh.emplace_back(d);
                }
                pq.pop();
            }
        }

        ////////////////////////////////////////////////////////
        // Find optimal weights for counterfactual imputation //
        ////////////////////////////////////////////////////////

        if (counterfactual_neigh.size() > 1) {
            Mat yy = y.col(jth);

            if (impute_knn) {
                Index deg_ = counterfactual_neigh.size();
                weights_neigh.resize(deg_);
                normalize_weights(deg_, dist_neigh, weights_neigh);
                Vec w0_ = eigen_vector(weights_neigh);
                Mat y0_ = read_block(counterfactual_neigh);

                const Scalar denom = w0_.sum(); // must be > 0
                y0.col(jth) = y0_ * w0_ / denom;

            } else {

                Mat xx =
                    read_block(counterfactual_neigh).unaryExpr(glm_feature);
                const bool glm_intercept = true;
                y0.col(jth) = predict_poisson_glm(xx,
                                                  yy,
                                                  glm_iter,
                                                  glm_reg,
                                                  glm_intercept,
                                                  glm_std,
                                                  glm_sd);
            }

        } else if (counterfactual_neigh.size() == 1) {
            y0.col(jth) = read_block(counterfactual_neigh).col(0);
        }
    }
    return y0;
}

Mat
matched_data_t::read_block(const idx_vec_t &cells_j)
{
    return Mat(mmutil::io::read_eigen_sparse_subset_col(mtx_file,
                                                        mtx_idx_tab,
                                                        cells_j));
}

int
matched_data_t::build_dictionary(const Rcpp::NumericMatrix r_V,
                                 const std::size_t NUM_THREADS)
{
    Vt = Rcpp::as<Mat>(r_V);
    Vt.transposeInPlace();
    normalize_columns(Vt);

    ASSERT_RET(Vt.rows() > 0 && Vt.cols() > 0, "Empty Vt");
    ASSERT_RET(Vt.cols() == Nsample, "#rows(V) != Nsample");

    const std::size_t rank = Vt.rows();

    if (param_bilink >= rank) {
        // WLOG("Shrink M value: " << param_bilink << " vs. " << rank);
        param_bilink = rank - 1;
    }

    if (param_bilink < 2) {
        // WLOG("too small M value");
        param_bilink = 2;
    }

    if (param_nnlist <= knn) {
        // WLOG("too small N value");
        param_nnlist = knn + 1;
    }

    TLOG("Building dictionaries for each treatment ...");

    for (Index tt = 0; tt < Ntrt; ++tt) {

        const Index n_tot = trt_index_set[tt].size();

        vs_vec_trt.emplace_back(std::make_shared<vs_type>(rank));

        vs_type &VS = *vs_vec_trt[tt].get();

        knn_lookup_trt.emplace_back(
            std::make_shared<KnnAlg>(&VS, n_tot, param_bilink, param_nnlist));
    }

    for (Index tt = 0; tt < Ntrt; ++tt) {
        const Index n_tot = trt_index_set[tt].size(); // # cells
        KnnAlg &alg = *knn_lookup_trt[tt].get();      // lookup
        float *mass = Vt.data();                      // raw data

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
        for (Index i = 0; i < n_tot; ++i) {
            const Index cell_j = trt_index_set.at(tt).at(i);
            alg.addPoint((void *)(mass + rank * cell_j), i);
        }
        TLOG("Built the dictionary [" << (tt + 1) << " / " << Ntrt << "]");
    }

    ASSERT_RET(knn_lookup_trt.size() > 0, "Failed to build look-up");

    return EXIT_SUCCESS;
}

Index
matched_data_t::num_treatment() const
{
    return Ntrt;
}

void
matched_data_t::set_treatment_info(const matched_data_t::str_vec_t &trt)
{
    Ntrt = 0;

    ASSERT(cols.size() == trt.size(), "|cols| != |trt|");

    const std::string NA("?");

    std::vector<std::string> temp(Nsample);
    std::fill(std::begin(temp), std::end(temp), NA);

    auto mtx_pos = make_position_dict<std::string, Index>(mtx_cols);

    for (Index j = 0; j < cols.size(); ++j) {
        if (mtx_pos.count(cols[j]) > 0) {
            const Index i = mtx_pos[cols[j]];
            temp[i] = trt[j];
        }
    }

    for (Index j = 0; j < Nsample; ++j) {
        if (temp[j] == NA)
            WLOG("A treatment group was not assigned for " << mtx_cols.at(j));
    }

    std::tie(trt_map, trt_id_name, std::ignore) =
        make_indexed_vector<std::string, Index>(temp);

    trt_index_set = make_index_vec_vec(trt_map);

    Ntrt = trt_index_set.size();

    TLOG("Found " << Ntrt << " treatment groups");
}
