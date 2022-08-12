#include "mmutil_cocoa_paired.hh"

Mat
paired_data_t::read_block(const Index i)
{
    ASSERT(i >= 0 && i < Nindv, "invalid i index: " << i);
    return _read_block(indv_index_set.at(i));
}

const paired_data_t::idx_vec_t &
paired_data_t::cell_indexes(const Index i) const
{
    ASSERT(i >= 0 && i < Nindv, "invalid i index: " << i);
    return indv_index_set.at(i);
}

const std::string
paired_data_t::indv_name(const Index i) const
{
    ASSERT(i < Nindv && i >= 0, "invalid index i:" << i << " vs. " << Nindv);
    return indv_id_name.at(i);
}

Mat
paired_data_t::_read_block(const idx_vec_t &cells_j)
{
    return Mat(mmutil::io::read_eigen_sparse_subset_col(mtx_file,
                                                        mtx_idx_tab,
                                                        cells_j));
}

Mat
paired_data_t::read_matched_block(const Index i, const Index j)
{
    ASSERT(i >= 0 && i < Nindv, "invalid i index: " << i);
    ASSERT(j >= 0 && j < Nindv, "invalid j index: " << j);

    const idx_vec_t &cells_i = indv_index_set.at(i);
    const idx_vec_t &cells_j = indv_index_set.at(j);
    KnnAlg &alg_j = *knn_lookup_indv[j].get();

    const Index n_i = cells_i.size();

    const std::size_t n_j = cells_j.size();
    const std::size_t nquery = std::min(knn, n_j);
    TLOG("#query per cell: " << nquery);

    float *mass = Vt.data();

    Mat y0(D, n_i);
    y0.setZero();

    const std::size_t rank = Vt.rows();
    // TLOG("Using " << rank << " x " << Vt.cols() << " data");

    for (Index ith = 0; ith < n_i; ++ith) {
        const Index _cell_i = cells_i.at(ith); //
        Index deg_i = 0;                       // # of neighbours

        idx_vec_t counterfactual_neigh;
        num_vec_t dist_neigh, weights_neigh;

        auto pq = alg_j.searchKnn((void *)(mass + rank * _cell_i), nquery);

        while (!pq.empty()) {
            float d = 0;                         // distance
            std::size_t k;                       // local index
            std::tie(d, k) = pq.top();           //
            const Index _cell_j = cells_j.at(k); // global index
            if (_cell_j != _cell_i) {
                counterfactual_neigh.emplace_back(_cell_j);
                dist_neigh.emplace_back(d);
            }
            pq.pop();
        }

        if (counterfactual_neigh.size() > 1) {

            Index deg_ = counterfactual_neigh.size();
            weights_neigh.resize(deg_);
            normalize_weights(deg_, dist_neigh, weights_neigh);
            Vec w0_ = eigen_vector(weights_neigh);
            Mat y0_ = _read_block(counterfactual_neigh);

            const Scalar denom = w0_.sum(); // must be > 0
            y0.col(ith) = y0_ * w0_ / denom;

        } else if (counterfactual_neigh.size() == 1) {
            y0.col(ith) = _read_block(counterfactual_neigh).col(0);
        }
    }

    return y0;
}

std::vector<std::tuple<Index, Index>>
paired_data_t::match_individuals()
{

    ASSERT(Nindv > 1, "only a single (or zero) individual");

    //////////////////////
    // 1. aggregate r_V //
    //////////////////////

    Mat M(Nsample, Nindv);
    M.setZero();

    for (Index ii = 0; ii < Nindv; ++ii) {
        for (auto jj : cell_indexes(ii)) {
            M(jj, ii) = 1;
        }
    }

    Mat Vind = Vt.transpose() * M;
    normalize_columns(Vind);
    const std::size_t rank = Vind.rows();

    TLOG("Aggregate Vind matrix");

    vs_type VS(rank);

    if (param_bilink >= rank) {
        param_bilink = rank - 1;
    }

    if (param_bilink < 2) {
        param_bilink = 2;
    }

    if (param_nnlist <= knn_indv) {
        param_nnlist = knn_indv + 1;
    }

    const std::size_t nquery =
        std::min(knn_indv + 1, static_cast<std::size_t>(Nindv));

    KnnAlg alg(&VS, Nindv, param_bilink, param_nnlist);
    float *mass = Vind.data();
    for (Index ii = 0; ii < Nindv; ++ii) {
        alg.addPoint((void *)(mass + rank * ii), ii);
    }

    std::vector<std::tuple<Index, Index>> indv_pairs;

    for (Index ii = 0; ii < Nindv; ++ii) {
        auto pq = alg.searchKnn((void *)(mass + rank * ii), nquery);

        while (!pq.empty()) {
            float d = 0;                // distance
            std::size_t jj;             // local index
            std::tie(d, jj) = pq.top(); //
            if (ii != jj) {
                indv_pairs.emplace_back(std::make_tuple(ii, jj));
                TLOG("matching " << ii << " against " << jj);
            }
            pq.pop();
        }
    }

    return indv_pairs;
}

Index
paired_data_t::num_individuals() const
{
    return Nindv;
}

void
paired_data_t::set_individual_info(const paired_data_t::str_vec_t &indv)
{
    Nindv = 0;
    ASSERT(cols.size() == indv.size(), "|cols| != |indv|");

    const std::string NA("?");

    std::vector<std::string> temp(Nsample);
    std::fill(std::begin(temp), std::end(temp), NA);

    auto mtx_pos = make_position_dict<std::string, Index>(mtx_cols);

    for (Index j = 0; j < cols.size(); ++j) {
        if (mtx_pos.count(cols[j]) > 0) {
            const Index i = mtx_pos[cols[j]];
            temp[i] = indv[j];
        }
    }

    for (Index j = 0; j < Nsample; ++j) {
        if (temp[j] == NA)
            WLOG("An individual was not assigned: " << mtx_cols.at(j));
    }

    std::tie(indv_map, indv_id_name, std::ignore) =
        make_indexed_vector<std::string, Index>(temp);

    indv_index_set = make_index_vec_vec(indv_map);

    Nindv = indv_index_set.size();

    TLOG("Found " << Nindv << " individuals");
}

int
paired_data_t::build_dictionary(const Rcpp::NumericMatrix r_V,
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

    TLOG("Building dictionaries for each individual ...");

    for (Index tt = 0; tt < Nindv; ++tt) {

        const Index n_tot = indv_index_set[tt].size();

        vs_vec_indv.emplace_back(std::make_shared<vs_type>(rank));

        vs_type &VS = *vs_vec_indv[tt].get();

        knn_lookup_indv.emplace_back(
            std::make_shared<KnnAlg>(&VS, n_tot, param_bilink, param_nnlist));
    }

    for (Index tt = 0; tt < Nindv; ++tt) {
        const Index n_tot = indv_index_set[tt].size(); // # cells
        KnnAlg &alg = *knn_lookup_indv[tt].get();      // lookup
        float *mass = Vt.data();                       // raw data

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
        for (Index i = 0; i < n_tot; ++i) {
            const Index cell_j = indv_index_set.at(tt).at(i);
            alg.addPoint((void *)(mass + rank * cell_j), i);
        }
        TLOG("Built the dictionary [" << (tt + 1) << " / " << Nindv << "]");
    }

    ASSERT_RET(knn_lookup_indv.size() > 0, "Failed to build look-up");
    return EXIT_SUCCESS;
}
