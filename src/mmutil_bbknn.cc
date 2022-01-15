#include "mmutil_bbknn.hh"

SpMat
build_bbknn(const svd_out_t &svd,
            const std::vector<std::vector<Index>> &batch_index_set,
            const std::size_t knn,
            const std::size_t KNN_BILINK = 10,
            const std::size_t KNN_NNLIST = 10,
            const std::size_t NUM_THREADS = 1)
{

    ////////////////////////////////
    // adjust matching parameters //
    ////////////////////////////////

    const std::size_t param_rank = svd.U.cols();
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

    /** Take spectral data for a particular treatment group "k"
     * @param k type index
     */
    auto build_spectral_data = [&](const Index k) -> Mat {
        const std::vector<Index> &col_k = batch_index_set[k];
        const Index Nk = col_k.size();

        Mat ret(param_rank, Nk);
        ret.setZero();

        for (Index j = 0; j < Nk; ++j) {
            const Index r = col_k[j];
            ret.col(j) = svd.V.row(r).transpose();
        }
        return ret;
    };

    /////////////////////////////////////////
    // construct dictionary for each batch //
    /////////////////////////////////////////

    std::vector<std::shared_ptr<hnswlib::InnerProductSpace>> vs_vec;
    std::vector<std::shared_ptr<KnnAlg>> knn_lookup_vec;

    const Index Nsample = svd.V.rows();
    const Index Nbatch = batch_index_set.size();

    Mat V(param_rank, Nsample);

    for (Index bb = 0; bb < Nbatch; ++bb) {
        const Index n_tot = batch_index_set[bb].size();

        using vs_type = hnswlib::InnerProductSpace;

        vs_vec.push_back(std::make_shared<vs_type>(param_rank));

        vs_type &VS = *vs_vec[vs_vec.size() - 1].get();
        knn_lookup_vec.push_back(
            std::make_shared<KnnAlg>(&VS, n_tot, param_bilink, param_nnlist));
    }

    {
        // progress_bar_t<Index> prog(Nsample, 1e2);

        for (Index bb = 0; bb < Nbatch; ++bb) {
            const Index n_tot = batch_index_set[bb].size();
            KnnAlg &alg = *knn_lookup_vec[bb].get();

            const auto &bset = batch_index_set.at(bb);

            Mat dat = build_spectral_data(bb);
            normalize_columns(dat);   // cosine distance
            float *mass = dat.data(); // adding data points

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
            for (Index i = 0; i < n_tot; ++i) {
                alg.addPoint((void *)(mass + param_rank * i), i);
                const Index j = bset.at(i);
                V.col(j) = dat.col(i);
                // prog.update();
                // prog(Rcpp::Rcerr);
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
        // progress_bar_t<Index> prog(Nsample, 1e2);

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
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
            // prog.update();
            // prog(std::cerr);
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

        // progress_bar_t<Index> prog(B.outerSize(), 1e2);

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
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

            // prog.update();
            // prog(std::cerr);
        }
    }

    TLOG("Adjusted kNN weights");

    return build_eigen_sparse(knn_index, Nsample, Nsample);
}
