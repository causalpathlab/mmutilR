#include "mmutil.hh"
#include "mmutil_spectral.hh"
#include "mmutil_match.hh"
#include "mmutil_util.hh"
#include "mmutil_index.hh"
#include "progress.hh"

#ifndef MMUTIL_BBKNN_HH_
#define MMUTIL_BBKNN_HH_

template <typename Derived>
void
reweight_knn_graph(
    Eigen::MatrixBase<Derived> &VD_rs,
    const std::vector<std::tuple<Index, Index, Scalar>> &backbone,
    const std::size_t param_knn,
    std::vector<std::tuple<Index, Index, Scalar, Scalar>> &knn_index,
    const std::size_t NUM_THREADS = 1)
{
    const Index Nsample = VD_rs.cols();
    const SpMat B = build_eigen_sparse(backbone, Nsample, Nsample);

    knn_index.clear();
    knn_index.reserve(B.nonZeros());

    progress_bar_t<Index> prog(B.outerSize(), 1e2);

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
    for (Index j = 0; j < B.outerSize(); ++j) {

        std::vector<Scalar> dist_j(param_knn);
        std::vector<Scalar> weights_j(param_knn);
        std::vector<Index> neigh_j(param_knn);

        Index deg_j = 0;
        for (SpMat::InnerIterator it(B, j); it; ++it) {
            Index k = it.col();
            dist_j[deg_j] = (VD_rs.col(k) - VD_rs.col(j))
                                .cwiseProduct(VD_rs.col(k) - VD_rs.col(j))
                                .mean();
            neigh_j[deg_j] = k;
            ++deg_j;
            if (deg_j >= param_knn)
                break;
        }

        normalize_weights(deg_j, dist_j, weights_j);

#pragma omp critical
        {
            for (Index i = 0; i < deg_j; ++i) {
                const Index k = neigh_j[i];
                const Scalar w = weights_j[i];
                const Scalar d = dist_j[i];

                knn_index.emplace_back(j, k, w, d);
            }

            prog.update();
            prog(std::cerr);
        } // critical
    }
}

template <typename Derived>
int
build_bbknn(Eigen::MatrixBase<Derived> &VD_rank_sample,
            const std::vector<std::vector<Index>> &batch_index_set,
            const std::size_t knn,
            std::vector<std::tuple<Index, Index, Scalar, Scalar>> &knn_index,
            const std::size_t KNN_BILINK = 10,
            const std::size_t KNN_NNLIST = 10,
            const std::size_t NUM_THREADS = 1)
{

    ////////////////////////////////
    // adjust matching parameters //
    ////////////////////////////////

    const std::size_t param_rank = VD_rank_sample.rows();
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

    /////////////////////////////////////////
    // construct dictionary for each batch //
    /////////////////////////////////////////

    /** Take spectral data for a particular treatment group "k"
     * @param k type index
     */
    auto build_spectral_data = [&](const Index k) -> Mat {
        const std::vector<Index> &col_k = batch_index_set[k];
        const Index Nk = col_k.size();
        Mat ret(param_rank, Nk);
        ret.setZero();

        for (Index j = 0; j < Nk; ++j) {
            ret.col(j) = VD_rank_sample.col(col_k[j]);
        }
        return ret;
    };

    using vs_type = hnswlib::L2Space;
    std::vector<std::shared_ptr<vs_type>> vs_vec;
    std::vector<std::shared_ptr<KnnAlg>> knn_lookup_vec;

    const Index Nsample = VD_rank_sample.cols();
    const Index Nbatch = batch_index_set.size();

    // Mat VD(param_rank, Nsample);

    for (Index bb = 0; bb < Nbatch; ++bb) {

        const Index n_tot = batch_index_set[bb].size();
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

            // const auto &bset = batch_index_set.at(bb);

            Mat dat = build_spectral_data(bb);
            float *mass = dat.data(); // adding data points

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
            for (Index i = 0; i < n_tot; ++i) {
#pragma omp critical
                {
                    alg.addPoint((void *)(mass + param_rank * i), i);
                    // const Index j = bset.at(i);
                    // VD.col(j) = dat.col(i);
                    prog.update();
                    prog(Rcpp::Rcerr);
                }
            }
        }
    }

    TLOG("Built the dictionaries for fast look-ups");

    ///////////////////////////////////////////////////
    // step 1: build mutual kNN graph across batches //
    ///////////////////////////////////////////////////
    std::vector<std::tuple<Index, Index, Scalar>> backbone;

    {
        float *mass = VD_rank_sample.derived().data();
        progress_bar_t<Index> prog(Nsample, 1e2);

        for (Index j = 0; j < Nsample; ++j) {

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif

            for (Index bb = 0; bb < Nbatch; ++bb) {
#pragma omp critical
                {
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
                } // critical
            }
            prog.update();
            prog(std::cerr);
        }
    }

    auto backbone_rec = keep_reciprocal_knn(backbone);

    TLOG("Constructed kNN graph backbone");

    ///////////////////////////////////
    // step2: calibrate edge weights //
    ///////////////////////////////////

    reweight_knn_graph(VD_rank_sample,
                       backbone_rec,
                       param_knn,
                       knn_index,
                       NUM_THREADS);

    TLOG("Adjusted kNN weights");
    return EXIT_SUCCESS;
}

#endif
