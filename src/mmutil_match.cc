#include "mmutil_match.hh"

int
search_knn(const SrcDataT _SrcData, //
           const TgtDataT _TgtData, //
           const KNN _knn,          //
           const BILINK _bilink,    //
           const NNLIST _nnlist,    //
           const Index NUM_THREADS, //
           index_triplet_vec &out)
{
    ERR_RET(_SrcData.vecdim != _TgtData.vecdim,
            "source and target must have the same dimensionality");

    const std::size_t knn = _knn.val;
    const std::size_t vecdim = _TgtData.vecdim;
    const std::size_t vecsize = _TgtData.vecsize;

    std::size_t param_bilink = _bilink.val;
    std::size_t param_nnlist = _nnlist.val;

    if (param_bilink >= vecdim) {
        WLOG("too big M value: " << param_bilink << " vs. " << vecdim);
        param_bilink = vecdim - 1;
    }

    if (param_bilink < 2) {
        WLOG("too small M value");
        param_bilink = 2;
    }

    if (param_nnlist <= knn) {
        WLOG("too small N value");
        param_nnlist = knn + 1;
    }

    // Construct KnnAlg interface
    // hnswlib::InnerProductSpace vecspace(vecdim);
    hnswlib::L2Space vecspace(vecdim);

    KnnAlg alg(&vecspace, vecsize, param_bilink, param_nnlist);
    alg.ef_ = param_nnlist;

    TLOG("Initializing kNN algorithm");

    {
        const float *mass = _TgtData.data;

        // progress_bar_t<Index> prog(vecsize, 1e2);

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
        for (Index i = 0; i < vecsize; ++i) {
            alg.addPoint((void *)(mass + vecdim * i),
                         static_cast<std::size_t>(i));
            // prog.update();
            // prog(Rcpp::Rcerr);
        }
    }

    ////////////
    // recall //
    ////////////

    {
        const Index N = _SrcData.vecsize;
        TLOG("Finding " << knn << " nearest neighbors for N = " << N);

        const float *mass = _SrcData.data;
        // progress_bar_t<Index> prog(_SrcData.vecsize, 1e2);

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
        for (Index i = 0; i < _SrcData.vecsize; ++i) {
            auto pq = alg.searchKnn((void *)(mass + vecdim * i), knn);
            float d = 0;
            std::size_t j;
            while (!pq.empty()) {
                std::tie(d, j) = pq.top();
                out.emplace_back(i, j, d);
                pq.pop();
            }
            // prog.update();
            // prog(Rcpp::Rcerr);
        }
    }
    TLOG("Done kNN searches");
    return EXIT_SUCCESS;
}

void
normalize_weights(const Index deg_i,
                  std::vector<float> &dist,
                  std::vector<float> &weights)
{
    if (deg_i < 2) {
        weights[0] = 1.;
        return;
    }

    const float _log2 = fasterlog(2.);
    const float _di = static_cast<float>(deg_i);
    const float log2K = fasterlog(_di) / _log2;

    float lambda = 10.0;

    const float dmin = *std::min_element(dist.begin(), dist.begin() + deg_i);

    // Find lambda values by a simple line-search
    auto f = [&](const float lam) -> float {
        float rhs = 0.;
        for (Index j = 0; j < deg_i; ++j) {
            float w = fasterexp(-(dist[j] - dmin) * lam);
            rhs += w;
        }
        float lhs = log2K;
        return (lhs - rhs);
    };

    float fval = f(lambda);

    const Index max_iter = 100;

    for (Index iter = 0; iter < max_iter; ++iter) {
        float _lam = lambda;
        if (fval < 0.) {
            _lam = lambda * 1.1;
        } else {
            _lam = lambda * 0.9;
        }
        float _fval = f(_lam);
        if (std::abs(_fval) > std::abs(fval)) {
            break;
        }
        lambda = _lam;
        fval = _fval;
    }

    for (Index j = 0; j < deg_i; ++j) {
        weights[j] = fasterexp(-(dist[j] - dmin) * lambda);
    }
}
