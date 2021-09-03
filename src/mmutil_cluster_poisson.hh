#include "mmutil_cluster.hh"

#ifndef MMUTIL_CLUSTER_POISSON_HH_
#define MMUTIL_CLUSTER_POISSON_HH_

namespace mmutil { namespace cluster {

struct poisson_component_t {

    explicit poisson_component_t(const dim_t dim_,
                                 const pseudo_count_t a0_,
                                 const pseudo_count_t b0_)
        : dim(dim_.val)
        , d(dim)
        , a0(a0_.val)
        , b0(b0_.val)
        , Freq_stat(dim)
        , N_stat(0)
        , V_stat(0)
    {
        clear();
    }

    void clear();

    void add_point(const SpMat &x);

    void remove_point(const SpMat &x);

    Scalar log_predictive(const SpMat &x);

    Vec MLE() const;

    const Index dim;
    const Scalar d;
    const Scalar a0;
    const Scalar b0;

    constexpr static Scalar TOL = 1e-4;

private:
    Vec Freq_stat;
    Scalar N_stat;
    Scalar V_stat;
};

}} // end of namespace

#endif
