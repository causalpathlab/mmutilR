#include "mmutil.hh"
#include "fastgamma.h"

#ifndef MMUTIL_LDA_HH_
#define MMUTIL_LDA_HH_

namespace mmutil { namespace topic {

struct NTOPIC {
    explicit NTOPIC(const Index _val)
        : val(_val)
    {
    }
    const Index val;
};

struct NFEATURE {
    explicit NFEATURE(const Index _val)
        : val(_val)
    {
    }
    const Index val;
};

struct PRIOR {
    explicit PRIOR(const Scalar _val)
        : val(_val)
    {
    }
    const Scalar val;
};

// Latent Dirichlet Allocation model
// Treat each column (cell) as a document
// Treat each row (gene) as a word

struct topic_data_t {

    explicit topic_data_t(const NFEATURE dd, const NTOPIC tt, const PRIOR prior)
        : D(dd.val)
        , T(tt.val)
        , S(T, D)
        , U(T, D)
        , R(T, D)
        , phi(prior.val)
    {
        S.setZero();
        U.setZero();
        R.setZero();
    }

    const Index D;
    const Index T;
    const Scalar phi;

    void add_sur(const Index feature_index,
                 const Index topic_index,
                 const Scalar s_gj,
                 const Scalar u_gj,
                 const Scalar r_gj);

    void remove_sur(const Index feature_index,
                    const Index topic_index,
                    const Scalar s_gj,
                    const Scalar u_gj,
                    const Scalar r_gj);

    Scalar log_predictive_sur(const Index g,
                              const Index t,
                              const Scalar s,
                              const Scalar u,
                              const Scalar r);

    void add_sr(const Index feature_index,
                const Index topic_index,
                const Scalar s_gj,
                const Scalar r_gj);

    void remove_sr(const Index feature_index,
                   const Index topic_index,
                   const Scalar s_gj,
                   const Scalar r_gj);

    Scalar log_predictive_sr(const Index g,
                             const Index t,
                             const Scalar s,
                             const Scalar r);

private:
    Mat S; // topic x feature, sum S[g,j] Z[g,j,t]
    Mat U; // topic x feature, sum U[g,j] Z[g,j,t]
    Mat R; // topic x feature, sum rho[j] Z[g,j,t]
};

}} // end of namespace

#endif
