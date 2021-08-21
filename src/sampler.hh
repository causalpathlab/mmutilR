#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <random>

#include "fastexp.h"
#include "fastlog.h"

#ifndef SAMPLER_HH_
#define SAMPLER_HH_

template <typename Scalar, typename Index>
struct discrete_sampler_t {
    explicit discrete_sampler_t(const Index k)
        : K(k)
    {
    }

    template <typename Derived>
    Index operator()(const Eigen::MatrixBase<Derived> &xx)
    {
        Index argmax_k;

        const Scalar maxval = xx.maxCoeff(&argmax_k);
        const Scalar exp_sumval = xx.unaryExpr([&maxval](const Scalar x) {
                                        return fasterexp(x - maxval);
                                    })
                                      .sum();

        const Scalar u = Unif(Rng) * exp_sumval;
        Scalar cum = 0.0;
        Index rIndex = 0;

        for (Index k = 0; k < K; ++k) {
            const Scalar val = xx(k);
            cum += fasterexp(val - maxval);
            if (u <= cum) {
                rIndex = k;
                break;
            }
        }
        return rIndex;
    }

    const Index K;

private:
    std::minstd_rand Rng{ std::random_device{}() };
    std::uniform_real_distribution<Scalar> Unif{ 0.0, 1.0 };

    template <typename Derived>
    inline Scalar _log_sum_exp(const Eigen::MatrixBase<Derived> &log_vec)
    {
        const Derived &xx = log_vec.derived();

        Scalar maxlogval = xx(0);
        for (Index j = 1; j < xx.size(); ++j) {
            if (xx(j) > maxlogval)
                maxlogval = xx(j);
        }

        Scalar ret = 0;
        for (Index j = 0; j < xx.size(); ++j) {
            ret += fasterexp(xx(j) - maxlogval);
        }
        return fasterlog(ret) + maxlogval;
    }
};

////////////////////////////////////
// Index sampler excluding itself //
////////////////////////////////////

template <typename Scalar, typename Index>
struct cf_index_sampler_t {

    using DS = discrete_sampler_t<Scalar, Index>;

    using Vec = typename Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    explicit cf_index_sampler_t(const Index ntrt)
        : Ntrt(ntrt)
        , obs_idx(0)
        , cf_idx(Ntrt - 1)
        , sampler(Ntrt - 1)
        , sampling_mass(Ntrt - 1)
    {
        sampling_mass.setZero();
        std::iota(cf_idx.begin(), cf_idx.end(), 1);
    }

    Index operator()(const Index obs)
    {
        _resolve_cf_idx(obs);
        return cf_idx.at(sampler(sampling_mass));
    }

    const Index Ntrt;

private:
    Index obs_idx;
    std::vector<Index> cf_idx;
    DS sampler;
    Vec sampling_mass;

    void _resolve_cf_idx(const Index new_obs_idx)
    {
        if (new_obs_idx != obs_idx) {
            ASSERT(new_obs_idx >= 0 && new_obs_idx < Ntrt,
                   "new index must be in [0, " << Ntrt << ")");
            Index ri = 0;
            for (Index r = 0; r < Ntrt; ++r) {
                if (r != new_obs_idx)
                    cf_idx[ri++] = r;
            }
            obs_idx = new_obs_idx;
        }
    }
};

#endif
