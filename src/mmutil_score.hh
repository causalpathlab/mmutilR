#include "mmutil.hh"
#include "mmutil_stat.hh"
#include "mmutil_util.hh"

#ifndef MMUTIL_SCORE_HH_
#define MMUTIL_SCORE_HH_

/**
 * @param collector : a visitor to collect statistics
 * @param take_s1 : a functor for S1
 * @param take_s2 : a functor for S2
 * @param take_n : a functor for N
 * @param take_ntot : a functor for total N
 **/

template <typename COLLECTOR,
          typename _S1,
          typename _S2,
          typename _N,
          typename _Ntot>
inline auto
compute_mtx_stat(const COLLECTOR &collector,
                 _S1 take_s1,
                 _S2 take_s2,
                 _N take_n,
                 _Ntot take_ntot)
{

    const Vec &s1 = take_s1(collector);
    const Vec &s2 = take_s2(collector);
    const IntVec &nvec = take_n(collector);
    const Scalar ntot = take_ntot(collector);
    const Scalar eps = 1e-8;

    auto cv_fun = [&eps](const Scalar &s, const Scalar &m) -> Scalar {
        if (std::abs(m) <= eps)
            return 0.0;
        return s / std::abs(m);
    };

    Vec mu = s1 / ntot;

    /////////////////////////////////
    // Unbiased standard deviation //
    /////////////////////////////////

    Vec sd = ((s2 - s1.cwiseProduct(s1 / ntot)) / std::max(ntot - 1.0, 1.0))
                 .cwiseSqrt();

    //////////////////////////////
    // coefficient of variation //
    //////////////////////////////

    Vec cv = sd.binaryExpr(mu, cv_fun);

    return std::make_tuple(mu,
                           sd,
                           cv,
                           nvec,
                           collector.max_row,
                           collector.max_col);
}

#endif
