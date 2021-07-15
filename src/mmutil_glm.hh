#include "mmutil.hh"

#ifndef MMUTIL_GLM_HH_
#define MMUTIL_GLM_HH_

struct poisson_pseudo_response_t {
    /// @param y
    /// @param eta
    Scalar operator()(const Scalar &y, const Scalar &eta) const
    {
        return eta + fasterexp(-eta) * y - one;
    }
    static constexpr Scalar one = 1.0;
};

struct poisson_weight_t {
    /// @param eta
    Scalar operator()(const Scalar &eta) const { return fasterexp(eta); }
};

struct poisson_invlink_t {
    /// @param eta
    Scalar operator()(const Scalar &eta) const { return fasterexp(eta); }
};

struct poisson_llik_t {
    /// @param y
    /// @param eta
    Scalar operator()(const Scalar &y, const Scalar &eta) const
    {
        return y * eta - fasterexp(eta);
    }
};

/// Fit GLM by coordinate-wise descent:
template <typename pseudo_response_t, typename weight_t, typename llik_t>
inline Mat
fit_glm(const Mat xx, const Mat y, const Index max_iter, const Scalar reg)
{
    pseudo_response_t resp;
    weight_t weight;
    llik_t llik;

    const Index n = xx.rows();
    const Index p = xx.cols();

    Mat _beta = Mat::Zero(p, 1);
    Mat _eta = xx * _beta;

    Mat _y(n, 1);
    Mat _w(n, 1);
    Mat _r(n, 1);

    Scalar llik_val = 0;
    const Scalar eps = 1e-4;

    for (Index iter = 0; iter < max_iter; ++iter) {

        _y = y.binaryExpr(_eta, resp);
        _w = _eta.unaryExpr(weight);

        for (Index j = 0; j < p; ++j) {
            _r = _eta - xx.col(j) * _beta(j);

            const Scalar num =
                (_y - _r).cwiseProduct(xx.col(j)).cwiseProduct(_w).sum();

            const Scalar denom =
                xx.col(j).cwiseProduct(xx.col(j)).cwiseProduct(_w).sum();

            _beta(j) = num / (denom + reg);

            _eta = _r + xx.col(j) * _beta(j);
        }

        const Scalar new_llik = y.binaryExpr(_eta, llik).mean();

        if (iter > 0 && (new_llik - llik_val) / std::abs(llik_val) < eps) {
            break;
        }
        llik_val = new_llik;
    }
    return _beta;
}

inline Mat
fit_poisson_glm(const Mat xx,
                const Mat y,
                const Index max_iter,
                const Scalar reg)
{
    using R = poisson_pseudo_response_t;
    using W = poisson_weight_t;
    using L = poisson_llik_t;
    return fit_glm<R, W, L>(xx, y, max_iter, reg);
}

inline Mat
predict_poisson_glm(const Mat xx,
                    const Mat y,
                    const Index max_iter,
                    const Scalar reg,
                    const bool intercept = true,
                    const bool do_std = true,
                    const Scalar SD_MIN = 1e-2)
{
    poisson_invlink_t invlink;

    Mat xx_std = do_std ? standardize(xx, SD_MIN) : xx;

    if (intercept) {
        const Mat xx_1 = hcat(xx_std, Mat::Ones(xx.rows(), 1));
        const Mat beta = fit_poisson_glm(xx_1, y, max_iter, reg);
        return (xx_1 * beta).unaryExpr(invlink);
    } else {
        const Mat beta = fit_poisson_glm(xx_std, y, max_iter, reg);
        return (xx_std * beta).unaryExpr(invlink);
    }
}

#endif

////////////////////////////////////////////////////////////////
// #' Prototype R code
// #' @param xx
// #' @param y
// fit.glm <- function(xx, y, reg = 1) {
//   p = ncol(xx)
//   beta = rep(0, p)
//   beta.se = rep(sqrt(reg), p)
//   eta = xx %*% beta
//   llik = 0
//   for(iter in 1:100) {
//     llik.old = llik
//     y.pseudo = -1 + eta + exp(-eta) * y
//     ww = exp(eta)
//     for(j in 1:p) {
//       .r = eta - xx[, j] * beta[j]
//       .num = sum(ww * xx[, j] * (y.pseudo - .r))
//       .denom = sum(ww * xx[, j]^2) + reg
//       beta[j] = .num / .denom
//       beta.se[j] = 1 / sqrt(sum(ww * xx[, j]^2))
//       eta = .r + xx[, j] * beta[j]
//     }
//     llik = mean(y * eta - exp(eta))
//     if(abs(llik.old - llik) / abs(llik.old + 1e-4) < 1e-4){
//       break
//     }
//   }
//   list(beta = beta, se = beta.se)
// }
////////////////////////////////////////////////////////////////
