#include "mmutil.hh"
#include <random>
#include <unordered_map>

#ifndef MMUTIL_POIS_HH_
#define MMUTIL_POIS_HH_

struct poisson_t {

    explicit poisson_t(const Mat _yy,
                       const Mat _zz,
                       const Scalar _a0,
                       const Scalar _b0)
        : yy(_yy)
        , zz(_zz)
        , a0(_a0)
        , b0(_b0)
        , D(yy.rows())
        , N(yy.cols())
        , K(zz.rows())
        , eval_cf(false)
        , rate_op(a0, b0)
        , rate_sd_op(a0, b0)
        , rate_ln_op(a0, b0)
        , rate_sd_ln_op(a0)
        , ent_op(a0, b0)
        , mu(K, D)
        , rho(N, 1)
        , rho_cf(N, 1)
        , ln_mu(K, D)
        , ln_rho(N, 1)
        , ln_rho_cf(N, 1)
        , ent_mu(K, D)
        , ent_rho(N, 1)
        , ent_rho_cf(N, 1)
        , ZY(K, D)
        , Ytot(N, 1)
        , Ytot_cf(N, 1)
        , denomK(K, 1)
        , denomN(N, 1)
        , onesD(D, 1)
    {
        // TLOG("Creating a model for " << D << " x " << N << " data");
        onesD.setOnes();

        rho.setConstant(a0 / b0);
        ln_rho.setConstant(fasterdigamma(a0) - fasterlog(b0));

        mu.setOnes();
        ln_mu.setZero();
        Ytot = yy.transpose() * onesD; // N x 1

        ZY = zz * yy.transpose(); // K x D
        verbose = false;
    }

    explicit poisson_t(const Mat _yy,
                       const Mat _zz,
                       const Mat _yy_cf,
                       const Mat _zz_cf,
                       const Scalar _a0,
                       const Scalar _b0)
        : yy(_yy)
        , zz(_zz)
        , yy_cf(_yy_cf)
        , zz_cf(_zz_cf)
        , a0(_a0)
        , b0(_b0)
        , D(yy.rows())
        , N(yy.cols())
        , K(zz.rows())
        , eval_cf(true)
        , rate_op(a0, b0)
        , rate_sd_op(a0, b0)
        , rate_ln_op(a0, b0)
        , rate_sd_ln_op(a0)
        , ent_op(a0, b0)
        , mu(K, D)
        , mu_resid(K, D)
        , rho(N, 1)
        , rho_cf(N, 1)
        , rho_resid(N, 1)
        , ln_mu(K, D)
        , ln_mu_resid(K, D)
        , ln_rho(N, 1)
        , ln_rho_cf(N, 1)
        , ln_rho_resid(N, 1)
        , ent_mu(K, D)
        , ent_mu_resid(K, D)
        , ent_rho(N, 1)
        , ent_rho_cf(N, 1)
        , ent_rho_resid(N, 1)
        , ZY(K, D)
        , ZY_resid(K, D)
        , Ytot(N, 1)
        , Ytot_cf(N, 1)
        , denomK(K, 1)
        , denomN(N, 1)
        , onesD(D, 1)
    {
        // TLOG("Creating a model for " << D << " x " << N << " data");

        onesD.setOnes();

        rho.setConstant(a0 / b0);
        rho_cf.setConstant(a0 / b0);

        ln_rho.setConstant(fasterdigamma(a0) - fasterlog(b0));
        ln_rho_cf.setConstant(fasterdigamma(a0) - fasterlog(b0));

        mu.setOnes();
        ln_mu.setZero();

        Ytot = yy.transpose() * onesD;       // N x 1
        Ytot_cf = yy_cf.transpose() * onesD; // N x 1

        ZY = zz * yy.transpose();        // K x D
        ZY += zz_cf * yy_cf.transpose(); // K x D
        verbose = false;
    }

    const Mat yy;
    const Mat zz;

    const Mat yy_cf;
    const Mat zz_cf;

    const Scalar a0;
    const Scalar b0;

    const Index D;
    const Index N;
    const Index K;

    const bool eval_cf;

public:
    inline Scalar elbo()
    {
        Scalar ret = 0.;
        // log-likelihood
        ret += ln_mu.cwiseProduct(ZY).sum();
        ret += ln_rho.cwiseProduct(Ytot).sum();
        ret -= (mu.transpose() * zz * rho).sum();

        // entropy
        ret += ent_mu.sum();
        ret += ent_rho.sum();

        // counterfactual model
        if (eval_cf) {
            ret += ln_rho_cf.cwiseProduct(Ytot_cf).sum();
            ret -= (mu.transpose() * zz_cf * rho_cf).sum();
            ret += ent_rho_cf.sum();
        }
        return ret;
    }

    inline Scalar optimize(const Index maxIter = 100, const Scalar tol = 1e-4)
    {
        const Scalar denom = static_cast<Scalar>(D * N);

        solve_mu();
        solve_rho();
        Scalar score = elbo() / denom;

        for (Index iter = 0; iter < maxIter; ++iter) {
            solve_mu();
            solve_rho();
            Scalar _score = elbo() / denom;
            Scalar diff = (score - _score) / (std::abs(score) + tol);

            if (diff < tol) {
                break;
            }

            score = _score;
        }

        return score;
    }

    inline Mat mu_DK() { return mu.transpose(); }

    inline Mat ln_mu_DK() { return ln_mu.transpose(); }

    inline Mat ln_mu_sd_DK()
    {
        Mat ret = ZY.unaryExpr(rate_sd_ln_op);
        ret.transposeInPlace();
        return ret;
    }

    inline Mat mu_sd_DK()
    {
        Mat ret(K, D);

        for (Index g = 0; g < D; ++g) {
            ret.col(g) = ZY.col(g).binaryExpr(denomK, rate_sd_op);
        }
        ret.transposeInPlace();
        return ret;
    }

public:
    inline Scalar elbo_resid()
    {
        Scalar ret = 0.;
        // log-likelihood
        ret += ln_mu.cwiseProduct(ZY_resid).sum();
        ret += ln_mu_resid.cwiseProduct(ZY_resid).sum();
        ret += ln_rho_resid.cwiseProduct(Ytot).sum();
        ret -= ((mu_resid.cwiseProduct(mu)).transpose() * zz * rho_resid).sum();

        // entropy
        ret += ent_mu_resid.sum();
        ret += ent_rho_resid.sum();

        return ret;
    }

    inline Scalar residual_optimize(const Index maxIter = 100,
                                    const Scalar tol = 1e-4)
    {
        const Scalar denom = static_cast<Scalar>(D * N);

        ZY_resid = zz * yy.transpose(); // K x D

        rho_resid.setConstant(a0 / b0); //  rho;
        solve_mu_resid();               //

        Scalar score = elbo_resid() / denom;

        for (Index iter = 0; iter < maxIter; ++iter) {
            solve_rho_resid(); //
            solve_mu_resid();

            Scalar _score = elbo_resid() / denom;
            Scalar diff = (score - _score) / (std::abs(score) + tol);

            if (diff < tol) {
                break;
            }

            score = _score;
        }

        return score;

        // mu_resid.setOnes(); // initialize that there is no diff effect
        // solve_rho_resid();  // just fix the rho
        // solve_mu_resid();
        // return elbo_resid() / denom;
    }

    inline Mat residual_mu_DK() { return mu_resid.transpose(); }

    inline Mat residual_mu_sd_DK()
    {
        Mat ret(K, D);
        ZY_resid = zz * yy.transpose(); // K x D
        denomK = zz * rho_resid;

        for (Index g = 0; g < D; ++g) {
            mu_resid.col(g) =
                ZY_resid.col(g).binaryExpr(denomK.cwiseProduct(mu.col(g)),
                                           rate_sd_op);
        }

        ret.transposeInPlace();
        return ret;
    }

    inline Mat ln_residual_mu_DK() { return ln_mu_resid.transpose(); }

    inline Mat ln_residual_mu_sd_DK()
    {
        Mat ret = ZY_resid.unaryExpr(rate_sd_ln_op);
        ret.transposeInPlace();
        return ret;
    }

    inline Mat rho_N() { return rho; }

    inline Mat rho_cf_N() { return rho_cf; }

private:
    inline void solve_mu()
    {
        if (eval_cf) {
            denomK = zz * rho + zz_cf * rho_cf;
        } else {
            denomK = zz * rho;
        }

        for (Index g = 0; g < D; ++g) {
            mu.col(g) = ZY.col(g).binaryExpr(denomK, rate_op);
            ln_mu.col(g) = ZY.col(g).binaryExpr(denomK, rate_ln_op);
            ent_mu.col(g) = ZY.col(g).binaryExpr(denomK, ent_op);
        }
    }

    inline void solve_rho()
    {
        // observed model
        denomN = zz.transpose() * mu * onesD;
        rho = Ytot.binaryExpr(denomN, rate_op);
        ln_rho = Ytot.binaryExpr(denomN, rate_ln_op);
        ent_rho = Ytot.binaryExpr(denomN, ent_op);

        // counterfactual model
        if (eval_cf) {
            denomN = zz_cf.transpose() * mu * onesD;
            rho_cf = Ytot_cf.binaryExpr(denomN, rate_op);
            ln_rho_cf = Ytot_cf.binaryExpr(denomN, rate_ln_op);
            ent_rho_cf = Ytot_cf.binaryExpr(denomN, ent_op);
        }
    }

private:
    inline void solve_mu_resid()
    {
        denomK = zz * rho_resid;

        for (Index g = 0; g < D; ++g) {
            mu_resid.col(g) =
                ZY_resid.col(g).binaryExpr(denomK.cwiseProduct(mu.col(g)),
                                           rate_op);

            ln_mu_resid.col(g) =
                ZY_resid.col(g).binaryExpr(denomK.cwiseProduct(mu.col(g)),
                                           rate_ln_op);

            ent_mu_resid.col(g) =
                ZY_resid.col(g).binaryExpr(denomK.cwiseProduct(mu.col(g)),
                                           ent_op);
        }
    }

    inline void solve_rho_resid()
    {
        denomN = zz.transpose() * (mu.cwiseProduct(mu_resid)) * onesD;
        rho_resid = Ytot.binaryExpr(denomN, rate_op);
        ln_rho_resid = Ytot.binaryExpr(denomN, rate_ln_op);
        ent_rho_resid = Ytot.binaryExpr(denomN, ent_op);
    }

public:
    // a/b
    struct rate_op_t {
        explicit rate_op_t(const Scalar _a0, const Scalar _b0)
            : a0(_a0)
            , b0(_b0)
        {
        }
        Scalar operator()(const Scalar &a, const Scalar &b) const
        {
            return (a + a0) / (b + b0);
        }
        const Scalar a0, b0;
    };

    // sqrt(a) / b
    struct rate_sd_op_t {
        explicit rate_sd_op_t(const Scalar _a0, const Scalar _b0)
            : a0(_a0)
            , b0(_b0)
        {
        }
        Scalar operator()(const Scalar &a, const Scalar &b) const
        {
            return std::max(std::sqrt(a + a0) / (b + b0),
                            static_cast<Scalar>(0.));
        }
        const Scalar a0, b0;
    };

    struct rate_ln_op_t {
        explicit rate_ln_op_t(const Scalar _a0, const Scalar _b0)
            : a0(_a0)
            , b0(_b0)
        {
        }
        Scalar operator()(const Scalar &a, const Scalar &b) const
        {
            const Scalar one = 1.0;
            const Scalar zero = 0.0;

            if ((a + a0) > one)
                return fasterdigamma(a + a0 - one) - fasterlog(b + b0);

            return fasterdigamma(a + a0) - fasterlog(b + b0);
        }
        const Scalar a0, b0;
    };

    // Delta method
    // sqrt V[ln(mu)] = sqrt (V[mu] / mu)
    //                = 1/sqrt(a -1 )
    // approximated at the mode = (a - 1)/b
    struct rate_sd_ln_op_t {
        explicit rate_sd_ln_op_t(const Scalar _a0)
            : a0(_a0)
        {
        }
        Scalar operator()(const Scalar &a) const
        {
            const Scalar one = 1.0;
            const Scalar zero = 0.0;
            if ((a + a0) > one)
                return std::max(one / std::sqrt(a + a0 - one), zero);

            return std::max(one / std::sqrt(a + a0), zero);
        }
        const Scalar a0;
    };

    struct ent_op_t {
        explicit ent_op_t(const Scalar _a0, const Scalar _b0)
            : a0(_a0)
            , b0(_b0)
        {
        }
        Scalar operator()(const Scalar &a, const Scalar &b) const
        {
            const Scalar _a = a + a0;
            const Scalar _b = b + b0;
            Scalar ret = -(_a)*fasterlog(_b);
            ret += fasterlgamma(_a);
            ret -= (_a - 1.) * (fasterdigamma(_a) - fasterlog(_b));
            ret += (_a);
            return ret;
        }
        const Scalar a0, b0;
    };

private:
    rate_op_t rate_op;
    rate_sd_op_t rate_sd_op;
    rate_ln_op_t rate_ln_op;
    rate_sd_ln_op_t rate_sd_ln_op;
    ent_op_t ent_op;

private:
    Mat mu;
    Mat mu_resid;
    Mat rho;
    Mat rho_cf;
    Mat rho_resid;

    Mat ln_mu;
    Mat ln_mu_resid;
    Mat ln_rho;
    Mat ln_rho_cf;
    Mat ln_rho_resid;

    Mat ent_mu;
    Mat ent_mu_resid;
    Mat ent_rho;
    Mat ent_rho_cf;
    Mat ent_rho_resid;

    Mat ZY;       // combined
    Mat ZY_resid; // combined (for residual calculation)
    Mat Ytot;     // observed
    Mat Ytot_cf;  // counterfactual

    Mat denomK;
    Mat denomN;
    Mat onesD;

    bool verbose;
};

#endif
