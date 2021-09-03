#include "mmutil_cluster_poisson.hh"

namespace mmutil { namespace cluster {

///////////////////////
// poisson component //
///////////////////////

// 1. Freq_stat[g] = sum_e x[e, g]
// 2. N_stat = sum_e 1
// 3. V_stat = sum_{e,g} x[e, g]
void
poisson_component_t::add_point(const SpMat &x)
{
#ifdef DEBUG
    // x : row-wise sample x feature data matrix
    ASSERT(x.cols() == dim && x.rows() > 0,
           "add: " << x.cols() << " vs. expected: " << dim);
#endif

    for (Index s = 0; s < x.outerSize(); ++s) {
        for (SpMat::InnerIterator gt(x, s); gt; ++gt) {
            const Index g = gt.col();
            const Scalar xg = gt.value();
            if (xg < TOL)
                continue;
            Freq_stat(g) += xg; // vertex frequency
            V_stat += 1.;       // volume
        }                       //
        N_stat += 1.;           // sample size
    }
}

void
poisson_component_t::remove_point(const SpMat &x)
{
#ifdef DEBUG
    ASSERT(x.cols() == dim && x.rows() > 0,
           "remove: " << x.cols() << " vs. expected: " << dim);

    const Scalar small = -1e-4;
#endif

    for (Index s = 0; s < x.outerSize(); ++s) {
        for (SpMat::InnerIterator gt(x, s); gt; ++gt) {
            const Index g = gt.col();
            const Scalar xg = gt.value();
            if (xg < TOL)
                continue;

            Freq_stat(g) -= xg; // frequency
            V_stat -= 1.;       // volume
#ifdef DEBUG
            ASSERT(V_stat > small, "V < 0");
            ASSERT(g < dim, "g = " << g);
            ASSERT(Freq_stat(g) > small,
                   "F[" << g << "] " << Freq_stat(g)
                        << " < 0 after updating x = " << xg);
#endif
        }
        N_stat -= 1.; // sample size
#ifdef DEBUG
        ASSERT(N_stat > small, "N < 0"); // check
#endif
    }
}

Vec
poisson_component_t::MLE() const
{
    return Freq_stat / (N_stat + b0);
}

// a[g] = F[g] + a0
// b = n + b0
//
// log-predictive after adding x[e, ]
// = sum_g lgamma(a[g] + x[e, g]) - lgamma(a[g])
// + sum_g a[g] * log(b)
// - sum_g (a[g] + x[e, g]) * log(b + 1)
//
// = sum_g lgamma(a[g] + x[e, g]) - lgamma(a[g])
// + (V + a0 * dim) * log(b)
// - (V + a0 * dim + deg[e]) * log(b + 1)
//
Scalar
poisson_component_t::log_predictive(const SpMat &x)
{
    ASSERT(x.cols() == dim && x.rows() > 0,
           "pred: " << x.cols() << " vs. expected: " << dim);

    Scalar deg = 0.;
    Scalar n = 0.;
    Scalar term1 = 0.;
    for (Index s = 0; s < x.outerSize(); ++s) {
        for (SpMat::InnerIterator gt(x, s); gt; ++gt) {
            const Index g = gt.col();
            const Scalar aa = a0 + Freq_stat[g];
            const Scalar xg = gt.value();
            if (xg < TOL)
                continue;
            deg += 1.;
            term1 += fasterlgamma(aa + xg);
            term1 -= fasterlgamma(aa);
        }
        n += 1.;
    }

    const Scalar b = b0 + N_stat;
    const Scalar term2 = (V_stat + a0 * d) * fasterlog(b);
    const Scalar term3 = (V_stat + a0 * d + deg) * fasterlog(b + n);
    return term1 + term2 - term3;
}

void
poisson_component_t::clear()
{
    Freq_stat.setZero();
    N_stat = 0;
    V_stat = 0;
}

}} // end of namespace
