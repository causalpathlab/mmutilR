#include "mmutil_topic.hh"

namespace mmutil { namespace topic {

void
topic_data_t::add_sr(const Index g,
                     const Index t,
                     const Scalar s_gj,
                     const Scalar r_gj)
{
    S(t, g) = S(t, g) + s_gj;
    R(t, g) = R(t, g) + r_gj;
}

void
topic_data_t::remove_sr(const Index g,
                        const Index t,
                        const Scalar s_gj,
                        const Scalar r_gj)
{
    S(t, g) = S(t, g) - s_gj;
    R(t, g) = R(t, g) - r_gj;
}

void
topic_data_t::add_sur(const Index g,
                      const Index t,
                      const Scalar s_gj,
                      const Scalar u_gj,
                      const Scalar r_gj)
{
    add_sr(g, t, s_gj, r_gj);
    U(t, g) = U(t, g) + u_gj;
}

void
topic_data_t::remove_sur(const Index g,
                         const Index t,
                         const Scalar s_gj,
                         const Scalar u_gj,
                         const Scalar r_gj)
{
    remove_sr(g, t, s_gj, r_gj);
    U(t, g) = U(t, g) - u_gj;
}

Scalar
topic_data_t::log_predictive_sr(const Index g,
                                const Index t,
                                const Scalar s_new,
                                const Scalar r_new)
{
    const Scalar ss = S(g, t);
    const Scalar rr = R(g, t);

    // Gamma(phi + s + S)  (phi + R)^(phi + S)
    // ------------------ ----------------------------
    // Gamma(phi + s)      (phi + r + R)^(phi + s + S)

    Scalar ret = fasterlgamma(phi + s_new + ss);
    ret -= fasterlgamma(phi + ss);
    ret += (phi + ss) * fasterlog(phi + rr);
    ret -= (phi + s_new + ss) * fasterlog(phi + r_new + rr);

    return ret;
}

Scalar
topic_data_t::log_predictive_sur(const Index g,
                                 const Index t,
                                 const Scalar s_new,
                                 const Scalar u_new,
                                 const Scalar r_new)
{
    const Scalar ss = S(g, t);
    const Scalar uu = U(g, t);

    //////////////////////
    // 1. p(z,s,r,data) //
    //////////////////////

    // Gamma(phi + s + S)  (phi + R)^(phi + S)
    // ------------------ ----------------------------
    // Gamma(phi + s)      (phi + r + R)^(phi + s + S)

    Scalar ret = log_predictive_sr(g, t, s_new, r_new);

    ////////////////////////
    // 2. p(z,u,s,r,data) //
    ////////////////////////

    // Gamma(phi + u + U)  (phi + S)^(phi + U)
    // ------------------ ----------------------------
    // Gamma(phi + U)      (phi + s + S)^(phi + u + U)

    ret += fasterlgamma(phi + u_new + uu);
    ret -= fasterlgamma(phi + uu);
    ret += (phi + uu) * fasterlog(phi + ss);
    ret -= (phi + u_new + uu) * fasterlog(phi + s_new + ss);

    return ret;
}

}} // end of namespace
