#include "mmutil.hh"
#include "mmutil_io.hh"
#include "mmutil_bgzf_util.hh"
#include "mmutil_index.hh"

#ifndef MMUTIL_VELOCITY_HH_
#define MMUTIL_VELOCITY_HH_

namespace mmutil { namespace velocity {

struct NGENES {
    explicit NGENES(const Index _val)
        : val(_val)
    {
    }
    const Index val;
};

struct NTYPES {
    explicit NTYPES(const Index _val)
        : val(_val)
    {
    }
    const Index val;
};

struct A0 {
    explicit A0(const Scalar _val)
        : val(_val)
    {
    }
    const Scalar val;
};

struct B0 {
    explicit B0(const Scalar _val)
        : val(_val)
    {
    }
    const Scalar val;
};

struct EPS {
    explicit EPS(const Scalar _val)
        : val(_val)
    {
    }
    const Scalar val;
};

////////////////////////////////////////////////////////////////
//' Data loader for one column of the spliced & unspliced mtx
//'
struct data_loader_t {

    using reader_t = mmutil::io::one_column_reader_t;

    explicit data_loader_t(const std::string _spliced_file,
                           const std::string _unspliced_file,
                           const std::vector<Index> &_spliced_idx_tab,
                           const std::vector<Index> &_unspliced_idx_tab,
                           const Index Ngene)
        : spliced_file(_spliced_file)
        , unspliced_file(_unspliced_file)
        , spliced_idx_tab(_spliced_idx_tab)
        , unspliced_idx_tab(_unspliced_idx_tab)
        , target_col(0)
        , spliced_reader(Ngene, target_col)
        , unspliced_reader(Ngene, target_col)
    {
    }

    int read(const Index j);
    const Mat &spliced() const;
    const Mat &unspliced() const;

private:
    void set_mem_loc(const std::vector<Index> &_idx_tab);

private:
    const std::string spliced_file;
    const std::string unspliced_file;
    const std::vector<Index> &spliced_idx_tab;
    const std::vector<Index> &unspliced_idx_tab;

    Index target_col;

    reader_t spliced_reader;
    reader_t unspliced_reader;

    Index lb;
    Index ub;
};

////////////////////////////////////////////////////////////////
//' Keeping track of gene x cell type statistics
//' without keeping/generating full gene x cell data
//'
struct aggregated_delta_model_t {

    aggregated_delta_model_t(const NGENES Ng,
                             const NTYPES Nt,
                             const A0 _a0,
                             const B0 _b0,
                             const EPS _Eps)
        : Ngenes(Ng.val)
        , Ntypes(Nt.val)
        , UC(Ngenes, Ntypes)
        , PhiC(Ngenes, Ntypes)
        , delta(Ngenes, Ntypes)
        , delta_old(Ngenes, Ntypes)
        , delta_agg(Ngenes, 1)
        , delta_old_agg(Ngenes, 1)
        , eps(_Eps.val)
        , phi_new_j(Ngenes)
        , phi_old_j(Ngenes)
        , update_phi_op(_Eps.val)
        , update_delta_op(_a0.val, _b0.val)
        , rate_sd_op(_a0.val, _b0.val)
        , rate_ln_op(_a0.val, _b0.val)
        , rate_sd_ln_op(_a0.val)
    {
        UC.setZero();
        PhiC.setZero();
        delta.setOnes();
        delta_old.setOnes();
        phi_new_j.setZero();
        phi_old_j.setZero();
        n = 0;
    }

    // UC[g, t] += U[g,j] * C[j, t] + e
    // phi[g,j] +=  (U[g,j] + S[g,j]) / (1 + delta[g])
    // PhiC[g, t] += phi[, j] * C[j, t]
    template <typename U, typename S, typename C>
    void add_stat(const Eigen::MatrixBase<U> &_u_j,
                  const Eigen::MatrixBase<S> &_s_j,
                  const Eigen::MatrixBase<C> &_c_j)
    {
        const U &u_j = _u_j.derived();
        const S &s_j = _s_j.derived();
        const C &c_j = _c_j.derived();

        // -- previous code --
        // UC += ((u_j * c_j.transpose()).array() + eps).matrix();
        for (Index g = 0; g < Ngenes; ++g) {
            if (u_j(g) > 0) {
                const Scalar ue = u_j(g) + eps;
                UC.row(g) += c_j.transpose() * ue;
            }
        }

        delta_agg = delta * c_j;

        // --- previous code --
        // phi_new_j = (u_j + s_j).binaryExpr(delta_agg, update_phi_op);
        phi_new_j.setZero();
        for (Index g = 0; g < Ngenes; ++g) {
            if (u_j(g) > 0) {
                const Scalar use = u_j(g) + s_j(g) + eps;
                phi_new_j(g) = use / (1. + delta_agg(g));
            }
        }

        for (Index g = 0; g < Ngenes; ++g) {
            if (u_j(g) > 0) {
                PhiC.row(g) += c_j.transpose() * phi_new_j(g);
            }
        }
        ++n;
    }

    // phi[g,j] =  (U[g,j] + S[g,j]) / (1 + delta[g])
    // update Phi * C accordingly
    // (1) remove old phi[g,j] information
    // (2) add new phi[g,j] information
    template <typename U, typename S, typename C>
    void update_phi_stat(const Eigen::MatrixBase<U> &_u_j,
                         const Eigen::MatrixBase<S> &_s_j,
                         const Eigen::MatrixBase<C> &_c_j)
    {
        const U &u_j = _u_j.derived();
        const S &s_j = _s_j.derived();
        const C &c_j = _c_j.derived();

        delta_agg = delta * c_j;
        delta_old_agg = delta_old * c_j;

        // --- previous code --
        // phi_old_j = (u_j + s_j).binaryExpr(delta_old_agg, update_phi_op);
        // phi_new_j = (u_j + s_j).binaryExpr(delta_agg, update_phi_op);
        phi_new_j.setZero();
        phi_old_j.setZero();
        for (Index g = 0; g < Ngenes; ++g) {
            if (u_j(g) > 0) {
                const Scalar use = u_j(g) + s_j(g) + eps;
                phi_old_j(g) = use / (1. + delta_old_agg(g));
                phi_new_j(g) = use / (1. + delta_agg(g));
            }
        }

        for (Index g = 0; g < Ngenes; ++g) {
            if (u_j(g) > 0) {
                const Scalar d_phi = phi_new_j(g) - phi_old_j(g);
                PhiC.row(g) += c_j.transpose() * d_phi;
            }
        }

        // for (Index k = 0; k < Ntypes; ++k)
        //     PhiC.col(k) += (phi_new_j - phi_old_j) * c_j(k);
    }

    // UC += U * C + e
    // phi =  (U + S + e) / (1 + delta)
    // PhiC += phi * C
    // template <typename U, typename S, typename C>
    // void add_stat_bulk(const Eigen::SparseMatrixBase<U> &_u,
    //                    const Eigen::SparseMatrixBase<S> &_s,
    //                    const Eigen::MatrixBase<C> &_c)
    // {
    //     const U &uu = _u.derived();
    //     const S &ss = _s.derived();
    //     const C &cc = _c.derived();
    //     ASSERT(uu.cols() == ss.cols(), "cols(U) !=  cols(S)");
    //     UC += ((uu * cc.transpose()).array() + eps).matrix();
    //     for (Index j = 0; j < uu.cols(); ++j) {
    //         phi_new_j = uu.col(j) + ss.col(j);
    //         PhiC += (phi_new_j.binaryExpr(delta * cc.col(j), update_phi_op))
    //         *
    //             cc.col(j).transpose();
    //     }
    //     n += uu.cols();
    // }

    // phi[g,j] =  (U[g,j] + S[g,j]) / (1 + delta[g])
    // update Phi * C accordingly
    // (1) remove old phi[g,j] information
    // (2) add new phi[g,j] information
    // template <typename U, typename S, typename C>
    // void update_phi_stat_bulk(const Eigen::SparseMatrixBase<U> &_u,
    //                           const Eigen::SparseMatrixBase<S> &_s,
    //                           const Eigen::MatrixBase<C> &_c)
    // {
    //     const U &uu = _u.derived();
    //     const S &ss = _s.derived();
    //     const C &cc = _c.derived();
    //     for (Index j = 0; j < uu.cols(); ++j) {
    //         phi_new_j = uu.col(j) + ss.col(j);
    //         phi_old_j = uu.col(j) + ss.col(j);
    //         PhiC +=
    //             (phi_new_j.binaryExpr(delta * cc.col(j), update_phi_op) -
    //              phi_old_j.binaryExpr(delta_old * cc.col(j), update_phi_op))
    //              *
    //             cc.col(j).transpose();
    //     }
    // }

    void update_delta_stat();

    Scalar update_diff();

    Index nsample() const;

    Mat get_delta() const;
    Mat get_sd_delta() const;
    Mat get_ln_delta() const;
    Mat get_sd_ln_delta() const;

    const Index Ngenes;
    const Index Ntypes;

private:
    ///////////////////////////
    // sufficient statistics //
    ///////////////////////////

    Mat UC;            // U * C (gene x type)
    Mat PhiC;          // Phi * C (gene x type)
    Mat delta;         // (gene x type)
    Mat delta_old;     // (gene x type)
    Mat delta_agg;     // (gene x 1)
    Mat delta_old_agg; // (gene x 1)

    const Scalar eps;

    Index n; // counter

    struct update_phi_op_t {
        explicit update_phi_op_t(const Scalar _eps)
            : eps(_eps)
        {
        }

        Scalar operator()(const Scalar &us, const Scalar &d) const
        {
            return (us + eps) / (one_val + d);
        }
        const Scalar eps;
        static constexpr Scalar one_val = 1.;
    };

    update_phi_op_t update_phi_op;

    struct update_delta_op_t {
        explicit update_delta_op_t(const Scalar a, const Scalar b)
            : a0(a)
            , b0(b)
        {
        }

        Scalar operator()(const Scalar &uc, const Scalar &phic) const
        {
            return (uc + a0) / (phic + b0);
        }

        const Scalar a0, b0;
    };

    update_delta_op_t update_delta_op;

    struct log1p_op_t {
        Scalar operator()(const Scalar &x) const { return fasterlog(x + one); }
        static constexpr Scalar one = 1.;
    };

    log1p_op_t log1p_op;

    struct abs_op_t {
        Scalar operator()(const Scalar &x) const { return std::abs(x); }
    };

    abs_op_t abs_op;

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

    rate_sd_op_t rate_sd_op;

    struct rate_ln_op_t {
        explicit rate_ln_op_t(const Scalar _a0, const Scalar _b0)
            : a0(_a0)
            , b0(_b0)
        {
        }
        Scalar operator()(const Scalar &a, const Scalar &b) const
        {
            if ((a + a0) > one)
                return fasterdigamma(a + a0 - one) - fasterlog(b + b0);

            return fasterdigamma(a + a0) - fasterlog(b + b0);
        }
        const Scalar a0, b0;
        static constexpr Scalar one = 1.0;
        static constexpr Scalar zero = 0.0;
    };

    rate_ln_op_t rate_ln_op;

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
            if ((a + a0) > one)
                return std::max(one / std::sqrt(a + a0 - one), zero);

            return std::max(one / std::sqrt(a + a0), zero);
        }
        const Scalar a0;
        static constexpr Scalar one = 1.0;
        static constexpr Scalar zero = 0.0;
    };

    rate_sd_ln_op_t rate_sd_ln_op;

private:
    Vec phi_new_j; // For one cell j (gene x 1)
    Vec phi_old_j; // For one cell j (gene x 1)
};

}} // namespace mmutil::velocity
#endif
