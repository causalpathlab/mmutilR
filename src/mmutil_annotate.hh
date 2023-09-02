#include <algorithm>
#include <random>
#include <unordered_map>
#include <unordered_set>

#include "mmutil.hh"
#include "mmutil_io.hh"
#include "mmutil_index.hh"
#include "mmutil_normalize.hh"
#include "mmutil_index.hh"
#include "mmutil_match.hh"
#include "progress.hh"
#include "mmutil_bgzf_util.hh"
#include "mmutil_util.hh"
#include "mmutil_stat.hh"

#ifdef __cplusplus
extern "C" {
#endif

#include "bgzf.h"

#ifdef __cplusplus
}
#endif

#ifndef MMUTIL_ANNOTATE_HH_
#define MMUTIL_ANNOTATE_HH_

struct annotation_model_t {

    explicit annotation_model_t(const Mat lab,
                                const Mat anti_lab,
                                const Scalar _kmax)
        : nmarker(lab.rows())
        , ntype(lab.cols())
        , mu(nmarker, ntype)
        , mu_anti(nmarker, ntype)
        , log_normalizer(ntype)
        , score(ntype)
        , nn(ntype)
        , kappa_init(1.0)  //
        , kappa_max(_kmax) //
        , kappa(kappa_init)
        , kappa_anti(kappa_init)
    {
        // initialization
        kappa = kappa_init;
        kappa_anti = kappa_init;
        log_normalizer.setZero();
        mu.setZero();
        mu_anti.setZero();
        mu += lab;
        mu_anti += anti_lab;
        nn.setZero();
    }

    template <typename Derived, typename Derived2, typename Derived3>
    void update_param(const Eigen::MatrixBase<Derived> &_xsum,
                      const Eigen::MatrixBase<Derived2> &_xsum_anti,
                      const Eigen::MatrixBase<Derived3> &_nsum)
    {

        const Derived &Stat = _xsum.derived();
        const Derived2 &Stat_anti = _xsum_anti.derived();
        const Derived3 &nsize = _nsum.derived();

        //////////////////////////////////////////////////
        // concentration parameter for von Mises-Fisher //
        //////////////////////////////////////////////////

        // We use the approximation proposed by Banerjee et al. (2005) for
        // simplicity and relatively stable performance
        //
        //          (rbar*d - rbar^3)
        // kappa = -------------------
        //          1 - rbar^2

        const Scalar d = static_cast<Scalar>(nmarker);

        // We may need to share this kappa estimate across all the
        // types since some of the types might be
        // under-represented.

        const Scalar r = Stat.rowwise().sum().norm() / nsize.sum();

        Scalar _kappa = r * (d - r * r) / (1.0 - r * r);

        if (_kappa > kappa_max) {
            _kappa = kappa_max;
        }

        kappa = _kappa;

        const Scalar r0 = Stat_anti.rowwise().sum().norm() / nsize.sum();

        Scalar _kappa_anti = r0 * (d - r0 * r0) / (1.0 - r0 * r0);

        if (_kappa_anti > kappa_max) {
            _kappa_anti = kappa_max;
        }

        kappa_anti = _kappa_anti;

        ////////////////////////
        // update mean vector //
        ////////////////////////

        // Add pseudo count
        nn = nsize.unaryExpr([](const Scalar &x) -> Scalar { return x + 1.; });

        mu = Stat * nn.cwiseInverse().asDiagonal();
        normalize_columns(mu);

        mu_anti = Stat_anti * nn.cwiseInverse().asDiagonal();
        normalize_columns(mu_anti);
        // update_log_normalizer();
    }

    template <typename Derived>
    inline const Vec &log_score(const Eigen::MatrixBase<Derived> &_x)
    {
        const Derived &x = _x.derived();
        score = (mu.transpose() * x) * kappa;
        score -= (mu_anti.transpose() * x) * kappa_anti;
        // score += log_normalizer; // not needed
        return score;
    }

    void update_log_normalizer()
    {
        // Normalizer for vMF
        //
        //            kappa^{d/2 -1}
        // C(kappa) = ----------------
        //            (2pi)^{d/2} I(d/2-1, kappa)
        //
        // where
        // I(v,x) = boost::math::cyl_bessel_i(v,x)
        //
        // ln C ~ (d/2 - 1) ln(kappa) - ln I(d/2, k)
        //

        const Scalar eps = 1e-8;
        const Scalar d = static_cast<Scalar>(nmarker);
        const Scalar df = d * 0.5 - 1.0 + eps;
        const Scalar ln2pi = std::log(2.0 * 3.14159265359);

        auto _log_denom = [&](const Scalar &kap) -> Scalar {
            Scalar ret = (0.5 * d - 1.0) * std::log(kap);
            ret -= ln2pi * (0.5 * d);
            ret -= _log_bessel_i(df, kap);
            return ret;
        };

        log_normalizer.setConstant(_log_denom(kappa));

        // std::cout << "\n\nnormalizer:\n"
        //           << log_normalizer.transpose() << std::endl;
    }

    const Index nmarker;
    const Index ntype;

    Mat mu;             // marker x type matrix
    Mat mu_anti;        // anti marker x type matrix
    Vec log_normalizer; // log-normalizer
    Vec score;          // temporary score
    Vec nn;             // number of samples

    const Scalar kappa_init;
    const Scalar kappa_max;
    Scalar kappa;
    Scalar kappa_anti;
};

struct annotation_stat_t {

    explicit annotation_stat_t(const Mat l1, const Mat l0, const SpMat lqc)
        : L(l1)
        , L0(l0)
        , Lqc(lqc)
        , M(L.rows())
        , K(L.cols())
        , Stat(M, K)
        , Stat_anti(M, K)
        , unc_stat(M, K)
        , unc_stat_anti(M, K)
        , nsize(K)
    {
        const Scalar pseudo = 1e-8;
        nsize.setConstant(pseudo);     //
        Stat.setConstant(pseudo);      // sum x(g,j) * z(j, k)
        Stat_anti.setConstant(pseudo); // sum x0(g,j) * z(j, k)

        unc_stat.setConstant(pseudo);      // sum x(g,j) * z(j, k)
        unc_stat_anti.setConstant(pseudo); // sum x0(g,j) * z(j, k)
    }

    void squeeze(const std::vector<Index> &ss)
    {
        L = row_sub(L, ss);
        L0 = row_sub(L0, ss);
        Lqc = row_sub(Lqc, ss);

        Stat = row_sub(Stat, ss);
        Stat_anti = row_sub(Stat_anti, ss);

        unc_stat = row_sub(unc_stat, ss);
        unc_stat_anti = row_sub(unc_stat_anti, ss);
        M = L.rows();

        subrow.clear();
        subrow.insert(std::end(subrow), std::begin(ss), std::end(ss));
    }

    Mat L;
    Mat L0;
    SpMat Lqc;

    Index M, K;

    Mat Stat;
    Mat Stat_anti;

    Mat unc_stat;
    Mat unc_stat_anti;

    Vec nsize;

    std::vector<std::string> labels;
    std::vector<Index> subrow;
};

struct mm_data_loader_t {

    template <typename T>
    mm_data_loader_t(const T &options)
        : mtx_file(options.mtx_file)
        , idx_file(options.mtx_file + ".index")
        , log_scale(options.log_scale)
    {

        if (!is_file_bgz(mtx_file))
            CHECK(mmutil::bgzf::convert_bgzip(mtx_file));

        CHECK(mmutil::bgzf::peek_bgzf_header(mtx_file, info));

        if (!file_exists(idx_file)) {
            mmutil::index::build_mmutil_index(options.mtx_file, idx_file);
        }

        CHECK(mmutil::index::read_mmutil_index(idx_file, idx_tab));
    }

    Mat
    operator()(const Index lb, const Index ub, const std::vector<Index> &subrow)
    {
        if (subcol.size() != (ub - lb))
            subcol.resize(ub - lb);
        std::iota(subcol.begin(), subcol.end(), lb);

        SpMat x = mmutil::io::read_eigen_sparse_subset_row_col(mtx_file,
                                                               idx_tab,
                                                               subrow,
                                                               subcol);

        if (log_scale) {
            x = x.unaryExpr(log2_op);
        }

        Mat xx = Mat(x);
        normalize_columns(xx);

        return xx;
    }

    Mat operator()(const Index lb, const std::vector<Index> &subrow)
    {

        SpMat x = mmutil::io::read_eigen_sparse_subset_row_col(mtx_file,
                                                               idx_tab,
                                                               subrow,
                                                               { lb });

        if (log_scale) {
            x = x.unaryExpr(log2_op);
        }

        Mat xx = Mat(x);
        normalize_columns(xx);

        return xx;
    }

    const std::string mtx_file; //
    const std::string idx_file; //
    const bool log_scale;       //

    mmutil::index::mm_info_reader_t info; // MM information
    std::vector<Index> idx_tab;           // column indexing
    std::vector<Index> subcol;

    struct log2_op_t {
        Scalar operator()(const Scalar &x) const { return std::log2(1.0 + x); }
    } log2_op;

    Index num_rows() const { return info.max_row; }
    Index num_columns() const { return info.max_col; }
};

struct svd_data_loader_t {

    svd_data_loader_t(const Mat &uu,
                      const Mat &dd,
                      const Mat &vv,
                      const bool _log_scale)
        : U(uu)
        , D(dd)
        , Vt(vv)
        , log_scale(_log_scale)
    {
        Vt.transposeInPlace();
        TLOG("U: " << U.rows() << " x " << U.cols());
        TLOG("Vt: " << Vt.rows() << " x " << Vt.cols());

        ASSERT(U.cols() == D.rows(), "U and D must have the same rank");
        ASSERT(D.cols() == 1, "D must be just a vector");
        Ud = U * D.asDiagonal();
    }

    Mat
    operator()(const Index lb, const Index ub, const std::vector<Index> &subrow)
    {
        Mat ret(subrow.size(), ub - lb);
        ret.setConstant(1e-4);

        for (Index j = 0; j < (ub - lb); ++j) {
            const Index c = j + lb;

            for (Index i = 0; i < subrow.size(); ++i) {
                const Index r = subrow.at(i);
                if (r < 0 || r >= Ud.rows())
                    continue;

                ///////////////////////////////////////////////
                // model only make sense for positive values //
                ///////////////////////////////////////////////
                const Scalar val = Ud.row(r) * Vt.col(c);

                if (val >= 0) {
                    if (log_scale) {
                        ret(i, j) += std::log2(1. + val);
                    } else {
                        ret(i, j) += val;
                    }
                }
            }
        }

        normalize_columns(ret);
        return ret;
    }

    Mat operator()(const Index c, const std::vector<Index> &subrow)
    {
        Mat ret(subrow.size(), 1);
        ret.setConstant(1e-4);

        for (Index i = 0; i < subrow.size(); ++i) {
            const Index r = subrow.at(i);
            if (r < 0 || r >= Ud.rows())
                continue;

            ///////////////////////////////////////////////
            // model only make sense for positive values //
            ///////////////////////////////////////////////
            const Scalar val = Ud.row(r) * Vt.col(c);

            if (val >= 0) {
                if (log_scale) {
                    ret(i, 0) += std::log2(1. + val);
                } else {
                    ret(i, 0) += val;
                }
            }
        }

        normalize_columns(ret);
        return ret;
    }

    Mat U;
    Mat D;
    Mat Vt;
    Mat Ud;

    const bool log_scale; //

    Index num_rows() const { return U.rows(); }
    Index num_columns() const { return Vt.cols(); }
};

std::tuple<SpMat, SpMat, SpMat>
read_annotation_matched(
    const std::unordered_map<std::string, Index> &row_pos,
    const std::unordered_map<std::string, Index> &label_pos,
    const std::vector<std::tuple<std::string, std::string>> &ann_pair_vec,
    const std::vector<std::tuple<std::string, std::string>> &anti_pair_vec,
    const std::vector<std::tuple<std::string, Scalar>> &qc_pair_vec)
{

    const Index max_rows = row_pos.size();
    const Index max_labels = label_pos.size();

    SpMat L(max_rows, max_labels);
    SpMat L0(max_rows, max_labels);
    SpMat Lqc(max_rows, 1);
    L.setZero();
    L0.setZero();
    Lqc.setZero();

    TLOG(max_rows << " rows and " << max_labels << " labels");

    using ET = Eigen::Triplet<Scalar>;

    if (ann_pair_vec.size() > 0) {
        std::vector<ET> triples;
        triples.reserve(ann_pair_vec.size());
        std::string kr, kl;
        for (auto pp : ann_pair_vec) {
            std::tie(kr, kl) = pp;
            if (row_pos.count(kr) > 0 && label_pos.count(kl) > 0) {
                const Index r = row_pos.at(kr);
                const Index l = label_pos.at(kl);
                if (r >= max_rows || l >= max_labels) {
                    WLOG(kr << "[" << r << "] " << kl << "[" << l
                            << "] will be ignored");
                    continue;
                }
                triples.push_back(ET(r, l, 1.0));
            }
        }
        L.reserve(triples.size());
        L.setFromTriplets(triples.begin(), triples.end());
        TLOG("Built the L matrix");
    }

    if (anti_pair_vec.size() > 0) {
        std::vector<ET> triples;
        triples.reserve(anti_pair_vec.size());
        std::string kr, kl;
        for (auto pp : anti_pair_vec) {
            std::tie(kr, kl) = pp;
            if (row_pos.count(kr) > 0 && label_pos.count(kl) > 0) {
                const Index r = row_pos.at(kr);
                const Index l = label_pos.at(kl);
                if (r >= max_rows || l >= max_labels) {
                    WLOG(kr << "[" << r << "] " << kl << "[" << l
                            << "] will be ignored");
                    continue;
                }
                triples.push_back(ET(r, l, 1.0));
            }
        }
        L0.reserve(triples.size());
        L0.setFromTriplets(triples.begin(), triples.end());

        TLOG("Built the L0 matrix");
    }

    if (qc_pair_vec.size() > 0) {
        std::vector<ET> qc_triples;
        qc_triples.reserve(qc_pair_vec.size());
        for (auto pp : qc_pair_vec) {
            if (row_pos.count(std::get<0>(pp)) > 0) {
                Index r = row_pos.at(std::get<0>(pp));
                Scalar threshold = std::get<1>(pp);
                if (r < max_rows) {
                    qc_triples.push_back(ET(r, 0, threshold));
                }
            }
        }
        Lqc.reserve(qc_triples.size());
        Lqc.setFromTriplets(qc_triples.begin(), qc_triples.end());
    }
    TLOG("Built the Lqc matrix");

    return std::make_tuple(L, L0, Lqc);
}

struct annotation_options_t {

    annotation_options_t()
    {
        mtx_file = "";
        log_scale = false;

        batch_size = 10000;
        max_em_iter = 100;
        em_tol = 1e-4;
        kappa_max = 100.;

        verbose = false;
        randomize_init = false;
        do_standardize = false;
    }

    std::string mtx_file;

    bool log_scale;

    Index batch_size;
    Index max_em_iter;
    Scalar em_tol;

    bool verbose;
    Scalar kappa_max;
    bool randomize_init;
    bool do_standardize;
};

template <typename STAT, typename DATA, typename OPTIONS>
auto
train_model(std::vector<std::shared_ptr<STAT>> &stat_vector,
            DATA &data_loader,
            const OPTIONS &options)
{

    Index batch_size = options.batch_size;
    const Index max_em_iter = options.max_em_iter;
    Vec score_trace(max_em_iter);

    const Index D = data_loader.num_rows();
    const Index N = data_loader.num_columns();

    TLOG("D = " << D << ", N = " << N);

    //////////////////
    // build models //
    //////////////////

    const std::size_t num_annot = stat_vector.size();
    std::vector<std::shared_ptr<annotation_model_t>> model_vector;

    Vec nnz;

    for (Index a = 0; a < num_annot; ++a) {

        annotation_stat_t &stat = *stat_vector.at(a).get();

        if (a == 0) {
            nnz.resize(stat.L.rows());
            nnz.setZero();
        }

        nnz += stat.L * Mat::Ones(stat.L.cols(), 1) +
            stat.L0.cwiseAbs() * Mat::Ones(stat.L0.cols(), 1);

        for (SpMat::InnerIterator it(stat.Lqc, 0); it; ++it) {
            const Index g = it.col();
            nnz(g) += 1;
        }
    }

    std::vector<Index> subrow;
    for (Index r = 0; r < nnz.size(); ++r) {
        if (nnz(r) > 0) {
            subrow.emplace_back(r);
        }
    }

    Index M = 0, K = 0;

    for (Index a = 0; a < num_annot; ++a) {
        annotation_stat_t &stat = *stat_vector.at(a).get();
        stat.squeeze(subrow);
        model_vector.emplace_back(
            std::make_shared<annotation_model_t>(stat.L,
                                                 stat.L0,
                                                 options.kappa_max));
        if (stat.L.rows() > M)
            M = stat.L.rows();
        if (stat.L.cols() > K)
            K = stat.L.cols();
    }

    /////////////////////////////
    // Initial Q/C  assignment //
    /////////////////////////////

    std::unordered_set<Index> taboo;

    for (Index a = 0; a < num_annot; ++a) {
        annotation_stat_t &stat = *stat_vector.at(a).get();
        SpMat Lqc = stat.Lqc.transpose();

        if (Lqc.cwiseAbs().sum() > 0) {

            for (Index lb = 0; lb < N; lb += batch_size) {
                const Index ub = std::min(N, batch_size + lb);

                const Mat xx = data_loader(lb, ub, subrow);

                for (Index j = 0; j < xx.cols(); ++j) {
                    const Index i = j + lb;
                    Vec xj = xx.col(j);

                    for (SpMat::InnerIterator it(Lqc, 0); it; ++it) {

                        const Index g = it.col();
                        const Scalar v = it.value();

                        if (xj(g) < it.value()) {
                            taboo.insert(i);
                        }
                    }
                }
            }
            TLOG("Found " << taboo.size() << " disqualified");
        }
    }

    //////////////////////////////////////////
    // standardization mean and inverse std //
    //////////////////////////////////////////

    ASSERT(subrow.size() == M, "subrow and rows(M) disagree");

    ///////////////////////////////////////////////////
    // Give contexts to annotation type-specific way //
    ///////////////////////////////////////////////////

    Mat C = Mat::Zero(M, num_annot);

    is_positive_op<Mat> _obs;

    for (Index a = 0; a < num_annot; ++a) {
        annotation_stat_t &stat = *stat_vector.at(a).get();
        C.col(a) = (stat.L * Mat::Ones(stat.L.cols(), 1)).unaryExpr(_obs);
    }

    ///////////////////////////////
    // Initial greedy assignment //
    ///////////////////////////////

    std::vector<Index> membership(N);
    std::fill(membership.begin(), membership.end(), -1);

    Scalar score_init = 0;

    auto initialization = [&](bool be_greedy = true) {
        Vec xj(M), xja(M);

        for (Index lb = 0; lb < N; lb += batch_size) {
            const Index ub = std::min(N, batch_size + lb);

            Mat xx = data_loader(lb, ub, subrow);

            discrete_sampler_t sampler_k(K); // sample discrete from log-mass
            Vec sj(K);

            for (Index j = 0; j < xx.cols(); ++j) {

                const Index i = j + lb;
                if (taboo.count(i) > 0)
                    continue;

                sj.setZero();

                if (options.do_standardize) {
                    xj /= xj.norm();
                } else {
                    xj = xx.col(j);
                }

                if (be_greedy) {
                    for (Index a = 0; a < num_annot; ++a) {
                        annotation_model_t &annot = *model_vector.at(a).get();
                        xja = xj.cwiseProduct(C.col(a));
                        normalize_columns(xja);
                        sj += annot.log_score(xja);
                    }
                }

                const Index k = sampler_k(sj);
                score_init += sj(k);

                if (xx.col(j).sum() > 0) {
                    for (Index a = 0; a < num_annot; ++a) {
                        annotation_stat_t &stat = *stat_vector.at(a).get();
                        xja = xj.cwiseProduct(C.col(a));
                        normalize_columns(xja);

                        stat.nsize(k) += 1.0;

                        stat.Stat.col(k) += xja.cwiseProduct(stat.L.col(k));
                        stat.Stat_anti.col(k) +=
                            xja.cwiseProduct(stat.L0.col(k));

                        stat.unc_stat.col(k) += xja;
                        stat.unc_stat_anti.col(k) += xja;
                    }

                    membership[i] = k;
                } else {
                    taboo.insert(i);
                }
            }

            if (options.verbose) {
                annotation_stat_t &stat = *stat_vector.at(0).get();
                std::vector<std::string> &labels = stat.labels;
                Rcpp::Rcerr << std::setw(10) << lb;
                for (Index k = 0; k < K; ++k) {
                    Rcpp::Rcerr << " [" << labels[k] << "] " << std::setw(10)
                                << stat.nsize(k);
                }
                Rcpp::Rcerr << std::endl;
            }
        }

        score_init /= static_cast<Scalar>(N);

        for (Index a = 0; a < num_annot; ++a) {
            annotation_stat_t &stat = *stat_vector.at(a).get();
            annotation_model_t &annot = *model_vector.at(a).get();
            annot.update_param(stat.Stat, stat.Stat_anti, stat.nsize);
        }
    };

    ////////////////////////////
    // Memoized online update //
    ////////////////////////////

    auto score_diff = [&options](const Index iter, const Vec &trace) -> Scalar {
        Scalar diff = std::abs(trace(iter));

        if (iter > 4) {
            Scalar score_old = trace.segment(iter - 3, 2).sum();
            Scalar score_new = trace.segment(iter - 1, 2).sum();
            diff = std::abs(score_old - score_new) /
                (std::abs(score_old) + options.em_tol);
        } else if (iter > 0) {
            diff = std::abs(trace(iter - 1) - trace(iter)) /
                (std::abs(trace(iter) + options.em_tol));
        }

        return diff;
    };

    std::vector<Scalar> em_score_out;

    auto monte_carlo_update = [&]() {
        Scalar score = score_init;

        Vec xj(M), xja(M);

        for (Index iter = 0; iter < max_em_iter; ++iter) {
            score = 0;

            // #pragma omp parallel for
            for (Index lb = 0; lb < N; lb += batch_size) {     // batch
                const Index ub = std::min(N, batch_size + lb); //

                discrete_sampler_t sampler_k(K);
                Vec sj(K); // type x 1 score vector

                const Mat xx = data_loader(lb, ub, subrow);

                for (Index j = 0; j < xx.cols(); ++j) {

                    const Index i = j + lb;

                    if (taboo.count(i) > 0)
                        continue;

                    const Index k_prev = membership[i];

                    sj.setZero();
                    if (options.do_standardize) {
                        xj /= xj.norm();
                    } else {
                        xj = xx.col(j);
                    }

                    for (Index a = 0; a < num_annot; ++a) {
                        annotation_model_t &annot = *model_vector.at(a).get();
                        xja = xj.cwiseProduct(C.col(a));
                        normalize_columns(xja);
                        sj += annot.log_score(xja);
                    }

                    const Index k_now = sampler_k(sj);
                    score += sj(k_now);

                    if (k_now != k_prev) {

                        for (Index a = 0; a < num_annot; ++a) {
                            annotation_stat_t &stat = *stat_vector.at(a).get();

                            xja = xj.cwiseProduct(C.col(a));
                            normalize_columns(xja);

                            stat.nsize(k_prev) -= 1.0;
                            stat.nsize(k_now) += 1.0;

                            stat.Stat.col(k_prev) -=
                                xja.cwiseProduct(stat.L.col(k_prev));
                            stat.Stat.col(k_now) +=
                                xja.cwiseProduct(stat.L.col(k_now));

                            stat.Stat_anti.col(k_prev) -=
                                xja.cwiseProduct(stat.L0.col(k_prev));
                            stat.Stat_anti.col(k_now) +=
                                xja.cwiseProduct(stat.L0.col(k_now));

                            stat.unc_stat.col(k_prev) -= xja;
                            stat.unc_stat.col(k_now) += xja;

                            stat.unc_stat_anti.col(k_prev) -= xja;
                            stat.unc_stat_anti.col(k_now) += xja;
                        }

                        membership[i] = k_now;
                    }

                } // end of data iteration

                if (options.verbose) {
                    annotation_stat_t &stat = *stat_vector.at(0).get();
                    std::vector<std::string> &labels = stat.labels;
                    Rcpp::Rcerr << std::setw(10) << lb;
                    for (Index k = 0; k < K; ++k) {
                        Rcpp::Rcerr << " [" << labels[k] << "] "
                                    << std::setw(10) << stat.nsize(k);
                    }
                    Rcpp::Rcerr << std::endl;
                }

            } // end of batch iteration

            for (Index a = 0; a < num_annot; ++a) {
                annotation_stat_t &stat = *stat_vector.at(a).get();
                annotation_model_t &annot = *model_vector.at(a).get();
                annot.update_param(stat.Stat, stat.Stat_anti, stat.nsize);
            }

            score = score / static_cast<Scalar>(N);
            score_trace(iter) = score;

            Scalar diff = score_diff(iter, score_trace);
            TLOG("Iter [" << iter << "] score = " << score
                          << ", diff = " << diff);

            if (iter > 4 && diff < options.em_tol) {

                TLOG("Converged < " << options.em_tol);

                for (Index t = 0; t <= iter; ++t) {
                    em_score_out.emplace_back(score_trace(t));
                }
                break;
            }
        } // end of EM iteration
    };

    if (options.randomize_init) {
        TLOG("Start randomized initialization");
        initialization(false);
        TLOG("Finished randomized initialization");
    } else {
        TLOG("Start greedy initialization");
        initialization(true);
        TLOG("Finished greedy initialization");
    }

    TLOG("Start training marker gene profiles");
    monte_carlo_update();
    TLOG("Finished training the main assignment model");

    Vec sj(K);
    Vec xj(M), xja(M);
    Vec zi(K);

    std::vector<Scalar> max_prob_vec;
    std::vector<Scalar> max_score_vec;
    std::vector<std::string> argmax_prob_vec;
    max_prob_vec.reserve(N);
    max_score_vec.reserve(N);
    argmax_prob_vec.reserve(N);

    Mat Pr(K, N);
    Pr.setZero();

    annotation_stat_t &stat = *stat_vector.at(0).get();
    std::vector<std::string> &labels = stat.labels;

    for (Index lb = 0; lb < N; lb += batch_size) {
        const Index ub = std::min(N, batch_size + lb);

        const Mat xx = data_loader(lb, ub, subrow);

        for (Index j = 0; j < xx.cols(); ++j) {
            const Index i = j + lb;
            if (taboo.count(i) > 0) {
                max_prob_vec.emplace_back(0);
                max_score_vec.emplace_back(-0);
                argmax_prob_vec.emplace_back("Incomplete");
                continue;
            }

            sj.setZero();

            if (options.do_standardize) {
                xj /= xj.norm();
            } else {
                xj = xx.col(j);
            }

            for (Index a = 0; a < num_annot; ++a) {
                annotation_model_t &annot = *model_vector.at(a).get();
                xja = xj.cwiseProduct(C.col(a));
                normalize_columns(xja);
                sj += annot.log_score(xja);
            }

            normalized_exp(sj, zi);

            Index argmax, argmin;
            const Scalar smax = sj.maxCoeff(&argmax);
            const Scalar smin = sj.maxCoeff(&argmin);
            Pr.col(i) = zi;

            max_prob_vec.emplace_back(zi(argmax));
            max_score_vec.emplace_back(smax);
            argmax_prob_vec.emplace_back(labels[argmax]);
        }
        TLOG("Annotated on the batch [" << lb << ", " << ub << ")");
    }

    return std::make_tuple<>(model_vector,
                             argmax_prob_vec,
                             max_prob_vec,
                             max_score_vec,
                             Pr);
}

#endif
