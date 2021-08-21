#include "mmutil.hh"
#include "math.hh"

#include <cmath>
#include <unordered_map>

#ifndef MMUTIL_STAT_HH_
#define MMUTIL_STAT_HH_

//////////////////////////////////////////////////////
// e.g.,					    //
//   row_stat_collector_t collector;		    //
//   visit_matrix_market_file(filename, collector); //
//////////////////////////////////////////////////////

struct row_stat_collector_t {
    using index_t = Index;
    using scalar_t = Scalar;

    explicit row_stat_collector_t()
    {
        max_row = 0;
        max_col = 0;
        max_elem = 0;
    }

    void eval_after_header(const index_t r, const index_t c, const index_t e)
    {
        std::tie(max_row, max_col, max_elem) = std::make_tuple(r, c, e);

#ifdef DEBUG
        TLOG("Start reading a list of triplets");
#endif

        Row_S1.resize(max_row);
        Row_S1.setZero();
        Row_S2.resize(max_row);
        Row_S2.setZero();
        Row_N.resize(max_row);
        Row_N.setZero();
    }

    void eval(const index_t row, const index_t col, const scalar_t weight)
    {
        if (row < max_row && col < max_col) {
            Row_S1(row) += weight;
            Row_S2(row) += (weight * weight);
            if (std::abs(weight) > 1e-8)
                Row_N(row)++;
        }
    }

    void eval_end_of_file()
    {
#ifdef DEBUG
        TLOG("Finished reading a list of triplets");
#endif
    }

    Index max_row;
    Index max_col;
    Index max_elem;
    Vec Row_S1;
    Vec Row_S2;
    IntVec Row_N;
};

struct col_stat_collector_t {
    using index_t = Index;
    using scalar_t = Scalar;

    explicit col_stat_collector_t()
    {
        max_row = 0;
        max_col = 0;
        max_elem = 0;
    }

    void eval_after_header(const index_t r, const index_t c, const index_t e)
    {
        std::tie(max_row, max_col, max_elem) = std::make_tuple(r, c, e);

#ifdef DEBUG
        TLOG("Start reading a list of triplets");
#endif

        Col_S1.resize(max_col);
        Col_S1.setZero();
        Col_S2.resize(max_col);
        Col_S2.setZero();
        Col_N.resize(max_col);
        Col_N.setZero();
    }

    void eval(const index_t row, const index_t col, const scalar_t weight)
    {
        if (row < max_row && col < max_col) {
            Col_S1(col) += weight;
            Col_S2(col) += (weight * weight);
            if (std::abs(weight) > 1e-8)
                Col_N(col)++;
        }
#ifdef DEBUG
        else {
            TLOG("[" << row << ", " << col << ", " << weight << "]");
            TLOG(max_row << " x " << max_col);
        }
#endif
    }

    void eval_end_of_file()
    {
#ifdef DEBUG
        TLOG("S1  : " << Col_S1.sum());
        TLOG("S2  : " << Col_S2.sum());
        TLOG("NNZ : " << Col_N.sum());
        TLOG("Finished reading a list of triplets");
#endif
    }

    Index max_row;
    Index max_col;
    Index max_elem;
    Vec Col_S1;
    Vec Col_S2;
    IntVec Col_N;
};

struct row_col_stat_collector_t {
    using index_t = Index;
    using scalar_t = Scalar;

    explicit row_col_stat_collector_t()
    {
        max_row = 0;
        max_col = 0;
        max_elem = 0;
    }

    void eval_after_header(const index_t r, const index_t c, const index_t e)
    {
        std::tie(max_row, max_col, max_elem) = std::make_tuple(r, c, e);

#ifdef DEBUG
        TLOG("Start reading a list of triplets");
#endif

        Row_S1.resize(max_row);
        Row_S1.setZero();
        Row_S2.resize(max_row);
        Row_S2.setZero();
        Row_N.resize(max_row);
        Row_N.setZero();

        Col_S1.resize(max_col);
        Col_S1.setZero();
        Col_S2.resize(max_col);
        Col_S2.setZero();
        Col_N.resize(max_col);
        Col_N.setZero();
    }

    void eval(const index_t row, const index_t col, const scalar_t weight)
    {
        if (row < max_row && col < max_col) {
            Row_S1(row) += weight;
            Row_S2(row) += (weight * weight);
            if (std::abs(weight) > 1e-8)
                Row_N(row)++;

            Col_S1(col) += weight;
            Col_S2(col) += (weight * weight);
            if (std::abs(weight) > 1e-8)
                Col_N(col)++;
        }
    }

    void eval_end_of_file()
    {
#ifdef DEBUG
        TLOG("Finished reading a list of triplets");
#endif
    }

    Index max_row;
    Index max_col;
    Index max_elem;

    Vec Row_S1;
    Vec Row_S2;
    IntVec Row_N;

    Vec Col_S1;
    Vec Col_S2;
    IntVec Col_N;
};

struct histogram_collector_t {
    using index_t = Index;
    using scalar_t = Scalar;

    explicit histogram_collector_t()
    {
        max_row = 0;
        max_col = 0;
        max_elem = 0;
    }

    void eval_after_header(const index_t r, const index_t c, const index_t e)
    {
        std::tie(max_row, max_col, max_elem) = std::make_tuple(r, c, e);

#ifdef DEBUG
        TLOG("Start reading a list of triplets");
#endif
    }

    void eval(const index_t row, const index_t col, const scalar_t weight)
    {
        if (row < max_row && col < max_col) {
            const index_t k = std::lround(weight);
            if (freq_map.count(k) == 0)
                freq_map[k] = 0;
            freq_map[k]++;
        }
    }

    void eval_end_of_file() {}

    Index max_row;
    Index max_col;
    Index max_elem;
    std::unordered_map<Index, Index> freq_map;
};

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

struct cf_index_sampler_t {

    using DS = discrete_sampler_t;

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
