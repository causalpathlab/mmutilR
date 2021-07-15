#include "mmutil.hh"
#include "mmutil_stat.hh"

#ifndef MMUTIL_NORMALIZE_HH_
#define MMUTIL_NORMALIZE_HH_

// 0. Run the stat-visitor on the columns
// 1. E <- S1 / median(S1)
// 2. E <- S2 / (E)
// 3. tau <- mean(E)
// 4. E <- sqrt(E + tau)

struct col_data_normalizer_t {
    using index_t = Index;
    using scalar_t = Scalar;

    explicit col_data_normalizer_t(const std::string _target,
                                   const std::vector<scalar_t> &_scale)
        : target_filename(_target)
        , col_scale(_scale)
    {
        elem_check = 0;
        max_row = 0;
        max_col = 0;
        max_elem = 0;
    }

    void eval_after_header(const index_t r, const index_t c, const index_t e)
    {
        std::tie(max_row, max_col, max_elem) = std::make_tuple(r, c, e);
        ASSERT(max_col <= col_scale.size(), "Insufficient #denominators");

        elem_check = 0;

        ofs.open(target_filename.c_str(), std::ios::out);
        ofs << "%%MatrixMarket matrix coordinate integer general" << std::endl;
        ofs << max_row << FS << max_col << FS << max_elem << std::endl;
    }

    void eval(const index_t row, const index_t col, const scalar_t weight)
    {
        if (row < max_row && col < max_col) {
            const index_t i = row + 1; // fix zero-based to one-based
            const index_t j = col + 1; // fix zero-based to one-based
            const scalar_t new_weight = weight * col_scale.at(col);
            ofs << i << FS << j << FS << new_weight << std::endl;
            elem_check++;
        }
    }

    void eval_end_of_file()
    {
        ofs.close();
#ifdef DEBUG
        // check
#endif
    }

    const std::string target_filename;
    const std::vector<scalar_t> &col_scale;

    index_t max_row;
    index_t max_col;
    index_t max_elem;
    index_t elem_check;

    ogzstream ofs;
    static constexpr char FS = ' ';
};

void
write_normalized(const std::string mtx_file, // input file
                 const std::string out_file, // output file
                 const Scalar tau_scale)
{
    col_stat_collector_t collector;
    visit_matrix_market_file(mtx_file, collector);

    const Vec &s1 = collector.Col_S1;
    const Vec &s2 = collector.Col_S2;
    std::vector<Scalar> _s1 = std_vector(s1);
    const Scalar _med = std::max(_s1[_s1.size() / 2], static_cast<Scalar>(1.0));

    Vec Deg(s1.size());
    Deg = s2.cwiseQuotient((s1 / _med).cwiseProduct(s1 / _med));

    const Scalar tau = tau_scale * Deg.mean();

    const Vec DegSqrtInv = Deg.unaryExpr([&tau](const Scalar dd) {
        // 1 / sqrt(dd + tau)
        const Scalar _one = 1.0;
        return _one / std::max(_one, std::sqrt(dd + tau));
    });

    std::vector<Scalar> col_scale(DegSqrtInv.size());
    std_vector(DegSqrtInv, col_scale);

    col_data_normalizer_t normalizer(out_file, col_scale);
    visit_matrix_market_file(mtx_file, normalizer);
}

template <typename Derived>
inline SpMat
normalize_to_median(const Eigen::SparseMatrixBase<Derived> &xx)
{
    const Derived &X = xx.derived();

    ///////////////////////
    // degree of columns //
    ///////////////////////

    const Vec deg = X.transpose() * Mat::Ones(X.rows(), 1);

    Index _argmin, _argmax;
    const Scalar _min_deg = deg.minCoeff(&_argmin);
    const Scalar _max_deg = deg.maxCoeff(&_argmax);

    std::vector<typename Derived::Scalar> _deg = std_vector(deg);
#ifdef DEBUG
    TLOG("search the median degree in [" << _min_deg << ", " << _max_deg
                                         << ")");
#endif
    std::nth_element(_deg.begin(), _deg.begin() + _deg.size() / 2, _deg.end());
    const Scalar median =
        std::max(_deg[_deg.size() / 2], static_cast<Scalar>(1.0));

    TLOG("Targeting the median degree " << median);

    const Vec degInverse = deg.unaryExpr([&median](const Scalar x) {
        const Scalar _one = 1.0;
        return median / std::max(x, _one);
    });

    SpMat ret(X.rows(), X.cols());
    ret = X * degInverse.asDiagonal();

    return ret;
}

template <typename Derived>
inline SpMat
normalize_to_fixed(const Eigen::SparseMatrixBase<Derived> &xx,
                   const float target)
{
    const Derived &X = xx.derived();

    ///////////////////////
    // degree of columns //
    ///////////////////////

    const Vec deg = X.transpose() * Mat::Ones(X.rows(), 1);
    const Vec degInverse = deg.unaryExpr([&target](const Scalar x) {
        const Scalar _one = 1.0;
        return target / std::max(x, _one);
    });

    SpMat ret(X.rows(), X.cols());
    ret = X * degInverse.asDiagonal();

    return ret;
}

template <typename Derived>
inline Mat
scale_by_degree(const Eigen::MatrixBase<Derived> &xx, const Scalar _tau)
{
    const Derived &X = xx.derived();

    const Index D = X.rows();
    const Mat Deg = X.cwiseProduct(X).transpose() * Mat::Ones(D, 1);

    const Scalar tau = Deg.mean() * _tau;
    const Mat dd = Deg.unaryExpr([&tau](const Scalar x) {
        const Scalar _one = 1.0;
        return _one / std::max(_one, std::sqrt(x + tau));
    });

    Mat ret = X * dd.asDiagonal();
    return ret;
}

#endif
