#ifndef RCPP_UTIL_HH_
#define RCPP_UTIL_HH_

#include <vector>
#include <string>
#include "util.hh"
#include "io.hh"

namespace rcpp { namespace util {

std::vector<std::string> copy(const Rcpp::StringVector &r_vec);

void copy(const Rcpp::StringVector &r_vec, std::vector<std::string> &vec);

template <typename Derived>
Rcpp::NumericMatrix
named(const Eigen::MatrixBase<Derived> &xx,
      const std::vector<std::string> &out_row_names,
      const std::vector<std::string> &out_col_names)
{
    Rcpp::NumericMatrix x = Rcpp::wrap(xx);
    if (xx.rows() == out_row_names.size()) {
        Rcpp::rownames(x) = Rcpp::wrap(out_row_names);
    }
    if (xx.cols() == out_col_names.size()) {
        Rcpp::colnames(x) = Rcpp::wrap(out_col_names);
    }
    return x;
}

template <typename Derived>
Rcpp::NumericMatrix
named_rows(const Eigen::MatrixBase<Derived> &xx,
           const std::vector<std::string> &out_row_names)
{
    Rcpp::NumericMatrix x = Rcpp::wrap(xx);
    if (xx.rows() == out_row_names.size()) {
        Rcpp::rownames(x) = Rcpp::wrap(out_row_names);
    }
    return x;
}

template <typename T>
void
convert_r_index(const std::vector<T> &cvec, std::vector<T> &rvec)
{
    rvec.resize(cvec.size());
    auto r_index = [](const T x) -> T { return x + 1; };
    std::transform(cvec.begin(), cvec.end(), rvec.begin(), r_index);
}

template <typename T>
std::vector<T>
convert_r_index(const std::vector<T> &cvec)
{
    std::vector<T> rvec(cvec.size());
    auto r_index = [](const T x) -> T { return x + 1; };
    std::transform(cvec.begin(), cvec.end(), rvec.begin(), r_index);
    return rvec;
}

template <typename T>
void
convert_c_index(const std::vector<T> &rvec, std::vector<T> &cvec)
{
    cvec.resize(rvec.size());
    auto c_index = [](const T x) -> T { return x - 1; };
    std::transform(rvec.begin(), rvec.end(), cvec.begin(), c_index);
}

template <typename T>
void
convert_c_index(const Rcpp::IntegerVector &r_vec, std::vector<T> &cvec)
{
    std::vector<T> rvec = Rcpp::as<std::vector<T>>(r_vec);
    cvec.resize(rvec.size());
    auto c_index = [](const T x) -> T { return x - 1; };
    std::transform(rvec.begin(), rvec.end(), cvec.begin(), c_index);
}

template <typename Derived>
Rcpp::List
build_sparse_list(const Eigen::SparseMatrixBase<Derived> &_A)
{
    using Scalar = typename Derived::Scalar;
    using Index = typename Derived::Index;

    const Derived &A = _A.derived();

    std::vector<Index> ii, jj;
    std::vector<Scalar> xx;
    ii.reserve(A.nonZeros());
    jj.reserve(A.nonZeros());
    xx.reserve(A.nonZeros());

    for (Index j = 0; j < A.outerSize(); ++j) {
        for (typename Derived::InnerIterator it(A, j); it; ++it) {
            const Index i = it.index();
            const Scalar x = it.value();

            ii.emplace_back(i);
            jj.emplace_back(j);
            xx.emplace_back(x);
        }
    }

    return Rcpp::List::create(convert_r_index(ii), convert_r_index(jj), xx);
}

template <typename Derived>
void
build_sparse_mat(const Rcpp::List &in_list,
                 const std::size_t nrow,
                 const std::size_t ncol,
                 Eigen::SparseMatrixBase<Derived> &ret_)
{
    using Scalar = typename Derived::Scalar;

    Derived &ret = ret_.derived();
    ret.resize(nrow, ncol);
    ret.setZero();
    std::vector<Eigen::Triplet<Scalar>> triples;

    if (in_list.size() == 3) {
        const std::vector<std::size_t> &ii = in_list[0];
        const std::vector<std::size_t> &jj = in_list[1];
        const std::vector<Scalar> &kk = in_list[2];
        const std::size_t m = ii.size();

        if (jj.size() == m && kk.size() == m) {
            triples.reserve(m);
            for (std::size_t e = 0; e < m; ++e) {
                const std::size_t i = ii.at(e), j = jj.at(e);
                if (i < 1 || j < 1) {
                    ELOG("Assume R's 1-based indexing");
                    continue;
                }
                if (i <= nrow && j <= ncol) {
                    // 1-based -> 0-based
                    triples.emplace_back(
                        Eigen::Triplet<Scalar>(i - 1, j - 1, kk.at(e)));
                }
            }
        } else {
            WLOG("input list sizes don't match");
        }
    } else if (in_list.size() == 2) {

        const std::vector<std::size_t> &ii = in_list[0];
        const std::vector<std::size_t> &jj = in_list[1];
        const std::size_t m = ii.size();

        if (jj.size() == m) {
            triples.reserve(m);
            for (std::size_t e = 0; e < m; ++e) {
                const std::size_t i = ii.at(e), j = jj.at(e);
                if (i < 1 || j < 1) {
                    ELOG("Assume R's 1-based indexing");
                    continue;
                }
                if (i <= nrow && j <= ncol) {
                    // 1-based -> 0-based
                    triples.emplace_back(
                        Eigen::Triplet<Scalar>(i - 1, j - 1, 1.));
                }
            }
        } else {
            WLOG("input list sizes don't match");
        }
    } else {
        WLOG("Need two or three vectors in the list");
    }

    ret.reserve(triples.size());
    ret.setFromTriplets(triples.begin(), triples.end());
}

template <typename S, typename I>
void
take_common_names(const std::vector<S> &name_files,
                  std::vector<S> &pos2name,
                  std::unordered_map<S, I> &name2pos,
                  bool take_union = false,
                  const std::size_t MAX_WORD = 2,
                  const char WORD_SEP = '_')
{
    if (take_union) {
        std::unordered_set<S> _names; // Take a unique set

        auto _insert = [&](S f) {
            std::vector<S> vv;
            CHECK(read_line_file(f, vv, MAX_WORD, WORD_SEP));
            const std::size_t sz = vv.size();

            std::sort(vv.begin(), vv.end());
            vv.erase(std::unique(vv.begin(), vv.end()), vv.end());
            WLOG_(vv.size() < sz, "Duplicate in \"" << f << "\"");

            for (auto r : vv)
                _names.insert(r);
        };

        std::for_each(name_files.begin(), name_files.end(), _insert);
        pos2name.reserve(_names.size());
        std::copy(_names.begin(), _names.end(), std::back_inserter(pos2name));

    } else {

        const std::size_t B = name_files.size();
        std::unordered_map<S, std::size_t> nn;

        for (std::size_t b = 0; b < B; ++b) {
            std::vector<S> vv;
            CHECK(read_line_file(name_files[b], vv, MAX_WORD, WORD_SEP));

            const std::size_t sz = vv.size();
            std::sort(vv.begin(), vv.end());
            vv.erase(std::unique(vv.begin(), vv.end()), vv.end());
            WLOG_(vv.size() < sz,
                  "Duplicate in \"" << name_files.at(b) << "\"");

            for (S x : vv) {
                if (nn.count(x) == 0) {
                    nn[x] = 1;
                } else {
                    nn[x] = nn[x] + 1;
                }
            }
        }

        pos2name.reserve(nn.size());

        for (auto &it : nn) {
            if (it.second >= B) {
                pos2name.emplace_back(it.first);
            }
        }
    }

    std::sort(pos2name.begin(), pos2name.end());
    for (I r = 0; r < pos2name.size(); ++r) {
        const S &s = pos2name.at(r);
        name2pos[s] = r;
    }
}

}} // namespace rcpp::util

#endif
