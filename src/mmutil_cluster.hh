#include "mmutil.hh"
#include "mmutil_util.hh"
#include "mmutil_index.hh"
#include "mmutil_io.hh"
#include "check.hh"
#include "math.hh"

#ifndef MMUTIL_CLUSTER_HH_
#define MMUTIL_CLUSTER_HH_

namespace mmutil { namespace cluster {

//////////////////////////
// helper type checking //
//////////////////////////

struct dim_t : public check_positive_t<Scalar> {
    explicit dim_t(const Scalar v)
        : check_positive_t<Scalar>(v)
    {
    }
};

struct pseudo_count_t : public check_positive_t<Scalar> {
    explicit pseudo_count_t(const Scalar v)
        : check_positive_t<Scalar>(v)
    {
    }
};

struct num_clust_t : public check_positive_t<Index> {
    explicit num_clust_t(const Index n)
        : check_positive_t<Index>(n)
    {
    }
};
struct num_sample_t : public check_positive_t<Index> {
    explicit num_sample_t(const Index n)
        : check_positive_t<Index>(n)
    {
    }
};

//////////////////////
// helper functions //
//////////////////////

template <typename T>
inline std::vector<T>
count_frequency(std::vector<T> &_membership, const T cutoff = 0)
{
    const auto N = _membership.size();

    const T kk = *std::max_element(_membership.begin(), _membership.end()) + 1;

    std::vector<T> _sz(kk, 0);

    if (kk < 1) {
        return _sz;
    }

    for (std::size_t j = 0; j < N; ++j) {
        const T k = _membership.at(j);
        if (k >= 0)
            _sz[k]++;
    }

    return _sz;
}

template <typename T>
inline T
sort_cluster_index(std::vector<T> &_membership, const T cutoff = 0)
{
    const auto N = _membership.size();
    std::vector<T> _sz = count_frequency(_membership, cutoff);
    const T kk = _sz.size();
    auto _order = std_argsort(_sz);

    std::vector<T> rename(kk, -1);
    T k_new = 0;
    for (T k : _order) {
        if (_sz.at(k) >= cutoff)
            rename[k] = k_new++;
    }

    for (std::size_t j = 0; j < N; ++j) {
        const T k_old = _membership.at(j);
        const T k_new = rename.at(k_old);
        _membership[j] = k_new;
    }

    return k_new;
}

template <typename T, typename OFS>
void
print_histogram(const std::vector<T> &nn, //
                OFS &ofs,                 //
                const T height = 50.0,    //
                const T cutoff = .01,     //
                const int ntop = 10)
{
    using std::accumulate;
    using std::ceil;
    using std::floor;
    using std::round;
    using std::setw;

    const Scalar ntot = (nn.size() <= ntop) ?
        (accumulate(nn.begin(), nn.end(), 1e-8)) :
        (accumulate(nn.begin(), nn.begin() + ntop, 1e-8));

    ofs << std::endl;

    auto _print = [&](const Index j) {
        const Scalar x = nn.at(j);
        ofs << setw(10) << (j + 1) << " [" << setw(10) << round(x) << "] ";
        for (int i = 0; i < ceil(x / ntot * height); ++i)
            ofs << "*";
        ofs << std::endl;
    };

    auto _args = std_argsort(nn);

    if (_args.size() <= ntop) {
        std::for_each(_args.begin(), _args.end(), _print);
    } else {
        std::for_each(_args.begin(), _args.begin() + ntop, _print);
    }

    ofs << std::endl;
}

inline std::vector<Index>
random_membership(const num_clust_t num_clust, //
                  const num_sample_t num_sample)
{
    std::random_device rd {};
    std::mt19937 gen { rd() };
    const Index k = num_clust.val;
    const Index n = num_sample.val;

    std::uniform_int_distribution<Index> runifK { 0, k - 1 };
    std::vector<Index> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::vector<Index> ret;
    ret.reserve(n);

    std::transform(idx.begin(),
                   idx.end(),
                   std::back_inserter(ret),
                   [&](const Index i) { return runifK(gen); });

    return ret;
}

}} // end of namespace

#endif
