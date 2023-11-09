#include "mmutil_match.hh"

void
normalize_weights(const Index deg_i,
                  std::vector<float> &dist,
                  std::vector<float> &weights)
{
    if (deg_i < 2) {
        weights[0] = 1.;
        return;
    }

    const float _log2 = fasterlog(2.);
    const float _di = static_cast<float>(deg_i);
    const float log2K = fasterlog(_di) / _log2;

    float lambda = 10.0;

    const float dmin = *std::min_element(dist.begin(), dist.begin() + deg_i);

    // Find lambda values by a simple line-search
    auto f = [&](const float lam) -> float {
        float rhs = 0.;
        for (Index j = 0; j < deg_i; ++j) {
            float w = fasterexp(-(dist[j] - dmin) * lam);
            rhs += w;
        }
        float lhs = log2K;
        return (lhs - rhs);
    };

    float fval = f(lambda);

    const Index max_iter = 100;

    for (Index iter = 0; iter < max_iter; ++iter) {
        float _lam = lambda;
        if (fval < 0.) {
            _lam = lambda * 1.1;
        } else {
            _lam = lambda * 0.9;
        }
        float _fval = f(_lam);
        if (std::abs(_fval) > std::abs(fval)) {
            break;
        }
        lambda = _lam;
        fval = _fval;
    }

    for (Index j = 0; j < deg_i; ++j) {
        weights[j] = fasterexp(-(dist[j] - dmin) * lambda);
    }
}

inline std::tuple<std::unordered_set<Index>, Index>
find_nz_cols(const std::string mtx_file)
{
    col_stat_collector_t collector;
    visit_matrix_market_file(mtx_file, collector);
    std::unordered_set<Index> valid;
    const IntVec &nn = collector.Col_N;
    for (Index j = 0; j < nn.size(); ++j) {
        if (nn(j) > 0.0)
            valid.insert(j);
    }
    const Index N = collector.max_col;
    return std::make_tuple(valid, N); // RVO
}

inline std::tuple<std::unordered_set<Index>, // valid
                  Index,                     // #total
                  std::vector<std::string>   // names
                  >
find_nz_col_names(const std::string mtx_file,
                  const std::string col_file,
                  const std::size_t MAX_COL_WORD = 100,
                  const char COL_WORD_SEP = '@')
{
    using valid_set_t = std::unordered_set<Index>;
    valid_set_t valid;
    Index N;

    std::tie(valid, N) = find_nz_cols(mtx_file);

    std::vector<std::string> col_names;

    if (file_exists(col_file)) {
        // CHECK(read_vector_file(col_file, col_names));
        CHECK(read_line_file(col_file, col_names, MAX_COL_WORD, COL_WORD_SEP));
        ASSERT(col_names.size() >= N,
               "Not enough # names in `find_nz_col_names`");
    } else {
        for (Index j = 0; j < N; ++j) {
            col_names.push_back(std::to_string(j + 1));
        }
    }

    return std::make_tuple(valid, N, col_names);
}
