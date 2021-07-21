#include "mmutil_normalize.hh"

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
