#include "mmutil.hh"

std::tuple<Index, Index, Scalar>
parse_triplet(const std::tuple<Index, Index, Scalar> &tt)
{
    return tt;
}

std::tuple<Index, Index, Scalar>
parse_triplet(const Eigen::Triplet<Scalar> &tt)
{
    return std::make_tuple(tt.row(), tt.col(), tt.value());
}

std::vector<std::string>
copy(const Rcpp::StringVector &r_vec)
{
    std::vector<std::string> vec;
    vec.reserve(r_vec.size());
    for (Index j = 0; j < r_vec.size(); ++j) {
        vec.emplace_back(r_vec(j));
    }
    return vec;
}
