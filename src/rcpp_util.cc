#include "rcpp_util.hh"

namespace rcpp { namespace util {

std::vector<std::string>
copy(const Rcpp::StringVector &r_vec)
{
    std::vector<std::string> vec;
    vec.reserve(r_vec.size());
    for (auto j = 0; j < r_vec.size(); ++j) {
        vec.emplace_back(r_vec(j));
    }
    return vec;
}

void
copy(const Rcpp::StringVector &r_vec, std::vector<std::string> &vec)
{
    vec.clear();
    vec.reserve(r_vec.size());
    for (auto j = 0; j < r_vec.size(); ++j) {
        vec.emplace_back(r_vec(j));
    }
}

}} // namespace rcpp::util
