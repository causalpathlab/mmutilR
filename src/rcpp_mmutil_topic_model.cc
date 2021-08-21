#include "mmutil_topic.hh"
#include "sampler.hh"

#include "mmutil_io.hh"
#include "mmutil_stat.hh"
#include "mmutil_index.hh"
#include "mmutil_bgzf_util.hh"
#include "mmutil_spectral.hh"
#include "mmutil_match.hh"
#include "mmutil_velocity.hh"

// [[Rcpp::plugins(openmp)]]

#include "io.hh"
#include "std_util.hh"

// [[Rcpp::export]]
Rcpp::List
rcpp_mmutil_fit_topic_model(const std::string spliced_mtx_file,
                            const std::string row_file,
                            const std::string col_file,
                            const float a0 = 1.0,
                            const float b0 = 1.0,
                            const std::size_t MAX_ITER = 100,
                            const float TOL = 1e-4,
                            const std::size_t NUM_THREADS = 1)
{
    CHECK(mmutil::bgzf::convert_bgzip(spliced_mtx_file));
    const std::string s_idx_file = spliced_mtx_file + ".index";
    CHECK(mmutil::index::build_mmutil_index(spliced_mtx_file, s_idx_file));
    std::vector<Index> spliced_idx;
    CHECK(mmutil::index::read_mmutil_index(s_idx_file, spliced_idx));

    return Rcpp::List::create();
}
