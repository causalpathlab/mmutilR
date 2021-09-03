#include "mmutil.hh"
#include "mmutil_stat.hh"
#include "mmutil_bgzf_util.hh"
#include "mmutil_util.hh"
#include "mmutil_index.hh"
#include "mmutil_io.hh"
#include "tuple_util.hh"

#include <algorithm>
#include <functional>
#include <string>
#include <unordered_set>

//' Cell type deconvolution of bulk data based on single-cell data
//'
//' Salamander (semi-supervised annotation of latent states by marker
//' gene-derived regression model)
//'
// [[Rcpp::export]]
Rcpp::List
rcpp_mmutil_deconvolve_svd(
    const std::string mtx_file = "",
    const std::string row_file = "",
    const std::string col_file = "",
    Rcpp::Nullable<Rcpp::NumericMatrix> r_U = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericMatrix> r_D = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericMatrix> r_V = R_NilValue,
    Rcpp::Nullable<Rcpp::StringVector> r_rows = R_NilValue,
    Rcpp::Nullable<Rcpp::StringVector> r_cols = R_NilValue,
    const bool TAKE_LN = false,
    const bool VERBOSE = false)
{



    return Rcpp::List::create();
}
