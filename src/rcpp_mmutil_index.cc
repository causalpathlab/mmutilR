#include "mmutil.hh"
#include "mmutil_util.hh"
#include "mmutil_index.hh"

//' Create an index file for a given MTX
//'
//' @param mtx_file data file
//' @param index_file index file
//'
//' @usage rcpp_mmutil_build_index(mtx_file, index_file)
//'
//' @return EXIT_SUCCESS or EXIT_FAILURE
//'
// [[Rcpp::export]]
int
rcpp_mmutil_build_index(const std::string mtx_file,
                        const std::string index_file = "")
{
    using namespace mmutil::index;
    CHK_RET(mmutil::bgzf::convert_bgzip(mtx_file));
    return build_mmutil_index(mtx_file, index_file);
}

//' Read an index file to R
//'
//' @param index_file index file
//'
//' @return a vector column index (a vector of memory locations)
//'
// [[Rcpp::export]]
Rcpp::NumericVector
rcpp_mmutil_read_index(const std::string index_file)
{
    using namespace mmutil::index;

    std::vector<Index> ret;
    if (read_mmutil_index(index_file, ret) != EXIT_SUCCESS)
        return Rcpp::NumericVector();

    return Rcpp::NumericVector(ret.begin(), ret.end());
}

//' Check if the index tab is valid
//'
//' @param mtx_file data file
//' @param index_tab index tab (a vector of memory locations)
//'
//' @return EXIT_SUCCESS or EXIT_FAILURE
//'
// [[Rcpp::export]]
int
rcpp_mmutil_check_index(const std::string mtx_file,
                        const Rcpp::NumericVector &index_tab)
{
    CHK_RET(mmutil::bgzf::convert_bgzip(mtx_file));
    using namespace mmutil::index;
    std::vector<Index> _idx(index_tab.begin(), index_tab.end());
    return check_index_tab(mtx_file, _idx);
}
