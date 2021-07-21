#include "mmutil.hh"
#include "mmutil_stat.hh"
#include "mmutil_score.hh"

//' Collect row-wise and column-wise statistics
//'
//' @param mtx_file data file
//' @return a list of stat vectors
//'
//' @examples
//' rr <- rgamma(10, 1, 1) # ten cells
//' mm <- matrix(rgamma(10 * 3, 1, 1), 10, 3)
//' dat <- mmutilR::rcpp_simulate_poisson_data(mm, rr, "sim_test")
//' scr <- mmutilR::rcpp_compute_scores(dat$mtx)
//' A <- as.matrix(Matrix::readMM(dat$mtx))
//' colMeans(A)
//' scr$col.mean
//' rowMeans(A)
//' scr$row.mean
//'
// [[Rcpp::export]]
Rcpp::List
rcpp_compute_scores(const std::string mtx_file)
{

    TLOG("collecting statistics... ");

    row_col_stat_collector_t collector;
    visit_matrix_market_file(mtx_file, collector);

    TLOG("... collected 'em all");

    Index max_row, max_col;

    Vec row_cv, row_sd, row_mean;
    IntVec row_nvec;
    std::tie(row_mean, row_sd, row_cv, row_nvec, max_row, max_col) =
        compute_mtx_stat(collector,
                         [](const auto &x) { return x.Row_S1; },
                         [](const auto &x) { return x.Row_S2; },
                         [](const auto &x) { return x.Row_N; },
                         [](const auto &x) { return x.max_col; });

    Vec col_cv, col_sd, col_mean;
    IntVec col_nvec;
    std::tie(col_mean, col_sd, col_cv, col_nvec, std::ignore, std::ignore) =
        compute_mtx_stat(collector,
                         [](const auto &x) { return x.Col_S1; },
                         [](const auto &x) { return x.Col_S2; },
                         [](const auto &x) { return x.Col_N; },
                         [](const auto &x) { return x.max_row; });

    return Rcpp::List::create(Rcpp::_["row.nnz"] = row_nvec,
                              Rcpp::_["row.mean"] = row_mean,
                              Rcpp::_["row.sd"] = row_sd,
                              Rcpp::_["row.cv"] = row_cv,
                              Rcpp::_["row.sum"] = collector.Row_S1,
                              Rcpp::_["row.sum.sq"] = collector.Row_S2,
                              Rcpp::_["col.nnz"] = col_nvec,
                              Rcpp::_["col.mean"] = col_mean,
                              Rcpp::_["col.sd"] = col_sd,
                              Rcpp::_["col.cv"] = col_cv,
                              Rcpp::_["col.sum"] = collector.Col_S1,
                              Rcpp::_["col.sum.sq"] = collector.Col_S2,
                              Rcpp::_["max.row"] = max_row,
                              Rcpp::_["max.col"] = max_col);
}
