#include "mmutil.hh"
#include "mmutil_stat.hh"
#include "mmutil_score.hh"

//' Collect row-wise and column-wise statistics
//'
//' @param mtx_file data file
//' @param row_file row file
//' @param col_file column file
//'
//' @return a list of stat vectors
//'
//' @examples
//' rr <- rgamma(10, 1, 1) # ten cells
//' mm <- matrix(rgamma(10 * 3, 1, 1), 10, 3)
//' dat <- mmutilR::rcpp_mmutil_simulate_poisson(mm, rr, "sim_test")
//' scr <- mmutilR::rcpp_mmutil_compute_scores(dat$mtx)
//' A <- as.matrix(Matrix::readMM(dat$mtx))
//' colMeans(A)
//' scr$col$mean
//' rowMeans(A)
//' scr$row$mean
//'
// [[Rcpp::export]]
Rcpp::List
rcpp_mmutil_compute_scores(
    const std::string mtx_file,
    Rcpp::Nullable<const std::string> row_file = R_NilValue,
    Rcpp::Nullable<const std::string> col_file = R_NilValue,
    const std::size_t MAX_ROW_WORD = 2,
    const char ROW_WORD_SEP = '_',
    const std::size_t MAX_COL_WORD = 100,
    const char COL_WORD_SEP = '@')
{

    TLOG("collecting statistics... ");

    row_col_stat_collector_t collector;
    visit_matrix_market_file(mtx_file, collector);

    TLOG("... collected 'em all");

    Index max_row, max_col;

    Vec row_cv, row_sd, row_mean;
    IntVec row_nvec;
    std::tie(row_mean, row_sd, row_cv, row_nvec, max_row, max_col) =
        compute_mtx_stat(
            collector,
            [](const auto &x) { return x.Row_S1; },
            [](const auto &x) { return x.Row_S2; },
            [](const auto &x) { return x.Row_N; },
            [](const auto &x) { return x.max_col; });

    Vec col_cv, col_sd, col_mean;
    IntVec col_nvec;
    std::tie(col_mean, col_sd, col_cv, col_nvec, std::ignore, std::ignore) =
        compute_mtx_stat(
            collector,
            [](const auto &x) { return x.Col_S1; },
            [](const auto &x) { return x.Col_S2; },
            [](const auto &x) { return x.Col_N; },
            [](const auto &x) { return x.max_row; });

    std::vector<std::string> row_names;
    row_names.reserve(row_nvec.size());

    std::vector<std::string> col_names;
    col_names.reserve(row_nvec.size());

    std::string _row_file = "", _col_file = "";

    if (row_file.isNotNull())
        _row_file = Rcpp::as<std::string>(row_file);

    if (col_file.isNotNull())
        _col_file = Rcpp::as<std::string>(col_file);

    if (file_exists(_row_file)) {
        CHK_RETL_(read_line_file(_row_file,
                                 row_names,
                                 MAX_ROW_WORD,
                                 ROW_WORD_SEP),
                  "Failed to read row names");
    } else {
        for (Index x = 0; x < max_row; ++x) {
            row_names.emplace_back(std::to_string(x + 1));
        }
    }

    if (file_exists(_col_file)) {
        CHK_RETL_(read_line_file(_col_file,
                                 col_names,
                                 MAX_COL_WORD,
                                 COL_WORD_SEP),
                  "Failed to read col names");
    } else {
        for (Index x = 0; x < max_col; ++x) {
            col_names.emplace_back(std::to_string(x + 1));
        }
    }

    Rcpp::List row_out =
        Rcpp::List::create(Rcpp::_["name"] =
                               Rcpp::StringVector(row_names.begin(),
                                                  row_names.end()),
                           Rcpp::_["nnz"] = row_nvec,
                           Rcpp::_["mean"] = row_mean,
                           Rcpp::_["sd"] = row_sd,
                           Rcpp::_["cv"] = row_cv,
                           Rcpp::_["sum"] = collector.Row_S1,
                           Rcpp::_["sum.sq"] = collector.Row_S2);

    Rcpp::List col_out =
        Rcpp::List::create(Rcpp::_["name"] =
                               Rcpp::StringVector(col_names.begin(),
                                                  col_names.end()),
                           Rcpp::_["nnz"] = col_nvec,
                           Rcpp::_["mean"] = col_mean,
                           Rcpp::_["sd"] = col_sd,
                           Rcpp::_["cv"] = col_cv,
                           Rcpp::_["sum"] = collector.Col_S1,
                           Rcpp::_["sum.sq"] = collector.Col_S2);

    return Rcpp::List::create(Rcpp::_["row"] = row_out,
                              Rcpp::_["col"] = col_out,
                              Rcpp::_["max.row"] = max_row,
                              Rcpp::_["max.col"] = max_col);
}
