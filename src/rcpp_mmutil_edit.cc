#include "mmutil_select.hh"
#include "mmutil_filter_col.hh"

//' Take a subset of rows and create another MTX file-set
//'
//' @param mtx_file data file
//' @param row_file row file
//' @param col_file column file
//' @param selected selected row names
//'
//'
// [[Rcpp::export]]
int
rcpp_copy_selected_rows(const std::string mtx_file,
                        const std::string row_file,
                        const std::string col_file,
                        const Rcpp::StringVector &r_selected,
                        const std::string output)
{

    ERR_RET(!file_exists(mtx_file), "missing the MTX file");
    ERR_RET(!file_exists(row_file), "missing the ROW file");
    ERR_RET(!file_exists(col_file), "missing the COL file");

    std::vector<std::string> selected;
    selected.reserve(r_selected.size());
    for (Index j = 0; j < r_selected.size(); ++j) {
        selected.emplace_back(r_selected(j));
    }

    // First pass: select rows and create temporary MTX
    copy_selected_rows(mtx_file, row_file, selected, output + "-temp");

    std::string temp_mtx_file = output + "-temp.mtx.gz";
    // Second pass: squeeze out empty columns
    filter_col_by_nnz(1, temp_mtx_file, col_file, output);

    if (file_exists(temp_mtx_file)) {
        std::remove(temp_mtx_file.c_str());
    }

    std::string out_row_file = output + ".rows.gz";

    if (file_exists(out_row_file)) {
        std::string temp_row_file = output + ".rows.gz-backup";
        WLOG("Remove existing output row file: " << out_row_file);
        copy_file(out_row_file, temp_row_file);
        remove_file(out_row_file);
    }

    {
        std::string temp_row_file = output + "-temp.rows.gz";
        rename_file(temp_row_file, out_row_file);
    }

    return EXIT_SUCCESS;
}

//' Take a subset of columns and create another MTX file-set
//'
//' @param mtx_file data file
//' @param row_file row file
//' @param col_file column file
//' @param selected selected column names
//'
// [[Rcpp::export]]
int
rcpp_copy_selected_columns(const std::string mtx_file,
                           const std::string row_file,
                           const std::string col_file,
                           const Rcpp::StringVector &r_selected,
                           const std::string output)
{

    ERR_RET(!file_exists(mtx_file), "missing the MTX file");
    ERR_RET(!file_exists(row_file), "missing the ROW file");
    ERR_RET(!file_exists(col_file), "missing the COL file");

    std::vector<std::string> selected;
    selected.reserve(r_selected.size());
    for (Index j = 0; j < r_selected.size(); ++j) {
        selected.emplace_back(r_selected(j));
    }

    const std::string out_row_file = output + ".rows.gz";
    copy_file(out_row_file, row_file);
    copy_selected_columns(mtx_file, col_file, selected, output);

    return EXIT_SUCCESS;
}
