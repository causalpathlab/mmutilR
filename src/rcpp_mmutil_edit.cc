#include "mmutil.hh"
#include "mmutil_select.hh"
#include "mmutil_merge_col.hh"
#include "mmutil_filter.hh"

#include <unordered_set>

//' Merge multiple 10x mtx file sets into one set
//'
//' @param r_headers file set headers
//' @param r_batches unique batch names for each header
//' @param r_mtx_files A list of mtx files
//' @param r_row_files A list of row files
//' @param r_col_files A list of col files
//' @param r_fixed_rows A list of rows/features
//' @param output output file header
//' @param nnz_cutoff number of non-zero cutoff for columns
//' @param delim delimiter in the column name
//'
//' @return a list of file names: {output}.{mtx,rows,cols}.gz
//'
//' @examples
//'
//' options(stringsAsFactors=FALSE)
//' rr <- rgamma(10, 1, 1) # ten cells
//' mm <- matrix(rgamma(10 * 3, 1, 1), 10, 3)
//' t1 <- mmutilR::rcpp_mmutil_simulate_poisson(mm, rr, "test1")
//' t2 <- mmutilR::rcpp_mmutil_simulate_poisson(mm, rr, "test2")
//' bats <- hdrs <- c("test1","test2")
//' t3 <- mmutilR::rcpp_mmutil_merge_file_sets(
//'                       hdrs, bats, output = "test3", nnz_cutoff = 0)
//' A1 <- Matrix::readMM(t1$mtx);
//' rownames(A1) <- unlist(read.table(gzfile(t1$row)))
//' A2 <- Matrix::readMM(t2$mtx)
//' rownames(A2) <- unlist(read.table(gzfile(t2$row)))
//' A3 <- Matrix::readMM(t3$mtx)
//' rownames(A3) <- unlist(read.table(gzfile(t3$row)))
//' print(cbind(A1, A2))
//' print(A3)
//' unlink(list.files(pattern = "test1"))
//' unlink(list.files(pattern = "test2"))
//' unlink(list.files(pattern = "test3"))
//'
// [[Rcpp::export]]
Rcpp::List
rcpp_mmutil_merge_file_sets(
    Rcpp::Nullable<const Rcpp::StringVector> r_headers = R_NilValue,
    Rcpp::Nullable<const Rcpp::StringVector> r_batches = R_NilValue,
    Rcpp::Nullable<const Rcpp::StringVector> r_mtx_files = R_NilValue,
    Rcpp::Nullable<const Rcpp::StringVector> r_row_files = R_NilValue,
    Rcpp::Nullable<const Rcpp::StringVector> r_col_files = R_NilValue,
    Rcpp::Nullable<const Rcpp::StringVector> r_fixed_rows = R_NilValue,
    const std::string output = "output",
    const double nnz_cutoff = 1,
    const std::string delim = "_",
    const std::size_t MAX_ROW_WORD = 2,
    const char ROW_WORD_SEP = '_',
    const std::size_t MAX_COL_WORD = 100,
    const char COL_WORD_SEP = '@')
{

    ASSERT_RETL(r_batches.isNotNull(), "must provide batch info");

    std::vector<std::string> batches = copy(Rcpp::StringVector(r_batches));

    std::vector<std::string> headers;
    if (r_headers.isNotNull()) {
        headers = copy(Rcpp::StringVector(r_headers));
        ASSERT_RETL(
            headers.size() == batches.size(),
            "Must have a list of batch names matching the list of headers");
    }

    std::unordered_set<std::string> _rows; // Take a unique set of
    std::vector<std::string> glob_rows;    // row names

    std::vector<std::string> mtx_files;
    std::vector<std::string> row_files;
    std::vector<std::string> col_files;

    if (r_mtx_files.isNull() || r_row_files.isNull() || r_col_files.isNull()) {

        for (auto s : headers) {
            std::vector<std::string> rows_s;
            std::string row_file_s = s + ".rows.gz";
            std::string mtx_file_s = s + ".mtx.gz";
            std::string col_file_s = s + ".cols.gz";

            ASSERT_RETL(file_exists(mtx_file_s),
                        "unable to find the mtx file: " << mtx_file_s);

            ASSERT_RETL(file_exists(row_file_s),
                        "unable to find the row file: " << row_file_s);

            ASSERT_RETL(file_exists(col_file_s),
                        "unable to find the col file: " << col_file_s);

            // ASSERT_RETL(read_vector_file(row_file_s, rows_s) == EXIT_SUCCESS,
            //             "unable to read the row file: " << row_file_s);
            CHK_RETL_(read_line_file(row_file_s,
                                     rows_s,
                                     MAX_ROW_WORD,
                                     ROW_WORD_SEP),
                      "unable to read the row file: " << row_file_s);

            for (auto r : rows_s) {
                _rows.insert(r);
            }

            mtx_files.emplace_back(mtx_file_s);
            row_files.emplace_back(row_file_s);
            col_files.emplace_back(col_file_s);
        }

    } else {

        for (auto s : headers) {
            TLOG("Ignore headers: " << s);
        }

        ASSERT_RETL(r_mtx_files.isNotNull() && r_row_files.isNotNull() &&
                        r_col_files.isNotNull(),
                    "We need mtx, row, col files");

        mtx_files = copy(Rcpp::StringVector(r_mtx_files));
        row_files = copy(Rcpp::StringVector(r_row_files));
        col_files = copy(Rcpp::StringVector(r_col_files));

        ASSERT_RETL(mtx_files.size() == row_files.size() &&
                        mtx_files.size() == col_files.size(),
                    "different number of mtx, row, col files");

        ASSERT_RETL(
            mtx_files.size() == batches.size(),
            "Must have a list of batch names matching the list of mtx files");

        for (auto row_file_s : row_files) {
            std::vector<std::string> rows_s;
            // ASSERT_RETL(read_vector_file(row_file_s, rows_s) == EXIT_SUCCESS,
            //             "unable to read the row file: " << row_file_s);
            ASSERT_RETL(read_line_file(row_file_s,
                                       rows_s,
                                       MAX_ROW_WORD,
                                       ROW_WORD_SEP) == EXIT_SUCCESS,
                        "unable to read the row file: " << row_file_s);
            for (auto r : rows_s) {
                _rows.insert(r);
            }
        }
    }

    if (r_fixed_rows.isNotNull()) {
        glob_rows = copy(Rcpp::StringVector(r_fixed_rows));
        Index n_overlap = 0;
        for (auto r : glob_rows) {
            if (_rows.count(r) > 0)
                n_overlap++;
        }
        ASSERT_RETL(
            n_overlap > 0,
            "no overlap between the row files and the row names provided");
    } else {
        glob_rows.reserve(_rows.size());
        std::copy(_rows.begin(), _rows.end(), std::back_inserter(glob_rows));
        std::sort(glob_rows.begin(), glob_rows.end());
    }

    CHECK(run_merge_col(glob_rows,
                        nnz_cutoff,
                        output,
                        mtx_files,
                        row_files,
                        col_files,
                        MAX_ROW_WORD,
                        ROW_WORD_SEP,
                        MAX_COL_WORD,
                        COL_WORD_SEP));

    const std::string mtx_file = output + ".mtx.gz";
    const std::string row_file = output + ".rows.gz";
    const std::string col_file = output + ".cols.gz";
    const std::string idx_file = mtx_file + ".index";

    ////////////////////
    // rename columns //
    ////////////////////

    std::vector<std::tuple<std::string, Index>> column_pairs;
    CHECK(read_pair_file(output + ".columns.gz", column_pairs));

    std::vector<std::string> cols;
    cols.reserve(column_pairs.size());
    std::transform(column_pairs.begin(),
                   column_pairs.end(),
                   std::back_inserter(cols),
                   [&batches, &delim](const auto pp) -> std::string {
                       const Index b = std::get<1>(pp) - 1;
                       return std::get<0>(pp) + delim + batches[b];
                   });

    write_vector_file(col_file, cols);

    ////////////////////////
    // index new mtx file //
    ////////////////////////

    if (file_exists(idx_file))
        remove_file(idx_file);

    CHECK(mmutil::index::build_mmutil_index(mtx_file, idx_file));

    return Rcpp::List::create(Rcpp::_["mtx"] = mtx_file,
                              Rcpp::_["row"] = row_file,
                              Rcpp::_["col"] = col_file,
                              Rcpp::_["idx"] = idx_file);
}

//' Take a subset of rows and create a new MTX file-set
//'
//' @description For the new mtx file, empty columns with only zero
//'   elements will be removed.
//'
//' @param mtx_file data file
//' @param row_file row file
//' @param col_file column file
//' @param selected selected row names
//' @param output output
//'
//' @return a list of file names: {output}.{mtx,rows,cols}.gz
//'
//' @examples
//'
//' options(stringsAsFactors=FALSE)
//' rr <- rgamma(20, 1, 1)
//' mm <- matrix(rgamma(10 * 2, 1, 1), 10, 2)
//' src.hdr <- "test_org"
//' src.files <- mmutilR::rcpp_mmutil_simulate_poisson(mm, rr, src.hdr)
//' Y <- Matrix::readMM(src.files$mtx)
//' rownames(Y) <- read.table(src.files$row)$V1
//' print(Y)
//' sub.rows <- sort(read.table(src.files$row)$V1[sample(10,3)])
//' print(sub.rows)
//' tgt.hdr <- "test_sub"
//' tgt.files <- mmutilR::rcpp_mmutil_copy_selected_rows(
//'                src.files$mtx,
//'                src.files$row,
//'                src.files$col,
//'                sub.rows,
//'                tgt.hdr)
//' Y <- Matrix::readMM(tgt.files$mtx)
//' colnames(Y) <- read.table(tgt.files$col)$V1
//' rownames(Y) <- read.table(tgt.files$row)$V1
//' print(Y)
//' unlink(list.files(pattern = src.hdr))
//' unlink(list.files(pattern = tgt.hdr))
//'
// [[Rcpp::export]]
Rcpp::List
rcpp_mmutil_copy_selected_rows(const std::string mtx_file,
                               const std::string row_file,
                               const std::string col_file,
                               const Rcpp::StringVector &r_selected,
                               const std::string output,
                               const std::size_t MAX_ROW_WORD = 100,
                               const char ROW_WORD_SEP = '@',
                               const std::size_t MAX_COL_WORD = 100,
                               const char COL_WORD_SEP = '@')
{

    ASSERT_RETL(file_exists(mtx_file), "missing the MTX file");
    ASSERT_RETL(file_exists(row_file), "missing the ROW file");
    ASSERT_RETL(file_exists(col_file), "missing the COL file");

    std::vector<std::string> selected = copy(r_selected);

    // First pass: select rows and create temporary MTX
    copy_selected_rows(mtx_file,
                       row_file,
                       selected,
                       output + "-temp",
                       MAX_ROW_WORD,
                       ROW_WORD_SEP);

    std::string temp_mtx_file = output + "-temp.mtx.gz";

    // Second pass: squeeze out empty columns
    CHK_RETL(filter_col_by_nnz(1,
                               temp_mtx_file,
                               col_file,
                               output,
                               MAX_COL_WORD,
                               COL_WORD_SEP));

    if (file_exists(temp_mtx_file)) {
        std::remove(temp_mtx_file.c_str());
    }

    std::string out_row_file = output + ".rows.gz";
    std::string out_col_file = output + ".cols.gz";
    std::string out_mtx_file = output + ".mtx.gz";

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

    ASSERT_RETL(file_exists(out_row_file),
                "unable to create the row file: " << out_row_file)

    ASSERT_RETL(file_exists(out_col_file),
                "unable to create the column file: " << out_col_file)

    ASSERT_RETL(file_exists(out_mtx_file),
                "unable to create the mtx file: " << out_mtx_file)

    ////////////////////////
    // index new mtx file //
    ////////////////////////

    const std::string out_idx_file = out_mtx_file + ".index";

    if (file_exists(out_idx_file))
        remove_file(out_idx_file);

    CHECK(mmutil::index::build_mmutil_index(out_mtx_file, out_idx_file));

    return Rcpp::List::create(Rcpp::_["mtx"] = out_mtx_file,
                              Rcpp::_["row"] = out_row_file,
                              Rcpp::_["col"] = out_col_file,
                              Rcpp::_["idx"] = out_idx_file);
}

//' Take a subset of columns and create a new MTX file-set
//'
//' @param mtx_file data file
//' @param row_file row file
//' @param col_file column file
//' @param selected selected column names
//'
//' @examples
//'
//' options(stringsAsFactors=FALSE)
//' rr <- rgamma(20, 1, 1)
//' mm <- matrix(rgamma(10 * 2, 1, 1), 10, 2)
//' src.hdr <- "test_org"
//' src.files <- mmutilR::rcpp_mmutil_simulate_poisson(mm, rr, src.hdr)
//' Y <- Matrix::readMM(src.files$mtx)
//' colnames(Y) <- read.table(src.files$col)$V1
//' print(Y)
//' sub.cols <- sort(read.table(src.files$col)$V1[sample(20,3)])
//' print(sub.cols)
//' tgt.hdr <- "test_sub"
//' tgt.files <- mmutilR::rcpp_mmutil_copy_selected_columns(
//'                          src.files$mtx,
//'                          src.files$row,
//'                          src.files$col,
//'                          sub.cols, tgt.hdr)
//' Y <- Matrix::readMM(tgt.files$mtx)
//' colnames(Y) <- read.table(tgt.files$col)$V1
//' print(Y)
//' unlink(list.files(pattern = src.hdr))
//' unlink(list.files(pattern = tgt.hdr))
//'
// [[Rcpp::export]]
Rcpp::List
rcpp_mmutil_copy_selected_columns(const std::string mtx_file,
                                  const std::string row_file,
                                  const std::string col_file,
                                  const Rcpp::StringVector &r_selected,
                                  const std::string output,
                                  const std::size_t MAX_COL_WORD = 100,
                                  const char COL_WORD_SEP = '@')
{

    ASSERT_RETL(file_exists(mtx_file), "missing the MTX file");
    ASSERT_RETL(file_exists(row_file), "missing the ROW file");
    ASSERT_RETL(file_exists(col_file), "missing the COL file");

    std::vector<std::string> selected = copy(r_selected);

    const std::string out_row_file = output + ".rows.gz";
    copy_file(row_file, out_row_file);
    copy_selected_columns(mtx_file,
                          col_file,
                          selected,
                          output,
                          MAX_COL_WORD,
                          COL_WORD_SEP);

    ASSERT_RETL(file_exists(out_row_file),
                "unable to create the row file: " << out_row_file)

    std::string out_col_file = output + ".cols.gz";
    ASSERT_RETL(file_exists(out_col_file),
                "unable to create the column file: " << out_col_file)

    std::string out_mtx_file = output + ".mtx.gz";
    ASSERT_RETL(file_exists(out_mtx_file),
                "unable to create the mtx file: " << out_mtx_file)

    ////////////////////////
    // index new mtx file //
    ////////////////////////

    const std::string out_idx_file = out_mtx_file + ".index";

    if (file_exists(out_idx_file))
        remove_file(out_idx_file);

    CHECK(mmutil::index::build_mmutil_index(out_mtx_file, out_idx_file));

    return Rcpp::List::create(Rcpp::_["mtx"] = out_mtx_file,
                              Rcpp::_["row"] = out_row_file,
                              Rcpp::_["col"] = out_col_file,
                              Rcpp::_["idx"] = out_idx_file);
}
