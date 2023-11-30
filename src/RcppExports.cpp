// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// rcpp_mmutil_bbknn_svd
Rcpp::List rcpp_mmutil_bbknn_svd(const std::string mtx_file, const Rcpp::StringVector& r_batches, const std::size_t knn, const std::size_t RANK, const bool RECIPROCAL_MATCH, const bool TAKE_LN, const double TAU, const double COL_NORM, const std::size_t EM_ITER, const double EM_TOL, const std::size_t KNN_BILINK, const std::size_t KNN_NNLIST, const std::size_t LU_ITER, const std::string row_weight_file, const std::size_t NUM_THREADS, const std::size_t BLOCK_SIZE);
RcppExport SEXP _mmutilR_rcpp_mmutil_bbknn_svd(SEXP mtx_fileSEXP, SEXP r_batchesSEXP, SEXP knnSEXP, SEXP RANKSEXP, SEXP RECIPROCAL_MATCHSEXP, SEXP TAKE_LNSEXP, SEXP TAUSEXP, SEXP COL_NORMSEXP, SEXP EM_ITERSEXP, SEXP EM_TOLSEXP, SEXP KNN_BILINKSEXP, SEXP KNN_NNLISTSEXP, SEXP LU_ITERSEXP, SEXP row_weight_fileSEXP, SEXP NUM_THREADSSEXP, SEXP BLOCK_SIZESEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string >::type mtx_file(mtx_fileSEXP);
    Rcpp::traits::input_parameter< const Rcpp::StringVector& >::type r_batches(r_batchesSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type knn(knnSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type RANK(RANKSEXP);
    Rcpp::traits::input_parameter< const bool >::type RECIPROCAL_MATCH(RECIPROCAL_MATCHSEXP);
    Rcpp::traits::input_parameter< const bool >::type TAKE_LN(TAKE_LNSEXP);
    Rcpp::traits::input_parameter< const double >::type TAU(TAUSEXP);
    Rcpp::traits::input_parameter< const double >::type COL_NORM(COL_NORMSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type EM_ITER(EM_ITERSEXP);
    Rcpp::traits::input_parameter< const double >::type EM_TOL(EM_TOLSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type KNN_BILINK(KNN_BILINKSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type KNN_NNLIST(KNN_NNLISTSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type LU_ITER(LU_ITERSEXP);
    Rcpp::traits::input_parameter< const std::string >::type row_weight_file(row_weight_fileSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type NUM_THREADS(NUM_THREADSSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type BLOCK_SIZE(BLOCK_SIZESEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_mmutil_bbknn_svd(mtx_file, r_batches, knn, RANK, RECIPROCAL_MATCH, TAKE_LN, TAU, COL_NORM, EM_ITER, EM_TOL, KNN_BILINK, KNN_NNLIST, LU_ITER, row_weight_file, NUM_THREADS, BLOCK_SIZE));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_mmutil_annotate_columns
Rcpp::List rcpp_mmutil_annotate_columns(const Rcpp::List pos_labels, Rcpp::Nullable<Rcpp::StringVector> r_rows, Rcpp::Nullable<Rcpp::StringVector> r_cols, Rcpp::Nullable<Rcpp::List> r_neg_labels, Rcpp::Nullable<Rcpp::NumericVector> r_qc_labels, const std::string mtx_file, const std::string row_file, const std::string col_file, Rcpp::Nullable<Rcpp::NumericMatrix> r_U, Rcpp::Nullable<Rcpp::NumericMatrix> r_D, Rcpp::Nullable<Rcpp::NumericMatrix> r_V, const double KAPPA_MAX, const bool TAKE_LN, const std::size_t BATCH_SIZE, const std::size_t EM_ITER, const double EM_TOL, const bool VERBOSE, const bool DO_STD);
RcppExport SEXP _mmutilR_rcpp_mmutil_annotate_columns(SEXP pos_labelsSEXP, SEXP r_rowsSEXP, SEXP r_colsSEXP, SEXP r_neg_labelsSEXP, SEXP r_qc_labelsSEXP, SEXP mtx_fileSEXP, SEXP row_fileSEXP, SEXP col_fileSEXP, SEXP r_USEXP, SEXP r_DSEXP, SEXP r_VSEXP, SEXP KAPPA_MAXSEXP, SEXP TAKE_LNSEXP, SEXP BATCH_SIZESEXP, SEXP EM_ITERSEXP, SEXP EM_TOLSEXP, SEXP VERBOSESEXP, SEXP DO_STDSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::List >::type pos_labels(pos_labelsSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::StringVector> >::type r_rows(r_rowsSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::StringVector> >::type r_cols(r_colsSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type r_neg_labels(r_neg_labelsSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::NumericVector> >::type r_qc_labels(r_qc_labelsSEXP);
    Rcpp::traits::input_parameter< const std::string >::type mtx_file(mtx_fileSEXP);
    Rcpp::traits::input_parameter< const std::string >::type row_file(row_fileSEXP);
    Rcpp::traits::input_parameter< const std::string >::type col_file(col_fileSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::NumericMatrix> >::type r_U(r_USEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::NumericMatrix> >::type r_D(r_DSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::NumericMatrix> >::type r_V(r_VSEXP);
    Rcpp::traits::input_parameter< const double >::type KAPPA_MAX(KAPPA_MAXSEXP);
    Rcpp::traits::input_parameter< const bool >::type TAKE_LN(TAKE_LNSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type BATCH_SIZE(BATCH_SIZESEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type EM_ITER(EM_ITERSEXP);
    Rcpp::traits::input_parameter< const double >::type EM_TOL(EM_TOLSEXP);
    Rcpp::traits::input_parameter< const bool >::type VERBOSE(VERBOSESEXP);
    Rcpp::traits::input_parameter< const bool >::type DO_STD(DO_STDSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_mmutil_annotate_columns(pos_labels, r_rows, r_cols, r_neg_labels, r_qc_labels, mtx_file, row_file, col_file, r_U, r_D, r_V, KAPPA_MAX, TAKE_LN, BATCH_SIZE, EM_ITER, EM_TOL, VERBOSE, DO_STD));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_mmutil_merge_file_sets
Rcpp::List rcpp_mmutil_merge_file_sets(Rcpp::Nullable<const Rcpp::StringVector> r_headers, Rcpp::Nullable<const Rcpp::StringVector> r_batches, Rcpp::Nullable<const Rcpp::StringVector> r_mtx_files, Rcpp::Nullable<const Rcpp::StringVector> r_row_files, Rcpp::Nullable<const Rcpp::StringVector> r_col_files, Rcpp::Nullable<const Rcpp::StringVector> r_fixed_rows, const std::string output, const double nnz_cutoff, const std::string delim, const std::size_t MAX_ROW_WORD, const char ROW_WORD_SEP, const std::size_t MAX_COL_WORD, const char COL_WORD_SEP);
RcppExport SEXP _mmutilR_rcpp_mmutil_merge_file_sets(SEXP r_headersSEXP, SEXP r_batchesSEXP, SEXP r_mtx_filesSEXP, SEXP r_row_filesSEXP, SEXP r_col_filesSEXP, SEXP r_fixed_rowsSEXP, SEXP outputSEXP, SEXP nnz_cutoffSEXP, SEXP delimSEXP, SEXP MAX_ROW_WORDSEXP, SEXP ROW_WORD_SEPSEXP, SEXP MAX_COL_WORDSEXP, SEXP COL_WORD_SEPSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::Nullable<const Rcpp::StringVector> >::type r_headers(r_headersSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<const Rcpp::StringVector> >::type r_batches(r_batchesSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<const Rcpp::StringVector> >::type r_mtx_files(r_mtx_filesSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<const Rcpp::StringVector> >::type r_row_files(r_row_filesSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<const Rcpp::StringVector> >::type r_col_files(r_col_filesSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<const Rcpp::StringVector> >::type r_fixed_rows(r_fixed_rowsSEXP);
    Rcpp::traits::input_parameter< const std::string >::type output(outputSEXP);
    Rcpp::traits::input_parameter< const double >::type nnz_cutoff(nnz_cutoffSEXP);
    Rcpp::traits::input_parameter< const std::string >::type delim(delimSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type MAX_ROW_WORD(MAX_ROW_WORDSEXP);
    Rcpp::traits::input_parameter< const char >::type ROW_WORD_SEP(ROW_WORD_SEPSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type MAX_COL_WORD(MAX_COL_WORDSEXP);
    Rcpp::traits::input_parameter< const char >::type COL_WORD_SEP(COL_WORD_SEPSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_mmutil_merge_file_sets(r_headers, r_batches, r_mtx_files, r_row_files, r_col_files, r_fixed_rows, output, nnz_cutoff, delim, MAX_ROW_WORD, ROW_WORD_SEP, MAX_COL_WORD, COL_WORD_SEP));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_mmutil_copy_selected_rows
Rcpp::List rcpp_mmutil_copy_selected_rows(const std::string mtx_file, const std::string row_file, const std::string col_file, const Rcpp::StringVector& r_selected, const std::string output, const std::size_t MAX_ROW_WORD, const char ROW_WORD_SEP, const std::size_t MAX_COL_WORD, const char COL_WORD_SEP);
RcppExport SEXP _mmutilR_rcpp_mmutil_copy_selected_rows(SEXP mtx_fileSEXP, SEXP row_fileSEXP, SEXP col_fileSEXP, SEXP r_selectedSEXP, SEXP outputSEXP, SEXP MAX_ROW_WORDSEXP, SEXP ROW_WORD_SEPSEXP, SEXP MAX_COL_WORDSEXP, SEXP COL_WORD_SEPSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string >::type mtx_file(mtx_fileSEXP);
    Rcpp::traits::input_parameter< const std::string >::type row_file(row_fileSEXP);
    Rcpp::traits::input_parameter< const std::string >::type col_file(col_fileSEXP);
    Rcpp::traits::input_parameter< const Rcpp::StringVector& >::type r_selected(r_selectedSEXP);
    Rcpp::traits::input_parameter< const std::string >::type output(outputSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type MAX_ROW_WORD(MAX_ROW_WORDSEXP);
    Rcpp::traits::input_parameter< const char >::type ROW_WORD_SEP(ROW_WORD_SEPSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type MAX_COL_WORD(MAX_COL_WORDSEXP);
    Rcpp::traits::input_parameter< const char >::type COL_WORD_SEP(COL_WORD_SEPSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_mmutil_copy_selected_rows(mtx_file, row_file, col_file, r_selected, output, MAX_ROW_WORD, ROW_WORD_SEP, MAX_COL_WORD, COL_WORD_SEP));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_mmutil_copy_selected_columns
Rcpp::List rcpp_mmutil_copy_selected_columns(const std::string mtx_file, const std::string row_file, const std::string col_file, const Rcpp::StringVector& r_selected, const std::string output, const std::size_t MAX_COL_WORD, const char COL_WORD_SEP);
RcppExport SEXP _mmutilR_rcpp_mmutil_copy_selected_columns(SEXP mtx_fileSEXP, SEXP row_fileSEXP, SEXP col_fileSEXP, SEXP r_selectedSEXP, SEXP outputSEXP, SEXP MAX_COL_WORDSEXP, SEXP COL_WORD_SEPSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string >::type mtx_file(mtx_fileSEXP);
    Rcpp::traits::input_parameter< const std::string >::type row_file(row_fileSEXP);
    Rcpp::traits::input_parameter< const std::string >::type col_file(col_fileSEXP);
    Rcpp::traits::input_parameter< const Rcpp::StringVector& >::type r_selected(r_selectedSEXP);
    Rcpp::traits::input_parameter< const std::string >::type output(outputSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type MAX_COL_WORD(MAX_COL_WORDSEXP);
    Rcpp::traits::input_parameter< const char >::type COL_WORD_SEP(COL_WORD_SEPSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_mmutil_copy_selected_columns(mtx_file, row_file, col_file, r_selected, output, MAX_COL_WORD, COL_WORD_SEP));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_mmutil_build_index
int rcpp_mmutil_build_index(const std::string mtx_file, const std::string index_file);
RcppExport SEXP _mmutilR_rcpp_mmutil_build_index(SEXP mtx_fileSEXP, SEXP index_fileSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string >::type mtx_file(mtx_fileSEXP);
    Rcpp::traits::input_parameter< const std::string >::type index_file(index_fileSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_mmutil_build_index(mtx_file, index_file));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_mmutil_read_index
Rcpp::NumericVector rcpp_mmutil_read_index(const std::string index_file);
RcppExport SEXP _mmutilR_rcpp_mmutil_read_index(SEXP index_fileSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string >::type index_file(index_fileSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_mmutil_read_index(index_file));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_mmutil_check_index
int rcpp_mmutil_check_index(const std::string mtx_file, const Rcpp::NumericVector& index_tab);
RcppExport SEXP _mmutilR_rcpp_mmutil_check_index(SEXP mtx_fileSEXP, SEXP index_tabSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string >::type mtx_file(mtx_fileSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type index_tab(index_tabSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_mmutil_check_index(mtx_file, index_tab));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_mmutil_info
Rcpp::List rcpp_mmutil_info(const std::string mtx_file);
RcppExport SEXP _mmutilR_rcpp_mmutil_info(SEXP mtx_fileSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string >::type mtx_file(mtx_fileSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_mmutil_info(mtx_file));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_mmutil_write_mtx
int rcpp_mmutil_write_mtx(const Eigen::SparseMatrix<float, Eigen::ColMajor>& X, const std::string mtx_file);
RcppExport SEXP _mmutilR_rcpp_mmutil_write_mtx(SEXP XSEXP, SEXP mtx_fileSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::SparseMatrix<float, Eigen::ColMajor>& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const std::string >::type mtx_file(mtx_fileSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_mmutil_write_mtx(X, mtx_file));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_mmutil_read_columns_sparse
Rcpp::List rcpp_mmutil_read_columns_sparse(const std::string mtx_file, const Rcpp::NumericVector& memory_location, const Rcpp::NumericVector& r_column_index, const bool verbose, const std::size_t NUM_THREADS, const std::size_t MIN_SIZE);
RcppExport SEXP _mmutilR_rcpp_mmutil_read_columns_sparse(SEXP mtx_fileSEXP, SEXP memory_locationSEXP, SEXP r_column_indexSEXP, SEXP verboseSEXP, SEXP NUM_THREADSSEXP, SEXP MIN_SIZESEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string >::type mtx_file(mtx_fileSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type memory_location(memory_locationSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type r_column_index(r_column_indexSEXP);
    Rcpp::traits::input_parameter< const bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type NUM_THREADS(NUM_THREADSSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type MIN_SIZE(MIN_SIZESEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_mmutil_read_columns_sparse(mtx_file, memory_location, r_column_index, verbose, NUM_THREADS, MIN_SIZE));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_mmutil_read_columns
Rcpp::NumericMatrix rcpp_mmutil_read_columns(const std::string mtx_file, const Rcpp::NumericVector& memory_location, const Rcpp::NumericVector& r_column_index, const bool verbose, const std::size_t NUM_THREADS, const std::size_t MIN_SIZE);
RcppExport SEXP _mmutilR_rcpp_mmutil_read_columns(SEXP mtx_fileSEXP, SEXP memory_locationSEXP, SEXP r_column_indexSEXP, SEXP verboseSEXP, SEXP NUM_THREADSSEXP, SEXP MIN_SIZESEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string >::type mtx_file(mtx_fileSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type memory_location(memory_locationSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type r_column_index(r_column_indexSEXP);
    Rcpp::traits::input_parameter< const bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type NUM_THREADS(NUM_THREADSSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type MIN_SIZE(MIN_SIZESEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_mmutil_read_columns(mtx_file, memory_location, r_column_index, verbose, NUM_THREADS, MIN_SIZE));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_mmutil_match_files
Rcpp::List rcpp_mmutil_match_files(const std::string src_mtx, const std::string tgt_mtx, const std::size_t knn, const std::size_t RANK, const bool TAKE_LN, const double TAU, const double COL_NORM, const std::size_t EM_ITER, const double EM_TOL, const std::size_t LU_ITER, const std::size_t KNN_BILINK, const std::size_t KNN_NNLIST, const std::string row_weight_file, const std::size_t NUM_THREADS, const std::size_t BLOCK_SIZE);
RcppExport SEXP _mmutilR_rcpp_mmutil_match_files(SEXP src_mtxSEXP, SEXP tgt_mtxSEXP, SEXP knnSEXP, SEXP RANKSEXP, SEXP TAKE_LNSEXP, SEXP TAUSEXP, SEXP COL_NORMSEXP, SEXP EM_ITERSEXP, SEXP EM_TOLSEXP, SEXP LU_ITERSEXP, SEXP KNN_BILINKSEXP, SEXP KNN_NNLISTSEXP, SEXP row_weight_fileSEXP, SEXP NUM_THREADSSEXP, SEXP BLOCK_SIZESEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string >::type src_mtx(src_mtxSEXP);
    Rcpp::traits::input_parameter< const std::string >::type tgt_mtx(tgt_mtxSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type knn(knnSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type RANK(RANKSEXP);
    Rcpp::traits::input_parameter< const bool >::type TAKE_LN(TAKE_LNSEXP);
    Rcpp::traits::input_parameter< const double >::type TAU(TAUSEXP);
    Rcpp::traits::input_parameter< const double >::type COL_NORM(COL_NORMSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type EM_ITER(EM_ITERSEXP);
    Rcpp::traits::input_parameter< const double >::type EM_TOL(EM_TOLSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type LU_ITER(LU_ITERSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type KNN_BILINK(KNN_BILINKSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type KNN_NNLIST(KNN_NNLISTSEXP);
    Rcpp::traits::input_parameter< const std::string >::type row_weight_file(row_weight_fileSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type NUM_THREADS(NUM_THREADSSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type BLOCK_SIZE(BLOCK_SIZESEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_mmutil_match_files(src_mtx, tgt_mtx, knn, RANK, TAKE_LN, TAU, COL_NORM, EM_ITER, EM_TOL, LU_ITER, KNN_BILINK, KNN_NNLIST, row_weight_file, NUM_THREADS, BLOCK_SIZE));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_mmutil_network_topic_data
Rcpp::List rcpp_mmutil_network_topic_data(const std::string mtx_file, const std::string row_file, const std::string col_file, const Eigen::MatrixXf latent_factor, const std::size_t knn, const std::string output, const bool write_sample_network, Rcpp::Nullable<std::string> output_sample_incidence, Rcpp::Nullable<std::string> output_sample_adjacency, Rcpp::Nullable<const Rcpp::StringVector> r_batches, const double CUTOFF, const bool WEIGHTED, const double MAXW, const std::size_t KNN_BILINK, const std::size_t KNN_NNLIST, const std::size_t NUM_THREADS, const std::size_t MAX_ROW_WORD, const char ROW_WORD_SEP, const std::size_t MAX_COL_WORD, const char COL_WORD_SEP);
RcppExport SEXP _mmutilR_rcpp_mmutil_network_topic_data(SEXP mtx_fileSEXP, SEXP row_fileSEXP, SEXP col_fileSEXP, SEXP latent_factorSEXP, SEXP knnSEXP, SEXP outputSEXP, SEXP write_sample_networkSEXP, SEXP output_sample_incidenceSEXP, SEXP output_sample_adjacencySEXP, SEXP r_batchesSEXP, SEXP CUTOFFSEXP, SEXP WEIGHTEDSEXP, SEXP MAXWSEXP, SEXP KNN_BILINKSEXP, SEXP KNN_NNLISTSEXP, SEXP NUM_THREADSSEXP, SEXP MAX_ROW_WORDSEXP, SEXP ROW_WORD_SEPSEXP, SEXP MAX_COL_WORDSEXP, SEXP COL_WORD_SEPSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string >::type mtx_file(mtx_fileSEXP);
    Rcpp::traits::input_parameter< const std::string >::type row_file(row_fileSEXP);
    Rcpp::traits::input_parameter< const std::string >::type col_file(col_fileSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXf >::type latent_factor(latent_factorSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type knn(knnSEXP);
    Rcpp::traits::input_parameter< const std::string >::type output(outputSEXP);
    Rcpp::traits::input_parameter< const bool >::type write_sample_network(write_sample_networkSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<std::string> >::type output_sample_incidence(output_sample_incidenceSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<std::string> >::type output_sample_adjacency(output_sample_adjacencySEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<const Rcpp::StringVector> >::type r_batches(r_batchesSEXP);
    Rcpp::traits::input_parameter< const double >::type CUTOFF(CUTOFFSEXP);
    Rcpp::traits::input_parameter< const bool >::type WEIGHTED(WEIGHTEDSEXP);
    Rcpp::traits::input_parameter< const double >::type MAXW(MAXWSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type KNN_BILINK(KNN_BILINKSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type KNN_NNLIST(KNN_NNLISTSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type NUM_THREADS(NUM_THREADSSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type MAX_ROW_WORD(MAX_ROW_WORDSEXP);
    Rcpp::traits::input_parameter< const char >::type ROW_WORD_SEP(ROW_WORD_SEPSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type MAX_COL_WORD(MAX_COL_WORDSEXP);
    Rcpp::traits::input_parameter< const char >::type COL_WORD_SEP(COL_WORD_SEPSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_mmutil_network_topic_data(mtx_file, row_file, col_file, latent_factor, knn, output, write_sample_network, output_sample_incidence, output_sample_adjacency, r_batches, CUTOFF, WEIGHTED, MAXW, KNN_BILINK, KNN_NNLIST, NUM_THREADS, MAX_ROW_WORD, ROW_WORD_SEP, MAX_COL_WORD, COL_WORD_SEP));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_mmutil_aggregate_pairwise
Rcpp::List rcpp_mmutil_aggregate_pairwise(const std::string mtx_file, const std::string row_file, const std::string col_file, Rcpp::Nullable<Rcpp::StringVector> r_indv, Rcpp::Nullable<Rcpp::NumericMatrix> r_V, Rcpp::Nullable<Rcpp::StringVector> r_cols, Rcpp::Nullable<Rcpp::StringVector> r_annot, Rcpp::Nullable<Rcpp::NumericMatrix> r_annot_mat, Rcpp::Nullable<Rcpp::StringVector> r_lab_name, const double a0, const double b0, const double eps, const std::size_t knn_cell, const std::size_t knn_indv, const std::size_t KNN_BILINK, const std::size_t KNN_NNLIST, const std::size_t NUM_THREADS, const bool IMPUTE_BY_KNN, const std::size_t MAX_ROW_WORD, const char ROW_WORD_SEP, const std::size_t MAX_COL_WORD, const char COL_WORD_SEP);
RcppExport SEXP _mmutilR_rcpp_mmutil_aggregate_pairwise(SEXP mtx_fileSEXP, SEXP row_fileSEXP, SEXP col_fileSEXP, SEXP r_indvSEXP, SEXP r_VSEXP, SEXP r_colsSEXP, SEXP r_annotSEXP, SEXP r_annot_matSEXP, SEXP r_lab_nameSEXP, SEXP a0SEXP, SEXP b0SEXP, SEXP epsSEXP, SEXP knn_cellSEXP, SEXP knn_indvSEXP, SEXP KNN_BILINKSEXP, SEXP KNN_NNLISTSEXP, SEXP NUM_THREADSSEXP, SEXP IMPUTE_BY_KNNSEXP, SEXP MAX_ROW_WORDSEXP, SEXP ROW_WORD_SEPSEXP, SEXP MAX_COL_WORDSEXP, SEXP COL_WORD_SEPSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string >::type mtx_file(mtx_fileSEXP);
    Rcpp::traits::input_parameter< const std::string >::type row_file(row_fileSEXP);
    Rcpp::traits::input_parameter< const std::string >::type col_file(col_fileSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::StringVector> >::type r_indv(r_indvSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::NumericMatrix> >::type r_V(r_VSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::StringVector> >::type r_cols(r_colsSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::StringVector> >::type r_annot(r_annotSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::NumericMatrix> >::type r_annot_mat(r_annot_matSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::StringVector> >::type r_lab_name(r_lab_nameSEXP);
    Rcpp::traits::input_parameter< const double >::type a0(a0SEXP);
    Rcpp::traits::input_parameter< const double >::type b0(b0SEXP);
    Rcpp::traits::input_parameter< const double >::type eps(epsSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type knn_cell(knn_cellSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type knn_indv(knn_indvSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type KNN_BILINK(KNN_BILINKSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type KNN_NNLIST(KNN_NNLISTSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type NUM_THREADS(NUM_THREADSSEXP);
    Rcpp::traits::input_parameter< const bool >::type IMPUTE_BY_KNN(IMPUTE_BY_KNNSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type MAX_ROW_WORD(MAX_ROW_WORDSEXP);
    Rcpp::traits::input_parameter< const char >::type ROW_WORD_SEP(ROW_WORD_SEPSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type MAX_COL_WORD(MAX_COL_WORDSEXP);
    Rcpp::traits::input_parameter< const char >::type COL_WORD_SEP(COL_WORD_SEPSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_mmutil_aggregate_pairwise(mtx_file, row_file, col_file, r_indv, r_V, r_cols, r_annot, r_annot_mat, r_lab_name, a0, b0, eps, knn_cell, knn_indv, KNN_BILINK, KNN_NNLIST, NUM_THREADS, IMPUTE_BY_KNN, MAX_ROW_WORD, ROW_WORD_SEP, MAX_COL_WORD, COL_WORD_SEP));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_mmutil_aggregate
Rcpp::List rcpp_mmutil_aggregate(const std::string mtx_file, const std::string row_file, const std::string col_file, Rcpp::Nullable<Rcpp::StringVector> r_cols, Rcpp::Nullable<Rcpp::StringVector> r_indv, Rcpp::Nullable<Rcpp::StringVector> r_annot, Rcpp::Nullable<Rcpp::NumericMatrix> r_annot_mat, Rcpp::Nullable<Rcpp::StringVector> r_lab_name, Rcpp::Nullable<Rcpp::StringVector> r_trt, Rcpp::Nullable<Rcpp::NumericMatrix> r_V, const double a0, const double b0, const double eps, const std::size_t knn, const std::size_t KNN_BILINK, const std::size_t KNN_NNLIST, const std::size_t NUM_THREADS, const bool IMPUTE_BY_KNN, const std::size_t MAX_ROW_WORD, const char ROW_WORD_SEP, const std::size_t MAX_COL_WORD, const char COL_WORD_SEP);
RcppExport SEXP _mmutilR_rcpp_mmutil_aggregate(SEXP mtx_fileSEXP, SEXP row_fileSEXP, SEXP col_fileSEXP, SEXP r_colsSEXP, SEXP r_indvSEXP, SEXP r_annotSEXP, SEXP r_annot_matSEXP, SEXP r_lab_nameSEXP, SEXP r_trtSEXP, SEXP r_VSEXP, SEXP a0SEXP, SEXP b0SEXP, SEXP epsSEXP, SEXP knnSEXP, SEXP KNN_BILINKSEXP, SEXP KNN_NNLISTSEXP, SEXP NUM_THREADSSEXP, SEXP IMPUTE_BY_KNNSEXP, SEXP MAX_ROW_WORDSEXP, SEXP ROW_WORD_SEPSEXP, SEXP MAX_COL_WORDSEXP, SEXP COL_WORD_SEPSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string >::type mtx_file(mtx_fileSEXP);
    Rcpp::traits::input_parameter< const std::string >::type row_file(row_fileSEXP);
    Rcpp::traits::input_parameter< const std::string >::type col_file(col_fileSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::StringVector> >::type r_cols(r_colsSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::StringVector> >::type r_indv(r_indvSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::StringVector> >::type r_annot(r_annotSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::NumericMatrix> >::type r_annot_mat(r_annot_matSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::StringVector> >::type r_lab_name(r_lab_nameSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::StringVector> >::type r_trt(r_trtSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::NumericMatrix> >::type r_V(r_VSEXP);
    Rcpp::traits::input_parameter< const double >::type a0(a0SEXP);
    Rcpp::traits::input_parameter< const double >::type b0(b0SEXP);
    Rcpp::traits::input_parameter< const double >::type eps(epsSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type knn(knnSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type KNN_BILINK(KNN_BILINKSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type KNN_NNLIST(KNN_NNLISTSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type NUM_THREADS(NUM_THREADSSEXP);
    Rcpp::traits::input_parameter< const bool >::type IMPUTE_BY_KNN(IMPUTE_BY_KNNSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type MAX_ROW_WORD(MAX_ROW_WORDSEXP);
    Rcpp::traits::input_parameter< const char >::type ROW_WORD_SEP(ROW_WORD_SEPSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type MAX_COL_WORD(MAX_COL_WORDSEXP);
    Rcpp::traits::input_parameter< const char >::type COL_WORD_SEP(COL_WORD_SEPSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_mmutil_aggregate(mtx_file, row_file, col_file, r_cols, r_indv, r_annot, r_annot_mat, r_lab_name, r_trt, r_V, a0, b0, eps, knn, KNN_BILINK, KNN_NNLIST, NUM_THREADS, IMPUTE_BY_KNN, MAX_ROW_WORD, ROW_WORD_SEP, MAX_COL_WORD, COL_WORD_SEP));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_mmutil_compute_scores
Rcpp::List rcpp_mmutil_compute_scores(const std::string mtx_file, Rcpp::Nullable<const std::string> row_file, Rcpp::Nullable<const std::string> col_file);
RcppExport SEXP _mmutilR_rcpp_mmutil_compute_scores(SEXP mtx_fileSEXP, SEXP row_fileSEXP, SEXP col_fileSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string >::type mtx_file(mtx_fileSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<const std::string> >::type row_file(row_fileSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<const std::string> >::type col_file(col_fileSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_mmutil_compute_scores(mtx_file, row_file, col_file));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_mmutil_simulate_poisson_mixture
Rcpp::List rcpp_mmutil_simulate_poisson_mixture(const Rcpp::List r_mu_list, const std::size_t Ncell, const std::string output, const float dir_alpha, const float gam_alpha, const float gam_beta, const std::size_t rseed, const std::size_t MAX_COL_WORD, const char COL_WORD_SEP);
RcppExport SEXP _mmutilR_rcpp_mmutil_simulate_poisson_mixture(SEXP r_mu_listSEXP, SEXP NcellSEXP, SEXP outputSEXP, SEXP dir_alphaSEXP, SEXP gam_alphaSEXP, SEXP gam_betaSEXP, SEXP rseedSEXP, SEXP MAX_COL_WORDSEXP, SEXP COL_WORD_SEPSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_mu_list(r_mu_listSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type Ncell(NcellSEXP);
    Rcpp::traits::input_parameter< const std::string >::type output(outputSEXP);
    Rcpp::traits::input_parameter< const float >::type dir_alpha(dir_alphaSEXP);
    Rcpp::traits::input_parameter< const float >::type gam_alpha(gam_alphaSEXP);
    Rcpp::traits::input_parameter< const float >::type gam_beta(gam_betaSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type rseed(rseedSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type MAX_COL_WORD(MAX_COL_WORDSEXP);
    Rcpp::traits::input_parameter< const char >::type COL_WORD_SEP(COL_WORD_SEPSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_mmutil_simulate_poisson_mixture(r_mu_list, Ncell, output, dir_alpha, gam_alpha, gam_beta, rseed, MAX_COL_WORD, COL_WORD_SEP));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_mmutil_simulate_poisson
Rcpp::List rcpp_mmutil_simulate_poisson(const Eigen::MatrixXf mu, const Eigen::VectorXf rho, const std::string output, Rcpp::Nullable<Rcpp::IntegerVector> r_indv, const std::size_t rseed, const std::size_t MAX_COL_WORD, const char COL_WORD_SEP);
RcppExport SEXP _mmutilR_rcpp_mmutil_simulate_poisson(SEXP muSEXP, SEXP rhoSEXP, SEXP outputSEXP, SEXP r_indvSEXP, SEXP rseedSEXP, SEXP MAX_COL_WORDSEXP, SEXP COL_WORD_SEPSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::MatrixXf >::type mu(muSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXf >::type rho(rhoSEXP);
    Rcpp::traits::input_parameter< const std::string >::type output(outputSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::IntegerVector> >::type r_indv(r_indvSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type rseed(rseedSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type MAX_COL_WORD(MAX_COL_WORDSEXP);
    Rcpp::traits::input_parameter< const char >::type COL_WORD_SEP(COL_WORD_SEPSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_mmutil_simulate_poisson(mu, rho, output, r_indv, rseed, MAX_COL_WORD, COL_WORD_SEP));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_mmutil_svd
Rcpp::List rcpp_mmutil_svd(const std::string mtx_file, const std::size_t RANK, const bool TAKE_LN, const double TAU, const double COL_NORM, const std::size_t EM_ITER, const double EM_TOL, const std::size_t LU_ITER, const std::string row_weight_file, const std::size_t NUM_THREADS, const std::size_t BLOCK_SIZE);
RcppExport SEXP _mmutilR_rcpp_mmutil_svd(SEXP mtx_fileSEXP, SEXP RANKSEXP, SEXP TAKE_LNSEXP, SEXP TAUSEXP, SEXP COL_NORMSEXP, SEXP EM_ITERSEXP, SEXP EM_TOLSEXP, SEXP LU_ITERSEXP, SEXP row_weight_fileSEXP, SEXP NUM_THREADSSEXP, SEXP BLOCK_SIZESEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string >::type mtx_file(mtx_fileSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type RANK(RANKSEXP);
    Rcpp::traits::input_parameter< const bool >::type TAKE_LN(TAKE_LNSEXP);
    Rcpp::traits::input_parameter< const double >::type TAU(TAUSEXP);
    Rcpp::traits::input_parameter< const double >::type COL_NORM(COL_NORMSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type EM_ITER(EM_ITERSEXP);
    Rcpp::traits::input_parameter< const double >::type EM_TOL(EM_TOLSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type LU_ITER(LU_ITERSEXP);
    Rcpp::traits::input_parameter< const std::string >::type row_weight_file(row_weight_fileSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type NUM_THREADS(NUM_THREADSSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type BLOCK_SIZE(BLOCK_SIZESEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_mmutil_svd(mtx_file, RANK, TAKE_LN, TAU, COL_NORM, EM_ITER, EM_TOL, LU_ITER, row_weight_file, NUM_THREADS, BLOCK_SIZE));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_mmutilR_rcpp_mmutil_bbknn_svd", (DL_FUNC) &_mmutilR_rcpp_mmutil_bbknn_svd, 16},
    {"_mmutilR_rcpp_mmutil_annotate_columns", (DL_FUNC) &_mmutilR_rcpp_mmutil_annotate_columns, 18},
    {"_mmutilR_rcpp_mmutil_merge_file_sets", (DL_FUNC) &_mmutilR_rcpp_mmutil_merge_file_sets, 13},
    {"_mmutilR_rcpp_mmutil_copy_selected_rows", (DL_FUNC) &_mmutilR_rcpp_mmutil_copy_selected_rows, 9},
    {"_mmutilR_rcpp_mmutil_copy_selected_columns", (DL_FUNC) &_mmutilR_rcpp_mmutil_copy_selected_columns, 7},
    {"_mmutilR_rcpp_mmutil_build_index", (DL_FUNC) &_mmutilR_rcpp_mmutil_build_index, 2},
    {"_mmutilR_rcpp_mmutil_read_index", (DL_FUNC) &_mmutilR_rcpp_mmutil_read_index, 1},
    {"_mmutilR_rcpp_mmutil_check_index", (DL_FUNC) &_mmutilR_rcpp_mmutil_check_index, 2},
    {"_mmutilR_rcpp_mmutil_info", (DL_FUNC) &_mmutilR_rcpp_mmutil_info, 1},
    {"_mmutilR_rcpp_mmutil_write_mtx", (DL_FUNC) &_mmutilR_rcpp_mmutil_write_mtx, 2},
    {"_mmutilR_rcpp_mmutil_read_columns_sparse", (DL_FUNC) &_mmutilR_rcpp_mmutil_read_columns_sparse, 6},
    {"_mmutilR_rcpp_mmutil_read_columns", (DL_FUNC) &_mmutilR_rcpp_mmutil_read_columns, 6},
    {"_mmutilR_rcpp_mmutil_match_files", (DL_FUNC) &_mmutilR_rcpp_mmutil_match_files, 15},
    {"_mmutilR_rcpp_mmutil_network_topic_data", (DL_FUNC) &_mmutilR_rcpp_mmutil_network_topic_data, 20},
    {"_mmutilR_rcpp_mmutil_aggregate_pairwise", (DL_FUNC) &_mmutilR_rcpp_mmutil_aggregate_pairwise, 22},
    {"_mmutilR_rcpp_mmutil_aggregate", (DL_FUNC) &_mmutilR_rcpp_mmutil_aggregate, 22},
    {"_mmutilR_rcpp_mmutil_compute_scores", (DL_FUNC) &_mmutilR_rcpp_mmutil_compute_scores, 3},
    {"_mmutilR_rcpp_mmutil_simulate_poisson_mixture", (DL_FUNC) &_mmutilR_rcpp_mmutil_simulate_poisson_mixture, 9},
    {"_mmutilR_rcpp_mmutil_simulate_poisson", (DL_FUNC) &_mmutilR_rcpp_mmutil_simulate_poisson, 7},
    {"_mmutilR_rcpp_mmutil_svd", (DL_FUNC) &_mmutilR_rcpp_mmutil_svd, 11},
    {NULL, NULL, 0}
};

RcppExport void R_init_mmutilR(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
