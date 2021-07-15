// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// rcpp_copy_selected_rows
int rcpp_copy_selected_rows(const std::string mtx_file, const std::string row_file, const std::string col_file, const Rcpp::StringVector& r_selected, const std::string output);
RcppExport SEXP _mmutilR_rcpp_copy_selected_rows(SEXP mtx_fileSEXP, SEXP row_fileSEXP, SEXP col_fileSEXP, SEXP r_selectedSEXP, SEXP outputSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string >::type mtx_file(mtx_fileSEXP);
    Rcpp::traits::input_parameter< const std::string >::type row_file(row_fileSEXP);
    Rcpp::traits::input_parameter< const std::string >::type col_file(col_fileSEXP);
    Rcpp::traits::input_parameter< const Rcpp::StringVector& >::type r_selected(r_selectedSEXP);
    Rcpp::traits::input_parameter< const std::string >::type output(outputSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_copy_selected_rows(mtx_file, row_file, col_file, r_selected, output));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_copy_selected_columns
int rcpp_copy_selected_columns(const std::string mtx_file, const std::string row_file, const std::string col_file, const Rcpp::StringVector& r_selected, const std::string output);
RcppExport SEXP _mmutilR_rcpp_copy_selected_columns(SEXP mtx_fileSEXP, SEXP row_fileSEXP, SEXP col_fileSEXP, SEXP r_selectedSEXP, SEXP outputSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string >::type mtx_file(mtx_fileSEXP);
    Rcpp::traits::input_parameter< const std::string >::type row_file(row_fileSEXP);
    Rcpp::traits::input_parameter< const std::string >::type col_file(col_fileSEXP);
    Rcpp::traits::input_parameter< const Rcpp::StringVector& >::type r_selected(r_selectedSEXP);
    Rcpp::traits::input_parameter< const std::string >::type output(outputSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_copy_selected_columns(mtx_file, row_file, col_file, r_selected, output));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_build_mmutil_index
int rcpp_build_mmutil_index(const std::string mtx_file, const std::string index_file);
RcppExport SEXP _mmutilR_rcpp_build_mmutil_index(SEXP mtx_fileSEXP, SEXP index_fileSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string >::type mtx_file(mtx_fileSEXP);
    Rcpp::traits::input_parameter< const std::string >::type index_file(index_fileSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_build_mmutil_index(mtx_file, index_file));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_read_mmutil_index
Rcpp::NumericVector rcpp_read_mmutil_index(const std::string index_file);
RcppExport SEXP _mmutilR_rcpp_read_mmutil_index(SEXP index_fileSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string >::type index_file(index_fileSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_read_mmutil_index(index_file));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_check_index_tab
int rcpp_check_index_tab(const std::string mtx_file, const Rcpp::NumericVector& index_tab);
RcppExport SEXP _mmutilR_rcpp_check_index_tab(SEXP mtx_fileSEXP, SEXP index_tabSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string >::type mtx_file(mtx_fileSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type index_tab(index_tabSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_check_index_tab(mtx_file, index_tab));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_read_columns
Rcpp::NumericMatrix rcpp_read_columns(const std::string mtx_file, const Rcpp::NumericVector& memory_location, const Rcpp::NumericVector& column_index);
RcppExport SEXP _mmutilR_rcpp_read_columns(SEXP mtx_fileSEXP, SEXP memory_locationSEXP, SEXP column_indexSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string >::type mtx_file(mtx_fileSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type memory_location(memory_locationSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type column_index(column_indexSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_read_columns(mtx_file, memory_location, column_index));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_mmutilR_rcpp_copy_selected_rows", (DL_FUNC) &_mmutilR_rcpp_copy_selected_rows, 5},
    {"_mmutilR_rcpp_copy_selected_columns", (DL_FUNC) &_mmutilR_rcpp_copy_selected_columns, 5},
    {"_mmutilR_rcpp_build_mmutil_index", (DL_FUNC) &_mmutilR_rcpp_build_mmutil_index, 2},
    {"_mmutilR_rcpp_read_mmutil_index", (DL_FUNC) &_mmutilR_rcpp_read_mmutil_index, 1},
    {"_mmutilR_rcpp_check_index_tab", (DL_FUNC) &_mmutilR_rcpp_check_index_tab, 2},
    {"_mmutilR_rcpp_read_columns", (DL_FUNC) &_mmutilR_rcpp_read_columns, 3},
    {NULL, NULL, 0}
};

RcppExport void R_init_mmutilR(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
