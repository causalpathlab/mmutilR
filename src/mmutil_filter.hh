#include "mmutil.hh"
#include "mmutil_io.hh"
#include "mmutil_stat.hh"

#ifndef MMUTIL_FILTER_HH_
#define MMUTIL_FILTER_HH_

int filter_col_by_nnz(const Index column_threshold,  //
                      const std::string mtx_file,    //
                      const std::string column_file, //
                      const std::string output);

#endif
