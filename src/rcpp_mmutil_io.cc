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

//' Read a subset of columns from the data matrix
//' @param mtx_file data file
//' @param memory_location column -> memory location
//' @param r_column_index column indexes to retrieve (1-based)
//'
//' @return a dense sub-matrix
//'
//' @examples
//'
//' rr <- rgamma(100, 1, 1) # one hundred cells
//' mm <- matrix(rgamma(10 * 3, 1, 1), 10, 3)
//' data.hdr <- "test_sim"
//' .files <- mmutilR::rcpp_simulate_poisson_data(mm, rr, data.hdr)
//' data.file <- .files$mtx
//' idx.file <- .files$idx
//' mtx.idx <- mmutilR::rcpp_read_mmutil_index(idx.file)
//' Y <- as.matrix(Matrix::readMM(data.file))
//' col.pos <- c(1,13,77) # 1-based
//' yy <- mmutilR::rcpp_read_columns(data.file, mtx.idx, col.pos)
//' all(Y[, col.pos, drop = FALSE] == yy)
//' print(head(Y[, col.pos, drop = FALSE]))
//' print(head(yy))
//' unlink(list.files(pattern = data.hdr))
//'
// [[Rcpp::export]]
Rcpp::NumericMatrix
rcpp_read_columns(const std::string mtx_file,
                  const Rcpp::NumericVector &memory_location,
                  const Rcpp::NumericVector &r_column_index)
{

    using namespace mmutil::io;

    mm_info_reader_t info;
    CHK_ERR_RETM(peek_bgzf_header(mtx_file, info),
                 "Failed to read the mtx file:" << mtx_file);
    const Index max_row = info.max_row;

    TLOG("info: " << info.max_row << ", " << info.max_col << " --> "
                  << info.max_elem);

    using triplet_reader_t = eigen_triplet_reader_remapped_cols_t;
    using Index = triplet_reader_t::index_t;

    Index lb = 0;
    Index ub = 0;
    Index max_col = 0;

    ///////////////////////////////////////////////
    // We normally expect 1-based columns from R //
    ///////////////////////////////////////////////

    // convert 1-based to 0-based
    std::vector<Index> column_index;
    column_index.reserve(r_column_index.size());
    for (auto k : r_column_index) {
        if (k >= 1 && k <= info.max_col) {
            column_index.emplace_back(k - 1);
        }
    }

    ASSERT_RETM(column_index.size() > 0, "empty column index");

    triplet_reader_t::index_map_t column_index_order; // we keep the same order
    for (auto k : column_index) {                     // of column_index

        const Index kk = static_cast<const Index>(k);
        column_index_order[kk] = max_col++;
    }

    triplet_reader_t::TripletVec Tvec; // keep accumulating this
    Tvec.clear();

    const auto blocks = find_consecutive_blocks(memory_location, column_index);

    for (auto block : blocks) {

        triplet_reader_t::index_map_t loc_map;

        for (Index j = block.lb; j < block.ub; ++j) {

            /////////////////////////////////////////////////
            // Sometimes...				   //
            // we may encounter discontinuous_index vector //
            // So, we need to check again                  //
            /////////////////////////////////////////////////

            if (column_index_order.count(j) > 0) {
                loc_map[j] = column_index_order[j];
            }
        }

        triplet_reader_t reader(Tvec, loc_map);

        CHK_ERR_RETM(visit_bgzf_block(mtx_file,
                                      block.lb_mem,
                                      block.ub_mem,
                                      reader),
                     "Failed to read this block: [" << block.lb << ", "
                                                    << block.ub << ")");
    }

    TLOG("Successfully read " << blocks.size() << " block(s)");

    /////////////////////////////////////////
    // populate items in the return matrix //
    /////////////////////////////////////////

    SpMat X(max_row, max_col);
    X.setZero();
    X.reserve(Tvec.size());
    X.setFromTriplets(Tvec.begin(), Tvec.end());

    SEXP xx = Rcpp::wrap(Mat(X));
    Rcpp::NumericMatrix ret(xx);
    return ret;
}
