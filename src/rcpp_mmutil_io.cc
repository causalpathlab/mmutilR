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
//' @param column_index column indexes to retrieve (1-based)
//'
//' @return a dense sub-matrix
//'
// [[Rcpp::export]]
Rcpp::NumericMatrix
rcpp_read_columns(const std::string mtx_file,
                  const Rcpp::NumericVector &memory_location,
                  const Rcpp::NumericVector &column_index)
{

    using namespace mmutil::io;

    mm_info_reader_t info;
    CHECK(peek_bgzf_header(mtx_file, info));
    const Index max_row = info.max_row;
    // const Index max_col = info.max_col;

    const auto blocks = find_consecutive_blocks(memory_location, column_index);

    using triplet_reader_t = eigen_triplet_reader_remapped_cols_t;
    using Index = triplet_reader_t::index_t;

    Index lb = 0;
    Index ub = 0;
    Index max_col = 0;

    triplet_reader_t::index_map_t column_index_order; // we keep the same order
    for (auto k : column_index) {                     // of column_index

        ///////////////////////////////////////////////
        // We normally expect 1-based columns from R //
        ///////////////////////////////////////////////

        if (k > 0 && k <= info.max_col) {
            const Index kk = static_cast<const Index>(k - 1);
            column_index_order[kk] = max_col++;
        }
    }

    triplet_reader_t::TripletVec Tvec; // keep accumulating this
    Tvec.clear();

    for (auto block : blocks) {

#ifdef DEBUG
        TLOG("Visiting [" << block.lb << ", " << block.ub << ")");
#endif

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

        CHECK(visit_bgzf_block(mtx_file, block.lb_mem, block.ub_mem, reader));
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
