#include "mmutil_io.hh"

namespace mmutil { namespace io {

using namespace mmutil::bgzf;
using namespace mmutil::index;

SpMat
read_eigen_sparse_subset_col(const std::string mtx_file,
                             const std::vector<Index> &index_tab,
                             const std::vector<Index> &subcol)
{

    mm_info_reader_t info;
    CHECK(peek_bgzf_header(mtx_file, info));

    using _reader_t = eigen_triplet_reader_remapped_cols_t;
    using Index = _reader_t::index_t;
#ifdef DEBUG
    CHECK(check_index_tab(mtx_file, index_tab));
#endif

    Index max_col = 0;                   // Make sure that
    _reader_t::index_map_t subcol_order; // we keep the same order
    for (auto k : subcol) {              // of subcol
        subcol_order[k] = max_col++;
    }

    const auto blocks = find_consecutive_blocks(index_tab, subcol);

    _reader_t::TripletVec Tvec; // keep accumulating this
    Tvec.clear();
    Index max_row = info.max_row;
    for (auto block : blocks) {
        _reader_t::index_map_t loc_map;

        for (Index j = block.lb; j < block.ub; ++j) {
            //////////////////////////////////////////////////////////
            // Caution: the blocks with discontinuous subcol vector //
            //////////////////////////////////////////////////////////
            if (subcol_order.count(j) > 0)
                loc_map[j] = subcol_order[j];
        }

        _reader_t reader(Tvec, loc_map);

        CHECK(visit_bgzf_block(mtx_file, block.lb_mem, block.ub_mem, reader));
    }

    SpMat X(max_row, max_col);
    X.setZero();
    X.reserve(Tvec.size());
    X.setFromTriplets(Tvec.begin(), Tvec.end());

#ifdef DEBUG
    TLOG("Constructed a sparse matrix with m = " << X.nonZeros());
#endif

    return X;
}

SpMat
read_eigen_sparse_subset_col(const std::string mtx_file,
                             const std::string index_file,
                             const std::vector<Index> &subcol)
{
    std::vector<Index> index_tab;
    CHECK(read_mmutil_index(index_file, index_tab));
    CHECK(check_index_tab(mtx_file, index_tab));
    return read_eigen_sparse_subset_col(mtx_file, index_tab, subcol);
}

SpMat
read_eigen_sparse_subset_row_col(const std::string mtx_file,
                                 const std::string index_file,
                                 const std::vector<Index> &subrow,
                                 const std::vector<Index> &subcol)
{

    std::vector<Index> index_tab;
    CHECK(read_mmutil_index(index_file, index_tab));
    return read_eigen_sparse_subset_row_col(mtx_file,
                                            index_tab,
                                            subrow,
                                            subcol);
}

SpMat
read_eigen_sparse_subset_row_col(const std::string mtx_file,
                                 const std::vector<Index> &index_tab,
                                 const std::vector<Index> &subrow,
                                 const std::vector<Index> &subcol)
{

    using _reader_t = eigen_triplet_reader_remapped_rows_cols_t;
    using Index = _reader_t::index_t;

    Index max_col = 0;                   // Make sure that
    _reader_t::index_map_t subcol_order; // we keep the same order
    for (auto k : subcol) {              // of subcol
        subcol_order[k] = max_col++;
    }

    const auto blocks = find_consecutive_blocks(index_tab, subcol);

    _reader_t::index_map_t remap_row;
    for (Index new_index = 0; new_index < subrow.size(); ++new_index) {
        const Index old_index = subrow.at(new_index);
        remap_row[old_index] = new_index;
    }

    _reader_t::TripletVec Tvec; // keep accumulating this

    Index max_row = subrow.size();
    for (auto block : blocks) {
        _reader_t::index_map_t remap_col;
        for (Index old_index = block.lb; old_index < block.ub; ++old_index) {
            remap_col[old_index] = subcol_order[old_index];
        }
        _reader_t reader(Tvec, remap_row, remap_col);
        CHECK(visit_bgzf_block(mtx_file, block.lb_mem, block.ub_mem, reader));
    }

    SpMat X(max_row, max_col);
    X.reserve(Tvec.size());
    X.setFromTriplets(Tvec.begin(), Tvec.end());

#ifdef DEBUG
    TLOG("Constructed a sparse matrix with m = " << X.nonZeros());
#endif

    return X;
}

}} // namespace mmutil::io
