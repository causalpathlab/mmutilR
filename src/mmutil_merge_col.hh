#include "mmutil.hh"
#include "mmutil_io.hh"

#ifndef MMUTIL_MERGE_COL_HH_
#define MMUTIL_MERGE_COL_HH_

int run_merge_col(const std::vector<std::string> &glob_rows, //
                  const Index column_threshold,              //
                  const std::string output,                  //
                  const std::vector<std::string> mtx_files,  //
                  const std::vector<std::string> row_files,  //
                  const std::vector<std::string> col_files);

////////////////////////////////
// lightweight column counter //
////////////////////////////////

struct col_counter_on_valid_rows_t {
    using index_t = Index;
    using scalar_t = Scalar;
    using index_map_t = std::unordered_map<index_t, index_t>;

    explicit col_counter_on_valid_rows_t(const index_map_t &_valid_rows)
        : valid_rows(_valid_rows)
    {
        max_row = 0;
        max_col = 0;
        max_elem = 0;
    }

    void eval_after_header(const index_t r, const index_t c, const index_t e)
    {
        std::tie(max_row, max_col, max_elem) = std::make_tuple(r, c, e);
        Col_N.resize(max_col);
        std::fill(Col_N.begin(), Col_N.end(), 0);
    }

    void eval(const index_t row, const index_t col, const scalar_t weight)
    {
        if (row < max_row && col < max_col && is_valid(row) &&
            std::abs(weight) > EPS) {
            Col_N[col]++;
        }
    }

    void eval_end_of_file()
    {
        // TLOG("Found " << Col_N.sum() << std::endl);
    }

    static constexpr scalar_t EPS = 1e-8;
    const index_map_t &valid_rows;

    Index max_row;
    Index max_col;
    Index max_elem;

    std::vector<index_t> Col_N;

    inline bool is_valid(const index_t row)
    {
        return valid_rows.count(row) > 0;
    }
};

///////////////////
// global copier //
///////////////////

struct glob_triplet_copier_t {
    using index_t = Index;
    using scalar_t = Scalar;
    using index_map_t = std::unordered_map<index_t, index_t>;

    explicit glob_triplet_copier_t(
        obgzf_stream &_ofs,            // output stream
        const index_map_t &_remap_row, // row mapper
        const index_map_t &_remap_col) // column mapper
        : ofs(_ofs)
        , remap_row(_remap_row)
        , remap_col(_remap_col)
    {
        max_row = 0;
        max_col = 0;
        max_elem = 0;
        ASSERT(remap_row.size() > 0, "Empty Remap");
        ASSERT(remap_col.size() > 0, "Empty Remap");
    }

    void eval_after_header(const index_t r, const index_t c, const index_t e)
    {
        // nothing
    }

    void eval(const index_t row, const index_t col, const scalar_t weight)
    {
        if (remap_col.count(col) > 0 && remap_row.count(row) > 0) {
            // fix zero-based to one-based
            const index_t i = remap_row.at(row) + 1;
            const index_t j = remap_col.at(col) + 1;
            ofs << i << FS << j << FS << weight << std::endl;
        }
    }

    void eval_end_of_file()
    {
        // nothing
    }

    static constexpr char FS = ' ';

    obgzf_stream &ofs;
    const index_map_t &remap_row;
    const index_map_t &remap_col;

    Index max_row;
    Index max_col;
    Index max_elem;
};

#endif
