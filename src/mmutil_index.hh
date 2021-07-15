#include "mmutil.hh"
#include "io.hh"
#include "mmutil_bgzf_util.hh"

#ifdef __cplusplus
extern "C" {
#endif

#include "bgzf.h"
#include "kstring.h"

#ifdef __cplusplus
}
#endif

#include <unordered_map>

/////////////////////////////////////////
// Matrix market column index format:  //
// column_index <space> memory_address //
/////////////////////////////////////////

#ifndef MMUTIL_INDEX_HH_
#define MMUTIL_INDEX_HH_

namespace mmutil { namespace index {

using namespace mmutil::bgzf;

/**
   @param bgz_file : bgzipped mtx file
   @param idx_file : index file for the bgz file
*/
int build_mmutil_index(std::string bgz_file, std::string idx_file);

/**
   @param idx_file : index file for the bgz file
   @param idx      : index map (a vector of memory locations)
*/
int read_mmutil_index(std::string idx_file, std::vector<Index> &idx);

/**
   @param mtx_file matrix market file
   @param index_tab a vector of index pairs
*/
int check_index_tab(std::string mtx_file, std::vector<Index> &index_tab);

////////////////////////////////////////////////////////////////

struct mm_column_indexer_t {

    explicit mm_column_indexer_t()
    {
        first_off = 0;
        last_off = 0;
        lineno = 0;
        last_col = 0;
        fp_set = false;
        col2file.clear();
    }

    void set_file(BGZF *_fp)
    {
        fp = _fp;
        fp_set = true;
    }

    void eval_after_header(Index max_row, Index max_col, Index max_nnz)
    {
        ASSERT(fp_set, "BGZF file pointer must be set");
        TLOG("#Rows: " << max_row << ", #Cols: " << max_col
                       << ", #NZ: " << max_nnz);
        last_col = 0; // coordinate index
        first_off = last_off = bgzf_tell(fp);
        col2file.reserve(max_col);
    }

    void eval(Index row, Index col, Scalar weight)
    {

        if (lineno == 0) {  // first column position &
            last_col = col; // offset are already found
            col2file.emplace_back(std::make_tuple(col, first_off));
        }

        if (col != last_col) { // the last one was a change point

            ASSERT(col > last_col, "MTX must be sorted by columns");

            Index save_off = bgzf_tell(fp);
            ASSERT(save_off >= last_off, "corrupted");
            col2file.emplace_back(std::make_tuple(col, last_off));
            last_col = col; // prepare for the next
        }
        last_off = bgzf_tell(fp);

        ++lineno;
    }

    void eval_end_of_file()
    {
        fp_set = false;
        TLOG("Finished indexing the file of " << lineno << " lines");
    }

    BGZF *fp; // little unsafe
    Index first_off;
    Index last_off;
    Index lineno;
    Index last_col;

    using map_t = std::vector<std::tuple<Index, Index>>;

    const map_t &operator()() const { return col2file; }

private:
    map_t col2file;
    bool fp_set;
};

struct mm_info_reader_t {
    using index_t = std::ptrdiff_t;
    explicit mm_info_reader_t()
    {
        max_row = 0;
        max_col = 0;
        max_elem = 0;
    }

    void set_file(BGZF *_fp) {}

    void eval_after_header(const index_t r, const index_t c, const index_t e)
    {
        max_row = r;
        max_col = c;
        max_elem = e;
    }
    index_t max_row;
    index_t max_col;
    index_t max_elem;
};

struct _index_checker_t {

    // using scalar_t = Scalar;
    // using index_t = Index;

    explicit _index_checker_t() { _found = 0; }

    void set_file(BGZF *_fp) { fp = _fp; }

    void eval_after_header(const Index r, const Index c, const Index e) {}

    void eval(const Index row, const Index col, const Scalar weight)
    {
        _found = col;
    }

    void eval_end_of_file() {}

    BGZF *fp;

public:
    bool check(const Index expected) const { return _found == expected; }

    bool missing(const Index expected) const { return _found > expected; }

    Index found() const { return _found; }

private:
    Index _found;
};

} // namespace index
} // namespace mmutil
#endif
