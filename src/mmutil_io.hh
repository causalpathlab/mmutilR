#include "mmutil.hh"
#include "mmutil_bgzf_util.hh"
#include "mmutil_index.hh"
#include "io.hh"

#ifdef __cplusplus
extern "C" {
#endif

#include "bgzf.h"
#include "kstring.h"

#ifdef __cplusplus
}
#endif

#include <unordered_map>

#ifndef MMUTIL_IO_HH_
#define MMUTIL_IO_HH_

namespace mmutil { namespace io {

using namespace mmutil::bgzf;
using namespace mmutil::index;

//////////////////////////
// read just one column //
//////////////////////////

// An example:
//
// Index target_col;
// one_column_reader_t reader(target_col);
// peek_bgzf_header(mtx_file, reader);
// Index lb_mem = idx_tab[target_col];
// Index ub_mem = 0;
// if((target_col + 1) < idx_tab.size()) ub_mem = idx_tab[target_col + 1];
// visit_bgzf_block(mtx_file, lb_mem, ub_mem, reader);
//
struct one_column_reader_t {
    using scalar_t = float;
    using index_t = std::ptrdiff_t;

    explicit one_column_reader_t(const index_t &dim, const index_t &_col)
        : data(dim, 1)
        , target_col(_col)
    {
        max_row = 0;
        max_col = 0;
        max_elem = 0;
    }

    void set_file(BGZF *_fp) { fp = _fp; }

    void eval_after_header(const index_t r, const index_t c, const index_t e)
    {
        max_row = r;
        max_col = c;
        max_elem = e;

        if (data.rows() != max_row) {
            data.resize(max_row, 1);
            data.setZero();
        }
    }

    void eval(const index_t row, const index_t col, const scalar_t weight)
    {
        if (col == target_col && row < data.rows()) {
            data(row, 0) = weight;
        }
    }

    void eval_end_of_file() { }

    BGZF *fp;

    index_t max_row;
    index_t max_col;
    index_t max_elem;
    Mat data;
    const index_t &target_col;
};

/////////////////////////////////////////////////////////////
// read matrix market triplets and construct sparse matrix //
/////////////////////////////////////////////////////////////

template <typename T>
struct _triplet_reader_remapped_cols_t {
    using scalar_t = float;
    using index_t = std::ptrdiff_t;
    using Triplet = T;
    using TripletVec = std::vector<T>;

    using index_map_t = std::unordered_map<index_t, index_t>;

    explicit _triplet_reader_remapped_cols_t(TripletVec &_tvec,
                                             const index_map_t &_remap,
                                             const index_t _nnz = 0)
        : Tvec(_tvec)
        , remap(_remap)
        , NNZ(_nnz)
    {
        max_row = 0;
        max_col = 0;
        max_elem = 0;
        if (NNZ > 0) {
            Tvec.reserve(NNZ);
        }
        ASSERT(remap.size() > 0, "Empty Remap");
    }

    void set_file(BGZF *_fp) { fp = _fp; }

    void eval_after_header(const index_t r, const index_t c, const index_t e)
    {
        max_row = r;
        max_col = c;
        max_elem = e;
    }

    void eval(const index_t row, const index_t col, const scalar_t weight)
    {
        if (remap.count(col) > 0) {
            Tvec.emplace_back(T(row, remap.at(col), weight));
        }
    }

    void eval_end_of_file()
    {
#ifdef DEBUG
        if (Tvec.size() < NNZ) {
            WLOG("This file may have lost elements : " << Tvec.size() << " vs. "
                                                       << NNZ);
        }
        TLOG("Tvec : " << Tvec.size() << " vs. " << NNZ << " vs. " << max_elem);
#endif
    }

    BGZF *fp;

    index_t max_row;
    index_t max_col;
    index_t max_elem;
    TripletVec &Tvec;
    const index_map_t &remap;
    const index_t NNZ;
};

template <typename T>
struct _triplet_reader_remapped_rows_cols_t {
    using scalar_t = float;
    using index_t = std::ptrdiff_t;
    using Triplet = T;
    using TripletVec = std::vector<T>;

    using index_map_t = std::unordered_map<index_t, index_t>;

    explicit _triplet_reader_remapped_rows_cols_t(TripletVec &_tvec,
                                                  const index_map_t &_remap_row,
                                                  const index_map_t &_remap_col,
                                                  const index_t _nnz = 0)
        : Tvec(_tvec)
        , remap_row(_remap_row)
        , remap_col(_remap_col)
        , NNZ(_nnz)
    {
        max_row = 0;
        max_col = 0;
        max_elem = 0;
        if (NNZ > 0) {
            Tvec.reserve(NNZ);
        }
        ASSERT(remap_row.size() > 0, "Empty Remap_Row");
        ASSERT(remap_col.size() > 0, "Empty Remap_Col");
    }

    void eval_after_header(const index_t r, const index_t c, const index_t e)
    {
        max_row = r;
        max_col = c;
        max_elem = e;
    }

    void eval(const index_t row, const index_t col, const scalar_t weight)
    {
        if (remap_col.count(col) > 0 && remap_row.count(row) > 0) {
            Tvec.emplace_back(T(remap_row.at(row), remap_col.at(col), weight));
        }
    }

    void eval_end_of_file()
    {
#ifdef DEBUG
        if (Tvec.size() < NNZ) {
            WLOG("This file may have lost elements : " << Tvec.size() << " vs. "
                                                       << NNZ);
        }
        TLOG("Tvec : " << Tvec.size() << " vs. " << NNZ << " vs. " << max_elem);
#endif
    }

    index_t max_row;
    index_t max_col;
    index_t max_elem;
    TripletVec &Tvec;
    const index_map_t &remap_row;
    const index_map_t &remap_col;
    const index_t NNZ;
};

template <typename T>
struct _triplet_reader_t {
    using scalar_t = float;
    using index_t = std::ptrdiff_t;
    using Triplet = T;
    using TripletVec = std::vector<T>;

    explicit _triplet_reader_t(TripletVec &_tvec)
        : Tvec(_tvec)
    {
        max_row = 0;
        max_col = 0;
        max_elem = 0;
        TLOG("Start reading a list of triplets");
    }

    void set_fp(BGZF *_fp) { fp = _fp; }

    void eval_after_header(const index_t r, const index_t c, const index_t e)
    {
        max_row = r;
        max_col = c;
        max_elem = e;
        Tvec.reserve(max_elem);
    }

    void eval(const index_t row, const index_t col, const scalar_t weight)
    {
        Tvec.emplace_back(T(row, col, weight));
    }

    void eval_end_of_file()
    {
        if (Tvec.size() < max_elem) {
            WLOG("This file may have lost elements : " << Tvec.size() << " vs. "
                                                       << max_elem);
        }
        TLOG("Finished reading a list of triplets");
    }

    BGZF *fp;

    index_t max_row;
    index_t max_col;
    index_t max_elem;
    TripletVec &Tvec;
};

using std_triplet_t = std::tuple<std::ptrdiff_t, std::ptrdiff_t, float>;
using std_triplet_reader_t = _triplet_reader_t<std_triplet_t>;
using eigen_triplet_reader_t = _triplet_reader_t<Eigen::Triplet<float>>;

using std_triplet_reader_remapped_cols_t =
    _triplet_reader_remapped_cols_t<std_triplet_t>;

using std_triplet_reader_remapped_rows_cols_t =
    _triplet_reader_remapped_rows_cols_t<std_triplet_t>;

using eigen_triplet_reader_remapped_cols_t =
    _triplet_reader_remapped_cols_t<Eigen::Triplet<float>>;

using eigen_triplet_reader_remapped_rows_cols_t =
    _triplet_reader_remapped_rows_cols_t<Eigen::Triplet<float>>;

template <typename IFS, typename READER>
inline auto
_read_matrix_market_stream(IFS &ifs)
{
    typename READER::TripletVec Tvec;
    READER reader(Tvec);

    visit_matrix_market_stream(ifs, reader);

    auto max_row = reader.max_row;
    auto max_col = reader.max_col;

    return std::make_tuple(Tvec, max_row, max_col);
}

template <typename READER>
inline auto
_read_matrix_market_file(const std::string filename)
{
    typename READER::TripletVec Tvec;
    READER reader(Tvec);

    visit_matrix_market_file(filename, reader);

    auto max_row = reader.max_row;
    auto max_col = reader.max_col;

    return std::make_tuple(Tvec, max_row, max_col);
}

inline auto
read_matrix_market_file(const std::string filename)
{
    return _read_matrix_market_file<std_triplet_reader_t>(filename);
}

inline auto
read_eigen_matrix_market_file(const std::string filename)
{
    return _read_matrix_market_file<eigen_triplet_reader_t>(filename);
}

template <typename IFS>
inline auto
read_matrix_market_stream(IFS &ifs)
{
    return _read_matrix_market_stream<IFS, std_triplet_reader_t>(ifs);
}

template <typename IFS>
inline auto
read_eigen_matrix_market_stream(IFS &ifs)
{
    return _read_matrix_market_stream<IFS, eigen_triplet_reader_t>(ifs);
}

/////////////
// utility //
/////////////

struct memory_block_t {
    Index lb;
    Index lb_mem;
    Index ub;
    Index ub_mem;
};

template <typename VEC1, typename VEC2>
std::vector<memory_block_t>
find_consecutive_blocks(const VEC1 &index_tab,
                        const VEC2 &subcol,
                        const Index gap = 10)
{

    const Index N = index_tab.size();
    ASSERT(N > 1, "Empty index map");

    VEC2 sorted(subcol.size());
    std::copy(subcol.begin(), subcol.end(), sorted.begin());
    std::sort(sorted.begin(), sorted.end());

    std::vector<std::tuple<Index, Index>> intervals;
    {
        Index beg = sorted[0];
        Index end = beg;

        for (Index jj = 1; jj < sorted.size(); ++jj) {
            const Index ii = sorted[jj];
            if (ii >= (end + gap)) {                  // Is it worth adding
                intervals.emplace_back(beg, end + 1); // a new block?
                beg = ii;                             // Start a new block
                end = ii;                             // with this ii
            } else {                                  //
                end = ii;                             // Extend the current one
            }                                         // to cover this point
        }                                             //
                                                      //
        if (beg <= sorted[sorted.size() - 1]) {       // Set the upper-bound
            intervals.emplace_back(beg, end + 1);     //
        }
    }

    std::vector<memory_block_t> ret;

    for (auto intv : intervals) {

        Index lb, lb_mem, ub, ub_mem = 0;
        std::tie(lb, ub) = intv;

        if (lb >= N)
            continue;

        lb_mem = index_tab[lb];

        if (ub < N) {
            ub_mem = index_tab[ub];
        }

        ret.emplace_back(memory_block_t { lb, lb_mem, ub, ub_mem });
    }

    return ret;
}

/////////////////////////////////////////
// read and write triplets selectively //
/////////////////////////////////////////

template <typename OFS, typename INDEX, typename SCALAR>
struct triplet_copier_remapped_rows_t {
    using index_t = INDEX;
    using scalar_t = SCALAR;

    //////////////////////////
    // mapping : old -> new //
    //////////////////////////

    using index_map_t = std::unordered_map<index_t, index_t>;

    explicit triplet_copier_remapped_rows_t(
        const std::string _filename, // output filename
        const index_map_t &_remap,   // valid rows
        const index_t _nnz)
        : filename(_filename)
        , remap(_remap)
        , NNZ(_nnz)
    {
        max_row = 0;
        max_col = 0;
        max_elem = 0;
        ASSERT(remap.size() > 0, "Empty Remap");
    }

    void eval_after_header(const index_t r, const index_t c, const index_t e)
    {
        std::tie(max_row, max_col, max_elem) = std::make_tuple(r, c, e);
        TLOG("Input size: " << max_row << " x " << max_col);

        const index_t new_max_row = find_new_max_row();
        const index_t new_max_elem = NNZ;
        TLOG("Reducing " << max_elem << " -> " << new_max_elem);
        elem_check = 0;

        ofs.open(filename.c_str(), std::ios::out);
        ofs << "%%MatrixMarket matrix coordinate integer general" << std::endl;
        ofs << new_max_row << FS << max_col << FS << new_max_elem << std::endl;
        TLOG("Start copying data on the selected rows: N = " << remap.size());
    }

    void eval(const index_t row, const index_t col, const scalar_t weight)
    {
        if (remap.count(row) > 0) {
            const index_t i = remap.at(row) + 1; // fix zero-based to one-based
            const index_t j = col + 1;           // fix zero-based to one-based
            ofs << i << FS << j << FS << weight << std::endl;
            elem_check++;
        }
    }

    void eval_end_of_file()
    {
        ofs.close();
        if (elem_check != NNZ) {
            WLOG("The number of non-zero elements is different:");
            WLOG(elem_check << " vs. " << NNZ);
        }
        TLOG("Finished copying data");
    }

    const std::string filename;
    const index_map_t &remap;
    const index_t NNZ;

    index_t elem_check;

    OFS ofs;
    static constexpr char FS = ' ';

    index_t max_row;
    index_t max_col;
    index_t max_elem;

private:
    index_t find_new_max_row() const
    {
        index_t ret = 0;
        std::for_each(remap.begin(),
                      remap.end(), //
                      [&ret](const auto &tt) {
                          index_t _old, _new;
                          std::tie(_old, _new) = tt;
                          index_t _new_size = _new + 1;
                          if (_new_size > ret)
                              ret = _new_size;
                      });
        return ret;
    }
};

template <typename OFS, typename INDEX, typename SCALAR>
struct triplet_copier_remapped_cols_t {
    using index_t = INDEX;
    using scalar_t = SCALAR;

    //////////////////////////
    // mapping : old -> new //
    //////////////////////////

    using index_map_t = std::unordered_map<index_t, index_t>;

    explicit triplet_copier_remapped_cols_t(const std::string _filename, //
                                            const index_map_t &_remap,   //
                                            const index_t _nnz)
        : filename(_filename)
        , remap(_remap)
        , NNZ(_nnz)
    {
        max_row = 0;
        max_col = 0;
        max_elem = 0;
        ASSERT(remap.size() > 0, "Empty Remap");
    }

    void eval_after_header(const index_t r, const index_t c, const index_t e)
    {
        std::tie(max_row, max_col, max_elem) = std::make_tuple(r, c, e);
        TLOG("Input size: " << max_row << " x " << max_col);

        const index_t new_max_col = find_new_max_col();
        const index_t new_max_elem = NNZ;

        TLOG("Reducing " << max_elem << " -> " << new_max_elem);

        elem_check = 0;

        ofs.open(filename.c_str(), std::ios::out);
        ofs << "%%MatrixMarket matrix coordinate integer general" << std::endl;
        ofs << max_row << FS << new_max_col << FS << new_max_elem << std::endl;
        TLOG("Start copying data on the selected cols: N = " << remap.size());
    }

    void eval(const index_t row, const index_t col, const scalar_t weight)
    {
        if (remap.count(col) > 0) {
            const index_t i = row + 1;           // fix zero-based to one-based
            const index_t j = remap.at(col) + 1; // fix zero-based to one-based
            ofs << i << FS << j << FS << weight << std::endl;
            elem_check++;
        }
    }

    void eval_end_of_file()
    {
        ofs.close();
        if (elem_check != NNZ) {
            WLOG("The number of non-zero elements is different:");
            WLOG(elem_check << " vs. " << NNZ);
        }
        TLOG("Finished copying data");
    }

    const std::string filename;
    const index_map_t &remap;
    const index_t NNZ;

    index_t elem_check;

    OFS ofs;
    static constexpr char FS = ' ';

    index_t max_row;
    index_t max_col;
    index_t max_elem;

private:
    index_t find_new_max_col() const
    {
        index_t ret = 0;
        std::for_each(remap.begin(),
                      remap.end(), //
                      [&ret](const auto &tt) {
                          index_t _old, _new;
                          std::tie(_old, _new) = tt;
                          index_t _new_size = _new + 1;
                          if (_new_size > ret)
                              ret = _new_size;
                      });
        return ret;
    }
};

SpMat read_eigen_sparse_subset_col(const std::string mtx_file,
                                   const Index lb,
                                   const Index ub,
                                   const Index lb_mem,
                                   const Index ub_mem);

SpMat read_eigen_sparse_subset_row_col(
    const std::string mtx_file,
    const eigen_triplet_reader_remapped_rows_cols_t::index_map_t &rows,
    const Index col_lb,
    const Index col_ub,
    const Index lb_mem,
    const Index ub_mem);

SpMat read_eigen_sparse_subset_col(const std::string mtx_file,
                                   const std::vector<Index> &index_tab,
                                   const std::vector<Index> &subcol);

SpMat read_eigen_sparse_subset_col(const std::string mtx_file,
                                   const std::string index_file,
                                   const std::vector<Index> &subcol);

SpMat read_eigen_sparse_subset_row_col(const std::string mtx_file,
                                       const std::vector<Index> &index_tab,
                                       const std::vector<Index> &subrow,
                                       const std::vector<Index> &subcol);

SpMat read_eigen_sparse_subset_row_col(const std::string mtx_file,
                                       const std::string index_file,
                                       const std::vector<Index> &subrow,
                                       const std::vector<Index> &subcol);

}} // namespace mmutil::io
#endif
