#include "mmutil.hh"
#include "mmutil_stat.hh"
#include "mmutil_bgzf_util.hh"
#include "mmutil_util.hh"
#include "mmutil_index.hh"
#include "mmutil_io.hh"

#include <algorithm>
#include <functional>
#include <string>
#include <unordered_set>

#ifndef MMUTIL_SELECT_HH_
#define MMUTIL_SELECT_HH_

/**
   @param mtx_file
   @param full_row_file
   @param _selected
   @param output
 */
template <typename STRVEC>
int
copy_selected_rows(const std::string mtx_file,
                   const std::string full_row_file,
                   const STRVEC &_selected,
                   const std::string output)
{

    using namespace mmutil::io;

    using Str = std::string;

    using copier_t =
        triplet_copier_remapped_rows_t<obgzf_stream, Index, Scalar>;

    using index_map_t = copier_t::index_map_t;

    std::vector<Str> features(0);
    CHK_RET_(read_vector_file(full_row_file, features),
             "Failed to read features");

    std::unordered_set<Str> selected(_selected.begin(), _selected.end());

    row_stat_collector_t collector;
    visit_matrix_market_file(mtx_file, collector);

    std::vector<Index> Nvec;
    std_vector(collector.Row_N, Nvec);

    const Index max_row = collector.max_row;
    const Index max_col = collector.max_col;

    std::vector<Index> rows(max_row);
    std::iota(std::begin(rows), std::end(rows), 0);
    std::vector<Index> valid_rows;
    auto _found = [&](const Index j) {
        return selected.count(features[j]) > 0;
    };
    std::copy_if(rows.begin(),
                 rows.end(),
                 std::back_inserter(valid_rows),
                 _found);

    index_map_t remap;
    Index i = 0;
    Index NNZ = 0;
    std::vector<Str> out_features;
    for (Index old_index : valid_rows) {
        remap[old_index] = i;
        Index j = valid_rows[i];
        out_features.emplace_back(features[j]);
        NNZ += Nvec[old_index];
        ++i;
    }

    Str output_feature_file = output + ".rows.gz";
    write_vector_file(output_feature_file, out_features);

    TLOG("Created valid row names");

    Str output_mtx_file = output + ".mtx.gz";
    copier_t copier(output_mtx_file, remap, NNZ);
    visit_matrix_market_file(mtx_file, copier);

    TLOG("Finished copying submatrix data");

    // std::string idx_file = output_mtx_file + ".index";
    // CHK_RET_(mmutil::index::build_mmutil_index(output_mtx_file, idx_file),
    //             "Failed to construct an index file: " << idx_file);

    TLOG("Done");
    return EXIT_SUCCESS;
}

/**
   @param mtx_file
   @param full_column_file
   @param selected_column_file
   @param output
 */
template <typename STRVEC>
int
copy_selected_columns(const std::string mtx_file,
                      const std::string full_column_file,
                      const STRVEC &_selected,
                      const std::string output)
{
    using namespace mmutil::io;
    using Str = std::string;
    using copier_t =
        triplet_copier_remapped_cols_t<obgzf_stream, Index, Scalar>;

    std::unordered_set<Str> selected(_selected.begin(), _selected.end());

    std::vector<Str> full_column_names(0);
    CHK_RET_(read_vector_file(full_column_file, full_column_names),
             "Failed to read column names");

    col_stat_collector_t collector;
    visit_matrix_market_file(mtx_file, collector);
    const IntVec &nnz_col = collector.Col_N;
    const Index max_row = collector.max_row, max_col = collector.max_col;

    ASSERT_RET(full_column_names.size() >= max_col,
               "Insufficient number of columns");

    std::vector<Index> cols(max_col);
    std::iota(std::begin(cols), std::end(cols), 0);
    std::vector<Index> valid_cols;
    auto _found = [&](const Index j) {
        return selected.count(full_column_names.at(j)) > 0;
    };
    std::copy_if(cols.begin(),
                 cols.end(),
                 std::back_inserter(valid_cols),
                 _found);

    TLOG("Found " << valid_cols.size() << " columns");

    copier_t::index_map_t remap;

    std::vector<Str> out_column_names;
    std::vector<Index> index_out(valid_cols.size());
    std::vector<Scalar> out_scores;
    Index i = 0;
    Index NNZ = 0;
    for (Index old_index : valid_cols) {
        remap[old_index] = i;
        out_column_names.push_back(full_column_names.at(old_index));

        NNZ += nnz_col(old_index);
        ++i;
    }

    TLOG("Created valid column names");

    const Str output_column_file = output + ".cols.gz";
    const Str output_mtx_file = output + ".mtx.gz";

    write_vector_file(output_column_file, out_column_names);

    copier_t copier(output_mtx_file, remap, NNZ);
    visit_matrix_market_file(mtx_file, copier);

    TLOG("Finished copying submatrix data");
    return EXIT_SUCCESS;
}

#endif
