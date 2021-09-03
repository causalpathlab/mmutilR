#include "mmutil_filter.hh"

int
filter_col_by_nnz(const Index column_threshold,  //
                  const std::string mtx_file,    //
                  const std::string column_file, //
                  const std::string output)
{
    using namespace mmutil::io;
    using Str = std::string;
    using copier_t =
        triplet_copier_remapped_cols_t<obgzf_stream, Index, Scalar>;

    std::vector<Str> column_names(0);
    CHK_RET_(read_vector_file(column_file, column_names),
             "couldn't read the column file");

    col_stat_collector_t collector;
    visit_matrix_market_file(mtx_file, collector);

    const IntVec &nnz_col = collector.Col_N;

    const Index max_row = collector.max_row, max_col = collector.max_col;

    ASSERT_RET(column_names.size() >= max_col,
               "Insufficient number of columns");

    ///////////////////////////////////////////////////////
    // Filter out columns with too few non-zero elements //
    ///////////////////////////////////////////////////////

    std::vector<Index> cols(max_col);
    std::iota(std::begin(cols), std::end(cols), 0);
    std::vector<Index> valid_cols;
    const Scalar _cutoff = static_cast<Scalar>(column_threshold);

    std::copy_if(cols.begin(),
                 cols.end(),
                 std::back_inserter(valid_cols),
                 [&](const Index j) { return nnz_col(j) >= _cutoff; });

    TLOG("Found " << valid_cols.size()
                  << " (with the nnz >=" << column_threshold << ")");

    copier_t::index_map_t remap;

    std::vector<Str> out_column_names;
    std::vector<Index> index_out(valid_cols.size());
    std::vector<Scalar> out_scores;
    Index i = 0;
    Index NNZ = 0;
    for (Index old_index : valid_cols) {
        remap[old_index] = i;
        out_column_names.push_back(column_names.at(old_index));
        out_scores.push_back(nnz_col(old_index));
        NNZ += nnz_col(old_index);
        ++i;
    }

    ASSERT_RET(remap.size() > 0, "empty remapping");
    TLOG("Created valid column names");

    const Str output_column_file = output + ".cols.gz";
    const Str output_full_score_file = output + ".full_scores.gz";
    const Str output_score_file = output + ".scores.gz";
    const Str output_mtx_file = output + ".mtx.gz";

    if (file_exists(output_mtx_file)) {
        remove_file(output_mtx_file);
    }

    if (file_exists(output_mtx_file + ".index")) {
        remove_file(output_mtx_file + ".index");
    }

    write_vector_file(output_column_file, out_column_names);

    auto out_full_scores = std_vector(nnz_col);
    write_vector_file(output_full_score_file, out_full_scores);

    write_vector_file(output_score_file, out_scores);

    copier_t copier(output_mtx_file, remap, NNZ);
    visit_matrix_market_file(mtx_file, copier);

    return EXIT_SUCCESS;
}
