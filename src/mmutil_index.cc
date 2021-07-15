#include "mmutil_index.hh"

namespace mmutil { namespace index {

using namespace mmutil::bgzf;

///////////////////////////////////////
// index bgzipped matrix market file //
///////////////////////////////////////

int build_mmutil_index(std::string mtx_file,        // bgzip file
                       std::string index_file = "") // index file
{

    if (index_file.length() == 0) {
        index_file = mtx_file + ".index";
    }

    if (bgzf_is_bgzf(mtx_file.c_str()) != 1) {
        ELOG("This file is not bgzipped: " << mtx_file);
        return EXIT_FAILURE;
    }

    if (file_exists(index_file)) {
        WLOG("Index file exists: " << index_file);
        return EXIT_SUCCESS;
    }

    mm_info_reader_t info;
    CHECK(peek_bgzf_header(mtx_file, info));

    mm_column_indexer_t indexer;
    CHECK(visit_bgzf(mtx_file, indexer));

    const mm_column_indexer_t::map_t &_map = indexer();

    //////////////////////////////////////////
    // Check if we can find all the columns //
    //////////////////////////////////////////

    TLOG("Check the index size: " << _map.size() << " vs. " << info.max_col);

    const Index sz = _map.size();
    const Index last_col = sz > 0 ? std::get<0>(_map[sz - 1]) : 0;

    if (last_col != (info.max_col - 1)) {
        ELOG("Failed to index all the columns: " << last_col << " < "
                                                 << (info.max_col - 1));
        ELOG("Filter out empty columns using `mmutil_filter_col`");
        return EXIT_FAILURE;
    }

    TLOG("Writing " << index_file << "...");

    ogzstream ofs(index_file.c_str(), std::ios::out);
    write_tuple_stream(ofs, _map);
    ofs.close();

    TLOG("Built the file: " << index_file);

    return EXIT_SUCCESS;
}

int
read_mmutil_index(std::string index_file, std::vector<Index> &_index)
{
    _index.clear();
    std::vector<std::tuple<Index, Index>> temp;
    igzstream ifs(index_file.c_str(), std::ios::in);
    int ret = read_pair_stream(ifs, temp);
    ifs.close();

    const Index N = temp.size();

    if (N < 1)
        return EXIT_FAILURE;

    Index MaxIdx = 0;
    for (auto pp : temp) {
        MaxIdx = std::max(std::get<0>(pp), MaxIdx);
    }

    // Fill in missing locations
    _index.resize(MaxIdx + 1);
    std::fill(std::begin(_index), std::end(_index), MISSING_POS);

    for (auto pp : temp) {
        _index[std::get<0>(pp)] = std::get<1>(pp);
    }

    // Update missing spots with the next one
    for (Index j = 0; j < (MaxIdx - 1); ++j) {
        // ELOG("j = " << j);
        if (_index[j] == MISSING_POS)
            _index[j] = _index[j + 1];
    }

    TLOG("Read " << MaxIdx << " indexes");
    return ret;
}

/**
   @param mtx_file matrix market file
   @param index_tab a vector of index pairs
*/
int
check_index_tab(std::string mtx_file, std::vector<Index> &index_tab)
{
    mm_info_reader_t info;
    CHECK(peek_bgzf_header(mtx_file, info));

    if (index_tab.size() < info.max_col) {
        return EXIT_FAILURE;
    }

    Index nerr = 0;
    _index_checker_t checker;
    for (Index j = 0; j < (info.max_col - 1); ++j) {
        const Index beg = index_tab[j];
        const Index end = index_tab[j];
        visit_bgzf_block(mtx_file, beg, end, checker);

        if (checker.missing(j)) {
            WLOG("Found an empty column: " << j);
            continue;
        }

        if (!checker.check(j)) {
            nerr++;
            ELOG("Expected: " << j << " at " << beg
                              << ", but found: " << checker.found());
        }
    }

    if (nerr > 0)
        return EXIT_FAILURE;

    return EXIT_SUCCESS;
}

} // namespace index
} // namespace mmutil
