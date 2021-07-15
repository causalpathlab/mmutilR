#include "mmutil_util.hh"

namespace mmutil { namespace bgzf {

int
convert_bgzip(std::string in_file)
{
    ASSERT_RET(file_exists(in_file),
               "Check if this file does exist: " << in_file);

    if (is_file_bgz(in_file)) {
        // WLOG("This file is already bgz: " << in_file);
        return EXIT_SUCCESS;
    }

    TLOG("Converting to bgzip format...");

    std::string temp_file = in_file + "_temp";
    std::rename(in_file.c_str(), temp_file.c_str());

    obgzf_stream ofs(in_file.c_str(), std::ios::out);

    ASSERT_RET(ofs, "Failed to open bgzip OFS");
    Index lines = 0;

    if (is_file_gz(in_file)) {
        igzstream ifs(temp_file.c_str(), std::ios::in);
        ASSERT_RET(ifs, "Failed to open IFS: " << temp_file);
        lines = convert_bgzip(ifs, ofs);
        ifs.close();
    } else {
        std::ifstream ifs(temp_file.c_str(), std::ios::in);
        ASSERT_RET(ifs, "Failed to open IFS: " << temp_file);
        lines = convert_bgzip(ifs, ofs);
        ifs.close();
    }

    ofs.close();

    TLOG("Created bgzip: " << in_file << " [" << lines << "] lines");

    return EXIT_SUCCESS;
}

}} // namespace
