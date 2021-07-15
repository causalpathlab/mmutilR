#include "mmutil.hh"
#include "io.hh"
#include <zlib.h>
#include <string>
#include <cstdio>

#ifndef MMUTIL_UTIL_HH_
#define MMUTIL_UTIL_HH_

namespace mmutil { namespace bgzf {

/// @param ifs : input file stream
/// @param ofs : output file stream
template <typename IFS, typename OFS>
Index
convert_bgzip(IFS &ifs, OFS &ofs)
{
    Index lines = 0;
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.length() > 4) { // at least i <space> j <space> w
            ofs << line << std::endl;
            ++lines;
        } else {
            WLOG("Ignore this line in #" << lines << ": " << line);
        }
    }
    return lines;
}

/// @param in_file
int convert_bgzip(std::string in_file);

}} // namespace
#endif
