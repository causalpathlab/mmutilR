#include <getopt.h>
#include <unordered_map>

#include "mmutil.hh"
#include "mmutil_stat.hh"

#ifndef MMUTIL_HISTOGRAM_HH_
#define MMUTIL_HISTOGRAM_HH_

std::unordered_map<Index, Index>
take_mtx_histogram(const std::string mtx_file)
{
    histogram_collector_t collector;
    visit_matrix_market_file(mtx_file, collector);
    return collector.freq_map;
}

template <typename OFS>
void
write_results(const std::unordered_map<Index, Index> &data,
              const std::vector<Index> &keys,
              OFS &ofs,
              const std::string FS = " ")
{

    for (auto k : keys) {
        ofs << k << FS << data.at(k) << std::endl;
    }
}

int
write_histogram_results(const std::string mtx_file, const std::string output)
{
    std::unordered_map<Index, Index> freq = take_mtx_histogram(mtx_file);
    std::vector<Index> keys;
    keys.reserve(freq.size());

    for (auto x : freq) {
        keys.emplace_back(std::get<0>(x));
    }

    std::sort(std::begin(keys), std::end(keys));

    if (output == "STDOUT") {
        write_results(freq, keys, std::cout);
    } else {
        ogzstream ofs(output.c_str(), std::ios::out);
        write_results(freq, keys, ofs);
        ofs.close();
    }

    return EXIT_SUCCESS;
}

#endif
