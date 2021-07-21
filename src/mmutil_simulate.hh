#include <getopt.h>
#include <cstdio>
#include <random>
#include "mmutil.hh"
#include "mmutil_index.hh"

#include "stat.hh"
#include "std_util.hh"
#include "bgzstream.hh"

#ifndef MMUTIL_SIMULATE_HH_
#define MMUTIL_SIMULATE_HH_

///////////////////////////////////////////
// - Given the model parameters: mu, rho //
// - Simulate full matrix Y[g, j]        //
// - Write out to the local file         //
// - Combine them by streaming	         //
///////////////////////////////////////////

/// @param mu D x 1
/// @param rho N x 1
/// @param col_offset column index offset
/// @param ofs write out sparse triplets here
/// @return number of non-zero elements
template <typename OFS>
Index
sample_poisson_data(const Vec mu,
                    const Vec rho,
                    const Index col_offset,
                    OFS &ofs,
                    const std::string FS = " ")
{
    const Index num_cols = rho.size();
    const Index num_rows = mu.size();

    rpois_t rpois;

    Vec temp(num_rows);

    Index nnz = 0;

    for (Index j = 0; j < num_cols; ++j) {
        const Scalar r = rho(j);

        temp = mu.unaryExpr(
            [&r, &rpois](const Scalar &m) -> Scalar { return rpois(r * m); });

        const Index col = col_offset + j + 1; // one-based

        if (temp.sum() < 1.) {   // for an empty column
            const Index row = 1; // one-based
            ofs << row << FS << col << FS << 0 << std::endl;
            nnz++; // not so true... will be removed later
            continue;
        }

        for (Index g = 0; g < num_rows; ++g) { // for each row
            if (temp(g) > 0.) {                //
                const Index row = g + 1;       // one-based
                ofs << row << FS << col << FS << temp(g) << std::endl;
                nnz++;
            }
        }
    }

    return nnz;
}

#endif
