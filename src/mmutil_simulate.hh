#include <getopt.h>
#include <cstdio>
#include <random>
#include "mmutil.hh"
#include "mmutil_index.hh"

// #include "stat.hh"
#include "std_util.hh"
#include "bgzstream.hh"

// [[Rcpp::depends(dqrng, sitmo, BH)]]
#include <dqrng.h>
#include <dqrng_distribution.h>
#include <boost/random/binomial_distribution.hpp>
#include <boost/random/poisson_distribution.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <xoshiro.h>

#ifndef MMUTIL_SIMULATE_HH_
#define MMUTIL_SIMULATE_HH_

///////////////////////////////////////////
// - Given the model parameters: mu, rho //
// - Simulate full matrix Y[g, j]        //
// - Write out to the local file         //
// - Combine them by streaming	         //
///////////////////////////////////////////

/// mu: D x 1
/// rho: N x 1
/// col_offset: column index offset
/// ofs: write out sparse triplets here
/// returns number of non-zero elements
template <typename OFS, typename RNG>
Index
sample_poisson_data(const Vec mu,
                    const Vec rho,
                    const Index col_offset,
                    OFS &ofs,
                    RNG &rng,
                    const std::string FS = " ")
{
    const Index num_cols = rho.size();
    const Index num_rows = mu.size();

    using rpois_t = boost::random::poisson_distribution<int>;
    // dqrng::xoshiro256plus rng;
    rpois_t rpois;
    inf_zero_op<Vec> inf_zero; // remove inf -> 0

    Vec temp(num_rows);

    Index nnz = 0;

    for (Index j = 0; j < num_cols; ++j) {
        const Scalar r = rho(j);

        temp = mu.unaryExpr(inf_zero).unaryExpr(
            [&r, &rpois, &rng](const Scalar &m) -> Scalar {
                return rpois(rng, rpois_t::param_type(r * m));
            });

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
