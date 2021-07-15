#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "eigen_util.hh"
#include "io.hh"
#include "io_visitor.hh"
#include "gzstream.hh"
#include "strbuf.hh"
#include "tuple_util.hh"
#include "util.hh"

#ifndef EIGEN_IO_HH_
#define EIGEN_IO_HH_

namespace eigen_io {

struct row_name_vec_t {
    using type = std::vector<std::string>;
    explicit row_name_vec_t(type &_val)
        : val(_val)
    {
    }
    type &val;
};

struct col_name_vec_t {
    using type = std::vector<std::string>;
    explicit col_name_vec_t(type &_val)
        : val(_val)
    {
    }
    type &val;
};

struct row_index_map_t {
    using type = std::unordered_map<std::string, std::ptrdiff_t>;
    explicit row_index_map_t(type &_val)
        : val(_val)
    {
    }
    type &val;
};

struct col_index_map_t {
    using type = std::unordered_map<std::string, std::ptrdiff_t>;
    explicit col_index_map_t(type &_val)
        : val(_val)
    {
    }
    type &val;
};

template <typename IFS, typename MATTYPE>
void
read_named_eigen_sparse_stream(IFS &ifs,                      //
                               row_name_vec_t row_name_vec,   //
                               col_name_vec_t col_name_vec,   //
                               row_index_map_t row_index_map, //
                               col_index_map_t col_index_map, //
                               MATTYPE &mat)
{
    row_name_vec_t::type &i_name = row_name_vec.val;
    col_name_vec_t::type &j_name = col_name_vec.val;
    row_index_map_t::type &i_index = row_index_map.val;
    col_index_map_t::type &j_index = col_index_map.val;

    //////////////////////////
    // Finite state machine //
    //////////////////////////

    typedef enum _state_t { S_COMMENT, S_WORD, S_EOW, S_EOL } state_t;
    const char eol = '\n';
    const char comment = '%';

    /////////////////////////
    // use buffered stream //
    /////////////////////////

    std::istreambuf_iterator<char> END;
    std::istreambuf_iterator<char> it(ifs);

    using Scalar = typename MATTYPE::Scalar;
    using Index = typename MATTYPE::Index;

    strbuf_t strbuf;
    state_t state = S_EOL;

    Index num_cols = 0;
    std::string i_str, j_str;
    Scalar weight;

    // std::unordered_map<std::string, Index> i_index;
    // std::unordered_map<std::string, Index> j_index;

    // std::vector<std::string> i_name;
    // std::vector<std::string> j_name;

    i_index.clear();
    j_index.clear();
    i_name.clear();
    j_name.clear();

    Index ii = 0, jj = 0;

    auto read_triplet = [&]() {
        switch (num_cols) {
        case 0:
            strbuf.take_string(i_str);
            break;
        case 1:
            strbuf.take_string(j_str);
            break;
        case 2:
            weight = strbuf.take_float();
            break;
        }
        state = S_EOW;
        strbuf.clear();
    };

    using Triplet = Eigen::Triplet<Scalar>;

    std::vector<Triplet> triplets;

    auto insert_triplet = [&]() {
        if (i_index.count(i_str) == 0) {
            i_index[i_str] = ii;
            i_name.push_back(i_str);
            ii++;
        }

        if (j_index.count(j_str) == 0) {
            j_index[j_str] = jj;
            j_name.push_back(j_str);
            jj++;
        }

        triplets.push_back(
            Triplet(i_index.at(i_str), j_index.at(j_str), weight));
    };

    Index num_rows = 0;

    for (; it != END; ++it) {
        char c = *it;

        // Skip the comment line. It doesn't count toward the line
        // count, and we don't bother reading the content.
        if (state == S_COMMENT) {
            if (c == eol)
                state = S_EOL;
            continue;
        }

        if (c == comment) {
            state = S_COMMENT;
            WLOG("Found a commented line in the middle: row = " << num_rows);
            continue;
        }

        if (c == eol) {
            if (state == S_WORD) {
                read_triplet();
                num_cols++;
            }

            state = S_EOL;

            if (num_cols == 3) {
                insert_triplet();
            } else {
                WLOG("Found corrupted a triplet: row = " << num_rows);
            }
            num_cols = 0;
            num_rows++;

        } else if (isspace(c) && strbuf.size() > 0) {
            read_triplet();
            num_cols++;
        } else {
            strbuf.add(c);
            state = S_WORD;
        }
    }

    mat.resize(i_name.size(), j_name.size());
    mat.reserve(triplets.size());
    mat.setFromTriplets(triplets.begin(), triplets.end());

    TLOG("Read " << i_name.size() << " rows");
    TLOG("Read " << j_name.size() << " cols");
    TLOG("Read " << triplets.size() << " triplets");
}

template <typename MATTYPE>
void
read_named_eigen_sparse_file(const std::string filename,    //
                             row_name_vec_t row_name_vec,   //
                             col_name_vec_t col_name_vec,   //
                             row_index_map_t row_index_map, //
                             col_index_map_t col_index_map, //
                             MATTYPE &mat)
{
    if (is_file_gz(filename)) {
        igzstream ifs(filename.c_str(), std::ios::in);
        read_named_eigen_sparse_stream(ifs,
                                       row_name_vec,
                                       col_name_vec,
                                       row_index_map,
                                       col_index_map,
                                       mat);
        ifs.close();
    } else {
        std::ifstream ifs(filename.c_str(), std::ios::in);
        read_named_eigen_sparse_stream(ifs,
                                       row_name_vec,
                                       col_name_vec,
                                       row_index_map,
                                       col_index_map,
                                       mat);
        ifs.close();
    }
}

} // eigen io

#endif
