#include "mmutil.hh"
#include <cmath>
#include <unordered_map>

#ifndef MMUTIL_STAT_HH_
#define MMUTIL_STAT_HH_

//////////////////////////////////////////////////////
// e.g.,					    //
//   row_stat_collector_t collector;		    //
//   visit_matrix_market_file(filename, collector); //
//////////////////////////////////////////////////////

struct row_stat_collector_t {
    using index_t = Index;
    using scalar_t = Scalar;

    explicit row_stat_collector_t()
    {
        max_row = 0;
        max_col = 0;
        max_elem = 0;
    }

    void eval_after_header(const index_t r, const index_t c, const index_t e)
    {
        std::tie(max_row, max_col, max_elem) = std::make_tuple(r, c, e);

#ifdef DEBUG
        TLOG("Start reading a list of triplets");
#endif

        Row_S1.resize(max_row);
        Row_S1.setZero();
        Row_S2.resize(max_row);
        Row_S2.setZero();
        Row_N.resize(max_row);
        Row_N.setZero();
    }

    void eval(const index_t row, const index_t col, const scalar_t weight)
    {
        if (row < max_row && col < max_col) {
            Row_S1(row) += weight;
            Row_S2(row) += (weight * weight);
            Row_N(row)++;
        }
    }

    void eval_end_of_file()
    {
#ifdef DEBUG
        TLOG("Finished reading a list of triplets");
#endif
    }

    Index max_row;
    Index max_col;
    Index max_elem;
    Vec Row_S1;
    Vec Row_S2;
    IntVec Row_N;
};

struct col_stat_collector_t {
    using index_t = Index;
    using scalar_t = Scalar;

    explicit col_stat_collector_t()
    {
        max_row = 0;
        max_col = 0;
        max_elem = 0;
    }

    void eval_after_header(const index_t r, const index_t c, const index_t e)
    {
        std::tie(max_row, max_col, max_elem) = std::make_tuple(r, c, e);

#ifdef DEBUG
        TLOG("Start reading a list of triplets");
#endif

        Col_S1.resize(max_col);
        Col_S1.setZero();
        Col_S2.resize(max_col);
        Col_S2.setZero();
        Col_N.resize(max_col);
        Col_N.setZero();
    }

    void eval(const index_t row, const index_t col, const scalar_t weight)
    {
        if (row < max_row && col < max_col) {
            Col_S1(col) += weight;
            Col_S2(col) += (weight * weight);
            Col_N(col)++;
        }
#ifdef DEBUG
        else {
            TLOG("[" << row << ", " << col << ", " << weight << "]");
            TLOG(max_row << " x " << max_col);
        }
#endif
    }

    void eval_end_of_file()
    {
#ifdef DEBUG
        TLOG("S1  : " << Col_S1.sum());
        TLOG("S2  : " << Col_S2.sum());
        TLOG("NNZ : " << Col_N.sum());
        TLOG("Finished reading a list of triplets");
#endif
    }

    Index max_row;
    Index max_col;
    Index max_elem;
    Vec Col_S1;
    Vec Col_S2;
    IntVec Col_N;
};

struct row_col_stat_collector_t {
    using index_t = Index;
    using scalar_t = Scalar;

    explicit row_col_stat_collector_t()
    {
        max_row = 0;
        max_col = 0;
        max_elem = 0;
    }

    void eval_after_header(const index_t r, const index_t c, const index_t e)
    {
        std::tie(max_row, max_col, max_elem) = std::make_tuple(r, c, e);

#ifdef DEBUG
        TLOG("Start reading a list of triplets");
#endif

        Row_S1.resize(max_row);
        Row_S1.setZero();
        Row_S2.resize(max_row);
        Row_S2.setZero();
        Row_N.resize(max_row);
        Row_N.setZero();

        Col_S1.resize(max_col);
        Col_S1.setZero();
        Col_S2.resize(max_col);
        Col_S2.setZero();
        Col_N.resize(max_col);
        Col_N.setZero();
    }

    void eval(const index_t row, const index_t col, const scalar_t weight)
    {
        if (row < max_row && col < max_col) {
            Row_S1(row) += weight;
            Row_S2(row) += (weight * weight);
            Row_N(row)++;

            Col_S1(col) += weight;
            Col_S2(col) += (weight * weight);
            Col_N(col)++;
        }
    }

    void eval_end_of_file()
    {
#ifdef DEBUG
        TLOG("Finished reading a list of triplets");
#endif
    }

    Index max_row;
    Index max_col;
    Index max_elem;

    Vec Row_S1;
    Vec Row_S2;
    IntVec Row_N;

    Vec Col_S1;
    Vec Col_S2;
    IntVec Col_N;
};

struct histogram_collector_t {
    using index_t = Index;
    using scalar_t = Scalar;

    explicit histogram_collector_t()
    {
        max_row = 0;
        max_col = 0;
        max_elem = 0;
    }

    void eval_after_header(const index_t r, const index_t c, const index_t e)
    {
        std::tie(max_row, max_col, max_elem) = std::make_tuple(r, c, e);

#ifdef DEBUG
        TLOG("Start reading a list of triplets");
#endif
    }

    void eval(const index_t row, const index_t col, const scalar_t weight)
    {
        if (row < max_row && col < max_col) {
            const index_t k = std::lround(weight);
            if (freq_map.count(k) == 0)
                freq_map[k] = 0;
            freq_map[k]++;
        }
    }

    void eval_end_of_file() {}

    Index max_row;
    Index max_col;
    Index max_elem;
    std::unordered_map<Index, Index> freq_map;
};

#endif
