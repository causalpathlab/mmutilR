#include "progress.hh"

#include "mmutil.hh"
#include "mmutil_spectral.hh"
#include "mmutil_match.hh"
#include "mmutil_filter.hh"
#include "mmutil_util.hh"
#include "mmutil_index.hh"
#include "mmutil_io.hh"
#include "mmutil_bbknn.hh"
#include "mmutil_cluster.hh"
#include "mmutil_cluster_poisson.hh"

#ifndef RCPP_MMUTIL_NETWORK_HH
#define RCPP_MMUTIL_NETWORK_HH

template <typename OFS>
struct sample_adjacency_writer_t {

    sample_adjacency_writer_t(const std::string _out_file)
        : out_file(_out_file)
    {
    }

    inline void eval_after_header(const Index r, const Index c, const Scalar e)
    {
        ofs.open(out_file.c_str(), std::ios::out);
        ofs << "%%MatrixMarket matrix coordinate integer general" << std::endl;
        ofs << r << FS << c << FS << e << std::endl;
    }

    inline void eval_end_of_data() { ofs.close(); }

    inline void eval(const Index j, const Index i, const Scalar Aij)
    {
        // Note: rowMajor -> colMajor
        ofs << (i + 1) << FS << (j + 1) << FS << Aij << std::endl;
    }

    const std::string out_file;
    static constexpr char FS = ' ';

private:
    OFS ofs;
};

template <typename OFS>
struct sample_incidence_writer_t {

    sample_incidence_writer_t(const std::string _out_file,
                              const std::vector<std::string> &_cols,
                              std::vector<std::string> &_pairs)
        : out_file(_out_file)
        , cols(_cols)
        , pairs(_pairs)
    {
    }

    inline void eval_after_header(const Index r, const Index c, const Scalar e)
    {
        ofs.open(out_file.c_str(), std::ios::out);

        ///////////////////////////////////
        // vertex (sample) x edge matrix //
        ///////////////////////////////////

        ofs << "%%MatrixMarket matrix coordinate integer general" << std::endl;
        ofs << std::max(r, c) << FS << e << FS << (e * 2) << std::endl;

        pairs.clear();
        npairs = 0;
    }

    inline void eval_end_of_data() { ofs.close(); }

    inline void eval(const Index j, const Index i, const Scalar wji)
    {
        ofs << (j + 1) << FS << (npairs + 1) << FS << 1 << std::endl;
        ofs << (i + 1) << FS << (npairs + 1) << FS << 1 << std::endl;
        ++npairs;
        pairs.emplace_back(cols.at(j) + "_" + cols.at(i));
    }

    const std::string out_file;
    const std::vector<std::string> &cols;
    std::vector<std::string> &pairs;
    static constexpr char FS = ' ';

private:
    Index npairs;
    OFS ofs;
};

template <typename OFS>
struct sample_pair_writer_t {

    sample_pair_writer_t(const std::string _out_file,
                         const std::vector<std::string> &_samp)
        : out_file(_out_file)
        , sample_names(_samp)
    {
    }

    inline void eval_after_header(const Index r, const Index c, const Scalar e)
    {
        ofs.open(out_file.c_str(), std::ios::out);
        ASSERT(sample_names.size() >= std::max(r, c),
               "sample names should cover all the rows and columns");
    }

    inline void eval_end_of_data() { ofs.close(); }

    inline void eval(const Index j, const Index i, const Scalar wji)
    {
        // Note: rowMajor -> colMajor
        ofs << sample_names.at(i) << FS << sample_names.at(j) << std::endl;
    }

    const std::string out_file;
    const std::vector<std::string> &sample_names;
    static constexpr char FS = '-';

private:
    OFS ofs;
};

template <typename OFS>
struct feature_incidence_writer_t {

    feature_incidence_writer_t(const std::string _mtx_file,
                               const std::vector<std::string> &_cols,
                               const std::string _out_file,
                               std::vector<std::string> &_pairs,
                               const Scalar cutoff,
                               const bool keep_weights = true,
                               const Scalar maxW = 100)
        : mtx_file(_mtx_file)
        , cols(_cols)
        , out_file(_out_file)
        , pairs(_pairs)
        , CUTOFF(cutoff)
        , WEIGHTED(keep_weights)
        , MAXW(maxW)
    {
        const std::string idx_file = mtx_file + ".index";
        CHECK(mmutil::index::read_mmutil_index(idx_file, idx_tab));

        mmutil::index::mm_info_reader_t info;
        CHECK(mmutil::bgzf::peek_bgzf_header(mtx_file, info));
        D = info.max_row;
        Nsample = info.max_col;
        _xj.resize(D);
        std::fill(_xj.begin(), _xj.end(), 0.);
    }

    inline void eval_after_header(const Index r, const Index c, const Scalar e)
    {
        ofs.open(out_file.c_str(), std::ios::out);
        pairs.clear();
        npairs = 0;
        mtot = 0;
    }

    inline void eval_end_of_data() { ofs.close(); }

    inline SpMat read_x(const Index j)
    {
        const Index lb = j, ub = j + 1;
        const Index lb_mem = lb < Nsample ? idx_tab[lb] : 0;
        const Index ub_mem = ub < Nsample ? idx_tab[ub] : 0;
        return mmutil::io::read_eigen_sparse_subset_col(mtx_file,
                                                        lb,
                                                        ub,
                                                        lb_mem,
                                                        ub_mem);
    }

    inline void eval(const Index j, const Index i, const Scalar wji)
    {

        using namespace mmutil::io;
        bool _add_pair = false;

        SpMat xi = read_x(i).transpose();
        SpMat xj = read_x(j).transpose();

        // SpMat xj =
        //     read_eigen_sparse_subset_col(mtx_file, idx_tab, { j
        //     }).transpose();

        // SpMat xi =
        //     read_eigen_sparse_subset_col(mtx_file, idx_tab, { i
        //     }).transpose();

        for (SpMat::InnerIterator xt(xj, 0); xt; ++xt) {
            const Index g = xt.col();
            const Scalar xjg = xt.value();
            if (xjg > CUTOFF)
                _xj[g] = xjg;
        }

        for (SpMat::InnerIterator xt(xi, 0); xt; ++xt) {
            const Index g = xt.col();
            const Scalar xig = xt.value();
            const Scalar xjg = _xj[g];
            if (xig > CUTOFF && xjg > CUTOFF) {
                const Scalar xx = (xjg + xig) * wji / 2.0;
                if (xx > CUTOFF) {
                    // fix zero-based to one-based
                    ofs << (g + 1) << FS << (npairs + 1);
                    if (WEIGHTED) {
                        ofs << FS << std::min(xx, MAXW) << std::endl;
                    } else {
                        ofs << FS << 1 << std::endl;
                    }
                    ++mtot;
                    _add_pair = true;
                }
            }
        }

        for (SpMat::InnerIterator xt(xj, 0); xt; ++xt) {
            const Index g = xt.col();
            _xj[g] = 0;
        }

        if (_add_pair) {
            ++npairs;
            pairs.emplace_back(cols.at(j) + "_" + cols.at(i));
        }
    }

    Index max_row() const { return D; }
    Index max_col() const { return npairs; }
    Index max_elem() const { return mtot; }

    const std::string mtx_file;
    const std::vector<std::string> &cols;
    const std::string out_file;
    std::vector<std::string> &pairs;
    const Scalar CUTOFF;
    const bool WEIGHTED;
    const Scalar MAXW;

    static constexpr char FS = ' ';

private:
    std::vector<Index> idx_tab;
    Index D;
    Index Nsample;
    std::vector<Scalar> _xj;

    OFS ofs;
    Index npairs;
    Index mtot;
};

#endif
