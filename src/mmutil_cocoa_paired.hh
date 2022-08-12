#include "mmutil.hh"
#include "mmutil_io.hh"
#include "mmutil_stat.hh"
#include "mmutil_index.hh"
#include "mmutil_bgzf_util.hh"
#include "mmutil_spectral.hh"
#include "mmutil_match.hh"
#include "mmutil_pois.hh"
#include "mmutil_glm.hh"

#include "io.hh"
#include "std_util.hh"

// [[Rcpp::plugins(openmp)]]

#ifndef MMUTIL_COCOA_PAIRED_HH_
#define MMUTIL_COCOA_PAIRED_HH_

struct paired_data_t {

    using vs_type = hnswlib::InnerProductSpace;

    using str_vec_t = std::vector<std::string>;
    using idx_vec_t = std::vector<Index>;
    using num_vec_t = std::vector<Scalar>;

    paired_data_t(const std::string _mtx_file,
                  const idx_vec_t &_mtx_idx_tab,
                  const str_vec_t &_mtx_cols,
                  const str_vec_t &_cols,
                  const KNN _knn,
                  const KNN _knn_indv,
                  const BILINK _bilink,
                  const NNLIST _nnlist)
        : mtx_file(_mtx_file)
        , mtx_idx_tab(_mtx_idx_tab)
        , mtx_cols(_mtx_cols)
        , cols(_cols)
        , Nsample(mtx_cols.size())
        , knn(_knn.val)
        , knn_indv(_knn_indv.val)
        , param_bilink(_bilink.val)
        , param_nnlist(_nnlist.val)
    {
        mmutil::io::mm_info_reader_t info;
        CHECK(mmutil::bgzf::peek_bgzf_header(mtx_file, info));
        D = info.max_row;
        ASSERT(Nsample == info.max_col, "|mtx_cols| != mtx's max_col");
        Nindv = 0;
    }

    Mat read_block(const Index i);
    Mat _read_block(const idx_vec_t &cells_j);
    Mat read_matched_block(const Index i, const Index j);

    void set_individual_info(const str_vec_t &indv);

    std::vector<std::tuple<Index, Index>>
    match_individuals(const Rcpp::NumericMatrix r_V);

    int build_dictionary(const Rcpp::NumericMatrix r_V,
                         const std::size_t NUM_THREADS);

    Index num_individuals() const;

    const std::string mtx_file;
    const idx_vec_t &mtx_idx_tab;
    const str_vec_t &mtx_cols;
    const str_vec_t &cols;
    const Index Nsample;

    const std::string indv_name(const Index i) const;
    const idx_vec_t &cell_indexes(const Index i) const;

private:
    Index D;
    Index Nindv;
    str_vec_t indv_id_name; // individual names
    idx_vec_t indv_map;     // map: col -> indv index
    std::vector<idx_vec_t> indv_index_set;
    Mat Vt; // rank x column matching data

    std::vector<std::shared_ptr<vs_type>> vs_vec_indv;
    std::vector<std::shared_ptr<KnnAlg>> knn_lookup_indv;

    const std::size_t knn;
    const std::size_t knn_indv;
    std::size_t param_bilink;
    std::size_t param_nnlist;
};

#endif
