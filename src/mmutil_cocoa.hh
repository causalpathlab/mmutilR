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

#ifndef MMUTIL_COCOA_HH_
#define MMUTIL_COCOA_HH_

struct matched_data_t {

    using vs_type = hnswlib::InnerProductSpace;

    using str_vec_t = std::vector<std::string>;
    using idx_vec_t = std::vector<Index>;
    using num_vec_t = std::vector<Scalar>;

    matched_data_t(const std::string _mtx_file,
                   const idx_vec_t &_mtx_idx_tab,
                   const str_vec_t &_mtx_cols,
                   const str_vec_t &_cols,
                   const KNN _knn,
                   const BILINK _bilink,
                   const NNLIST _nnlist,
                   Scalar glm_pseudo = 1.)
        : mtx_file(_mtx_file)
        , mtx_idx_tab(_mtx_idx_tab)
        , mtx_cols(_mtx_cols)
        , cols(_cols)
        , Nsample(mtx_cols.size())
        , knn(_knn.val)
        , param_bilink(_bilink.val)
        , param_nnlist(_nnlist.val)
        , glm_feature(glm_pseudo)
    {
        mmutil::io::mm_info_reader_t info;
        CHECK(mmutil::bgzf::peek_bgzf_header(mtx_file, info));
        D = info.max_row;
        ASSERT(Nsample == info.max_col, "|mtx_cols| != mtx's max_col");
        Ntrt = 0;

        glm_iter = 100;
        glm_reg = 1e-4;
        glm_sd = 1e-2;
        glm_std = true;
    }

    void set_treatment_info(const str_vec_t &trt);

    int build_dictionary(const Rcpp::NumericMatrix r_V,
                         const std::size_t NUM_THREADS);

    Mat read_cf_block(const idx_vec_t &cells_j, bool impute_knn);

    Mat read_block(const idx_vec_t &cells_j);

    Index num_treatment() const;

    const std::string mtx_file;
    const idx_vec_t &mtx_idx_tab;
    const str_vec_t &mtx_cols;
    const str_vec_t &cols;
    const Index Nsample;

private:
    Index D;
    Index Ntrt;
    str_vec_t trt_id_name; // treatment names
    idx_vec_t trt_map;     // map: col -> trt index
    std::vector<idx_vec_t> trt_index_set;
    Mat Vt; // rank x column matching data

    std::vector<std::shared_ptr<vs_type>> vs_vec_trt;
    std::vector<std::shared_ptr<KnnAlg>> knn_lookup_trt;

    const std::size_t knn;
    std::size_t param_bilink;
    std::size_t param_nnlist;

    Scalar glm_pseudo;
    Index glm_iter;
    Index glm_reg;
    Scalar glm_sd;
    bool glm_std;

    struct glm_feature_op_t {
        explicit glm_feature_op_t(const Scalar pseudo)
            : glm_pseudo(pseudo)
        {
        }
        Scalar operator()(const Scalar &x) const
        {
            return fasterlog(x + glm_pseudo);
        }
        const Scalar glm_pseudo;
    };

    const glm_feature_op_t glm_feature;
};

#endif
