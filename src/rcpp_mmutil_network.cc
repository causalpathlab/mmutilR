#include "rcpp_mmutil_network.hh"

//' Construct a kNN cell-cell interaction network and identify gene topics
//'
//' @param mtx_file data file (feature x n)
//' @param row_file row file (feature x 1)
//' @param col_file row file (n x 1)
//' @param latent_factor (n x K)
//' @param knn kNN parameter
//' @param output a file header for resulting files
//'
//' @param r_batches batch names (n x 1, default: NULL)
//'
//' @param CUTOFF expression present/absent call cutoff (default: 1e-2)
//' @param KNN_BILINK num. of bidirectional links (default: 10)
//' @param KNN_NNLIST num. of nearest neighbor lists (default: 10)
//' @param NUM_THREADS number of threads for multi-core processing
//'
//' @return feature.incidence, sample.incidence, edges, adjacency matrix files
//'
// [[Rcpp::export]]
Rcpp::List
rcpp_mmutil_network_topic_data(
    const std::string mtx_file,
    const std::string row_file,
    const std::string col_file,
    const Eigen::MatrixXf latent_factor,
    const std::size_t knn,
    const std::string output,
    Rcpp::Nullable<const Rcpp::StringVector> r_batches = R_NilValue,
    const float CUTOFF = 1e-2,
    const bool WEIGHTED = false,
    const float MAXW = 1,
    const std::size_t KNN_BILINK = 10,
    const std::size_t KNN_NNLIST = 10,
    const std::size_t NUM_THREADS = 1)
{

    mmutil::index::mm_info_reader_t info;
    CHK_RETL(mmutil::bgzf::peek_bgzf_header(mtx_file, info));

    TLOG("info: " << info.max_row << " x " << info.max_col
                  << " (NNZ=" << info.max_elem << ")");

    const Index D = info.max_row;
    const Index Nsample = info.max_col;

    ASSERT_RETL(latent_factor.rows() == Nsample,
                "latent factor matrix should have " << Nsample << " rows");

    std::vector<std::string> rows;
    if (file_exists(row_file)) {
        read_vector_file(row_file, rows);
        ASSERT_RETL(rows.size() == D,
                    "The sample size does not match with the row name file.");
    } else {
        for (Index j = 0; j < D; ++j)
            rows.push_back(std::to_string(j + 1));
    }

    std::vector<std::string> cols;
    if (file_exists(col_file)) {
        read_vector_file(col_file, cols);
        ASSERT_RETL(
            cols.size() == Nsample,
            "The sample size does not match with the column name file.");
    } else {
        for (Index j = 0; j < Nsample; ++j)
            cols.push_back(std::to_string(j + 1));
    }

    const std::string idx_file = mtx_file + ".index";

    if (!file_exists(idx_file)) // if needed
        CHK_RETL(mmutil::index::build_mmutil_index(mtx_file, idx_file));

    /////////////////////////////////////////
    // Step 1. build a weighted kNN matrix //
    /////////////////////////////////////////

    std::vector<std::string> batch_membership;
    if (r_batches.isNotNull()) {
        batch_membership = copy(Rcpp::StringVector(r_batches));
        ASSERT_RETL(
            batch_membership.size() == Nsample,
            "This batch membership vector mismatches with the mtx data");
    } else {
        batch_membership.resize(Nsample);
        std::fill(batch_membership.begin(), batch_membership.end(), "0");
    }

    std::vector<std::string> batch_id_name;
    std::vector<Index> batch; // map: col -> batch index

    std::tie(batch, batch_id_name, std::ignore) =
        make_indexed_vector<std::string, Index>(batch_membership);

    auto batch_index_set = make_index_vec_vec(batch);

    ASSERT_RETL(batch.size() >= Nsample,
                "Need batch membership for each column");
    const Index Nbatch = batch_id_name.size();
    TLOG("Identified " << Nbatch << " batches");

    std::vector<std::tuple<Index, Index, Scalar, Scalar>> knn_index;

    Mat VD_rank_sample = latent_factor.transpose();

    CHECK(build_bbknn(VD_rank_sample,
                      batch_index_set,
                      knn,
                      knn_index,
                      KNN_BILINK,
                      KNN_NNLIST,
                      NUM_THREADS));

    SpMat W = build_eigen_sparse(knn_index, Nsample, Nsample);
    TLOG("A weighted adjacency matrix W");

    /////////////////////////////////////
    // symmetrize the adjacency matrix //
    /////////////////////////////////////

    SpMat Wt = W.transpose();
    SpMat Wsym = W * .5 + Wt * .5;

    TLOG("A weighted adjacency matrix W");

    /////////////////////////////////////////////
    // Step 2. build a sparse incidence matrix //
    /////////////////////////////////////////////

    const std::string out_feat_inc = output + "_feat_inc.mtx.gz";
    const std::string out_pair_names = output + ".pairs.gz";
    const std::string out_samp_inc = output + "_samp_inc.mtx.gz";
    const std::string out_samp_adj = output + "_samp_adj.mtx.gz";

    auto rm_mtx = [](const std::string mtx) {
        if (file_exists(mtx)) {
            WLOG("Removing the existing mtx file: " << mtx);
            remove_file(mtx);
        }
        if (file_exists(mtx + ".index")) {
            WLOG("Removing the existing index file: " << mtx + ".index");
            remove_file(mtx + ".index");
        }
    };

    auto rm = [](const std::string ff) {
        if (file_exists(ff))
            remove_file(ff);
    };

    rm_mtx(out_feat_inc);
    rm_mtx(out_samp_inc);
    rm_mtx(out_samp_adj);

    //////////////////////////////////////
    // feature (row/gene) x edge matrix //
    //////////////////////////////////////

    {
        using writer_t = feature_incidence_writer_t<obgzf_stream>;
        const std::string temp_file = out_feat_inc + "_temp";
        rm(temp_file);
        writer_t writer(mtx_file, temp_file, CUTOFF, WEIGHTED, MAXW);
        visit_sparse_matrix(W, writer);

        obgzf_stream ofs(out_feat_inc.c_str(), std::ios::out);

        ofs << "%%MatrixMarket matrix coordinate integer general" << std::endl;
        ofs << writer.max_row() << " " << writer.max_col() << " "
            << writer.max_elem() << std::endl;

        std::string line;

        ibgzf_stream ifs(temp_file.c_str(), std::ios::in);
        while (std::getline(ifs, line)) {
            ofs << line << std::endl;
        }

        ofs.close();
        ifs.close();
        remove_file(temp_file);
        TLOG("Wrote feature incidence matrix: " << out_feat_inc);
    }

    {
        using writer_t = sample_pair_writer_t<obgzf_stream>;
        writer_t writer(out_pair_names, cols);
        visit_sparse_matrix(W, writer);
        TLOG("Wrote pair names: " << out_pair_names);
    }

    ///////////////////////////////////
    // vertex (sample) x edge matrix //
    ///////////////////////////////////

    {
        using writer_t = sample_incidence_writer_t<obgzf_stream>;
        writer_t writer(out_samp_inc);
        visit_sparse_matrix(Wsym, writer);
        TLOG("Wrote sample incidence matrix: " << out_samp_inc);
    }

    {
        using writer_t = sample_adjacency_writer_t<obgzf_stream>;
        writer_t writer(out_samp_adj);
        visit_sparse_matrix(Wsym, writer);
        TLOG("Wrote sample adjacency matrix: " << out_samp_adj);
    }

    write_vector_file(output + ".features.gz", rows);
    write_vector_file(output + ".samples.gz", cols);

    TLOG("Done");
    return Rcpp::List::create(Rcpp::_["feature.incidence"] = out_feat_inc,
                              Rcpp::_["pairs"] = output + ".pairs.gz",
                              Rcpp::_["sample.incidence"] = out_samp_inc,
                              Rcpp::_["sample.adjacency"] = out_samp_adj,
                              Rcpp::_["features"] = output + ".features.gz",
                              Rcpp::_["samples"] = output + ".samples.gz");
}
