#include "mmutil.hh"
#include "mmutil_simulate.hh"
#include "mmutil_filter.hh"

#include <random>

//' Simulate sparse counting data with a mixture of Poisson parameters
//'
//'
//' @param r_mu_list a list of gene x individual matrices
//' @param Ncell the total number of cells (may not make it if too sparse)
//' @param output a file header string for output files
//' @param dir_alpha a parameter for Dirichlet(alpha * [1, ..., 1])
//' @param gam_alpha a parameter for Gamma(alpha, beta)
//' @param gam_beta a parameter for Gamma(alpha, beta)
//' @param rseed random seed
//'
// [[Rcpp::export]]
Rcpp::List
rcpp_mmutil_simulate_poisson_mixture(const Rcpp::List r_mu_list,
                                     const std::size_t Ncell,
                                     const std::string output,
                                     const float dir_alpha = 1.0,
                                     const float gam_alpha = 2.0,
                                     const float gam_beta = 2.0,
                                     const std::size_t rseed = 42)
{

    ASSERT_RETL(r_mu_list.size() > 0, "must have a list of mu matrices");

    const Index K = r_mu_list.size();
    std::vector<Eigen::MatrixXf> mu_list(K);
    Index max_row = 0, Nind = 0;

    TLOG("Importing gene x individual matrices");

    for (Index k = 0; k < K; ++k) {
        mu_list[k] = Rcpp::as<Eigen::MatrixXf>(r_mu_list.at(k));
        if (k == 0) {
            max_row = mu_list.at(k).rows();
            Nind = mu_list.at(k).cols();
        } else {
            ASSERT_RETL(max_row == mu_list.at(k).rows(),
                        "mu's must have the same number of rows");
            ASSERT_RETL(Nind == mu_list.at(k).cols(),
                        "mu's must have the same number of columns");
        }
    }

    TLOG("Read " << K << " mu matrices [" << Nind << " x " << Ncell << "]");

    /////////////////////////////
    // random number generator //
    /////////////////////////////

    dqrng::xoshiro256plus rng(rseed);

    inf_zero_op<Vec> inf_zero; // remove inf -> 0

    /////////////////////////////////
    // Assign cells to individuals //
    /////////////////////////////////

    boost::random::uniform_int_distribution<Index> _rind(0, Nind - 1);
    auto rind = [&rng, &_rind](Index &x) { x = _rind(rng); };

    std::vector<Index> indv(Ncell);
    TLOG("Sampling column to individual membership...");
    std::for_each(std::begin(indv), std::end(indv), rind);

    const auto partition = make_index_vec_vec(indv);
    TLOG("Distributed " << Ncell << " into " << partition.size()
                        << " groups/individuals");

    /////////////////////////////////////////////////////////////
    // step 1: simulate each individual with a mixture of mu's //
    /////////////////////////////////////////////////////////////
    using gamma_distrib = boost::random::gamma_distribution<Scalar>;

    gamma_distrib _rgamma(gam_alpha, gam_beta);
    auto rgamma = [&rng, &_rgamma](const Scalar &x) -> Scalar {
        return _rgamma(rng);
    };

    Index nnz = 0;
    Index ncol = 0;

    std::vector<std::string> temp_files;
    temp_files.reserve(partition.size() * K);
    const std::string FS = " ";

    std::vector<std::tuple<Index, Index>> indv_vec;
    indv_vec.reserve(Ncell);

    std::vector<Index> col_names;
    col_names.reserve(Ncell);

    for (Index i = 0; i < partition.size(); ++i) {
        const Index n_i = partition[i].size();

        if (n_i < 1)
            continue;

        ///////////////////////////////
        // Sample mixing proportions //
        ///////////////////////////////

        gamma_distrib _rdir(dir_alpha, static_cast<Scalar>(1));
        auto rdir = [&rng, &_rdir](Scalar &x) { x = _rdir(rng); };
        std::vector<Scalar> mixing_prop(K);
        std::for_each(std::begin(mixing_prop), std::end(mixing_prop), rdir);
        const Scalar tot =
            std::accumulate(std::begin(mixing_prop), std::end(mixing_prop), 0.);
        std::for_each(std::begin(mixing_prop),
                      std::end(mixing_prop),
                      [&tot, &n_i](Scalar &x) { x = x / tot * n_i; });

        ////////////////////////////
        // Sample cell membership //
        ////////////////////////////

        std::vector<Index> cells(n_i);
        boost::random::discrete_distribution<Index> _rdisc(mixing_prop);
        auto rdisc = [&rng, &_rdisc](Index &x) { x = _rdisc(rng); };
        std::for_each(std::begin(cells), std::end(cells), rdisc);

        const auto cell_partition = make_index_vec_vec(cells);

        for (Index k = 0; k < cell_partition.size(); ++k) {
            const Index n_ik = cell_partition[k].size();

            TLOG("Simulating " << n_ik << " cells...");

            if (n_ik < 1)
                continue;

            Vec rho_ik(n_ik);
            rho_ik = rho_ik.unaryExpr(rgamma).unaryExpr(inf_zero);

            std::string _temp_file = output + "_temp_" + std::to_string(i) +
                "_k_" + std::to_string(k) + ".gz";
            obgzf_stream ofs(_temp_file.c_str(), std::ios::out);

            const Index m_ik = sample_poisson_data(mu_list.at(k).col(i),
                                                   rho_ik,
                                                   ncol,
                                                   ofs,
                                                   rng,
                                                   FS);

            ofs.close();
            temp_files.emplace_back(_temp_file);

            if (m_ik > 0) {
                const Index ii = i + 1;
                for (Index j = 0; j < n_ik; ++j) {
                    const Index cc = (ncol + j + 1);
                    col_names.emplace_back(cc);
                    indv_vec.emplace_back(std::make_tuple<>(cc, ii));
                }
                nnz += m_ik;
                ncol += n_ik;
            }
        }
    }

    //////////////////////////
    // step 2: combine them //
    //////////////////////////

    const std::string temp_mtx_file = output + ".mtx-temp.gz";
    obgzf_stream ofs(temp_mtx_file.c_str(), std::ios::out);

    ofs << "%%MatrixMarket matrix coordinate integer general" << std::endl;
    ofs << max_row << FS << ncol << FS << nnz << std::endl;

    for (std::string f : temp_files) {
        ibgzf_stream ifs(f.c_str(), std::ios::in);
        std::string line;
        while (std::getline(ifs, line)) {
            ofs << line << std::endl;
        }
        ifs.close();

        if (file_exists(f))
            remove_file(f);
    }

    ofs.close();

    ///////////////////////////////
    // step3 : squeeze zeros out //
    ///////////////////////////////

    const std::string temp_col_file = output + ".cols-temp.gz";
    write_vector_file(temp_col_file, col_names);
    filter_col_by_nnz(1, temp_mtx_file, temp_col_file, output);
    remove_file(temp_col_file);
    remove_file(temp_mtx_file);

    const std::string mtx_file = output + ".mtx.gz";
    const std::string col_file = output + ".cols.gz";
    const std::string row_file = output + ".rows.gz";

    std::vector<Index> rows(max_row);
    std::iota(rows.begin(), rows.end(), 1);
    std::vector<std::string> row_names;
    std::transform(rows.begin(),
                   rows.end(),
                   std::back_inserter(row_names),
                   [&max_row](const Index r) -> std::string {
                       return "r" + zeropad(r, max_row);
                   });

    write_vector_file(row_file, row_names);

    ASSERT_RETL(file_exists(mtx_file), "missing file: " << mtx_file);
    ASSERT_RETL(file_exists(col_file), "missing file: " << col_file);
    ASSERT_RETL(file_exists(row_file), "missing file: " << row_file);

    if (file_exists(mtx_file + ".index"))
        remove_file(mtx_file + ".index");

    const std::string idx_file = mtx_file + ".index";

    CHECK(mmutil::index::build_mmutil_index(mtx_file, idx_file));

    const std::string indv_file = output + ".indv.gz";
    write_tuple_file(indv_file, indv_vec);

    return Rcpp::List::create(Rcpp::_["mtx"] = mtx_file,
                              Rcpp::_["row"] = row_file,
                              Rcpp::_["col"] = col_file,
                              Rcpp::_["idx"] = idx_file,
                              Rcpp::_["indv"] = indv_file);
}

//' Simulation Poisson data based on Mu
//'
//' M= num. of features and n= num. of indv
//'
//' @param mu depth-adjusted mean matrix (M x n)
//' @param rho column depth vector (N x 1), N= num. of cells
//' @param output header for ${output}.{mtx.gz,cols.gz,indv.gz}
//' @param r_indv N x 1 individual membership (1-based, [1 .. n])
//' @param rseed random seed
//'
//' @return a list of file names: {output}.{mtx,rows,cols}.gz
//'
// [[Rcpp::export]]
Rcpp::List
rcpp_mmutil_simulate_poisson(
    const Eigen::MatrixXf mu,
    const Eigen::VectorXf rho,
    const std::string output,
    Rcpp::Nullable<Rcpp::IntegerVector> r_indv = R_NilValue,
    const std::size_t rseed = 42)
{
    const Index max_row = mu.rows();
    const Index Nind = mu.cols();
    const Index Nsample = rho.rows();

    TLOG(Nind << " individuals, " << Nsample << " cells");

    // Partition columns into individuals
    dqrng::xoshiro256plus rng(rseed);
    // std::minstd_rand rng { std::random_device {}() };

    // uniformly select individual of origin
    boost::random::uniform_int_distribution<Index> unif_ind(0, Nind - 1);

    // uniformly select cell depth parameters
    boost::random::uniform_int_distribution<Index> unif_rho(0, Nsample - 1);

    auto _sample_ind = [&rng, &unif_ind](const Index) -> Index {
        return unif_ind(rng);
    };

    std::vector<Index> indv;
    if (r_indv.isNotNull()) {
        TLOG("Column to individual memberships were provided");
        Rcpp::IntegerVector _indv(r_indv);
        ASSERT_RETL(_indv.size() == Nsample,
                    "Must have the same number of samples");
        indv.reserve(Nsample);
        for (Index j = 0; j < _indv.size(); ++j) {
            const Index i = _indv[j];
            if (i >= 1 && i <= Nind)
                indv.emplace_back(i - 1);
        }
        ASSERT_RETL(indv.size() == Nsample,
                    "Incomplete membership information");
    } else {
        TLOG("Sampling column to individual memberships");
        indv.resize(Nsample);
        std::transform(std::begin(indv),
                       std::end(indv),
                       std::begin(indv),
                       _sample_ind);
    }

    const auto partition = make_index_vec_vec(indv);

    TLOG("Distributing " << Nsample << " into " << partition.size()
                         << " group");

    //////////////////////////////////
    // step 1: simulate local stuff //
    //////////////////////////////////

    Index nnz = 0;
    Index ncol = 0;

    std::vector<std::string> temp_files;
    temp_files.reserve(partition.size());
    const std::string FS = " ";

    std::vector<std::tuple<Index, Index>> indv_vec;
    indv_vec.reserve(Nsample);

    std::vector<Index> col_names;
    col_names.reserve(Nsample);

    inf_zero_op<Vec> inf_zero; // remove inf -> 0

    for (Index i = 0; i < partition.size(); ++i) {
        const Index n_i = partition[i].size();

        if (n_i < 1)
            continue;

        std::string _temp_file = output + "_temp_" + std::to_string(i) + ".gz";
        obgzf_stream ofs(_temp_file.c_str(), std::ios::out);

        // Copy from the sampled rho values for this individual i
        Vec rho_i(n_i);
        for (Index j = 0; j < n_i; ++j) {
            const Index k = unif_rho(rng);
            rho_i(j) = rho(k);
        }

        rho_i = rho_i.unaryExpr(inf_zero);

        const Index nelem =
            sample_poisson_data(mu.col(i), rho_i, ncol, ofs, rng, FS);

        ofs.close();

        temp_files.emplace_back(_temp_file);

        if (nelem > 0) {

            const Index ii = i + 1;

            for (Index j = 0; j < n_i; ++j) {
                const Index cc = (ncol + j + 1);
                col_names.emplace_back(cc);
                indv_vec.emplace_back(std::make_tuple<>(cc, ii));
            }

            nnz += nelem;
            ncol += n_i;
        }
    }

    //////////////////////////
    // step 2: combine them //
    //////////////////////////

    const std::string temp_mtx_file = output + ".mtx-temp.gz";
    obgzf_stream ofs(temp_mtx_file.c_str(), std::ios::out);

    ofs << "%%MatrixMarket matrix coordinate integer general" << std::endl;
    ofs << max_row << FS << ncol << FS << nnz << std::endl;

    for (std::string f : temp_files) {
        ibgzf_stream ifs(f.c_str(), std::ios::in);
        std::string line;
        while (std::getline(ifs, line)) {
            ofs << line << std::endl;
        }
        ifs.close();

        if (file_exists(f))
            remove_file(f);
    }

    ofs.close();

    ///////////////////////////////
    // step3 : squeeze zeros out //
    ///////////////////////////////

    const std::string temp_col_file = output + ".cols-temp.gz";
    write_vector_file(temp_col_file, col_names);
    filter_col_by_nnz(1, temp_mtx_file, temp_col_file, output);
    remove_file(temp_col_file);
    remove_file(temp_mtx_file);

    const std::string mtx_file = output + ".mtx.gz";
    const std::string col_file = output + ".cols.gz";
    const std::string row_file = output + ".rows.gz";

    std::vector<Index> rows(max_row);
    std::iota(rows.begin(), rows.end(), 1);
    std::vector<std::string> row_names;
    std::transform(rows.begin(),
                   rows.end(),
                   std::back_inserter(row_names),
                   [&max_row](const Index r) -> std::string {
                       return "r" + zeropad(r, max_row);
                   });

    write_vector_file(row_file, row_names);

    ASSERT_RETL(file_exists(mtx_file), "missing file: " << mtx_file);
    ASSERT_RETL(file_exists(col_file), "missing file: " << col_file);
    ASSERT_RETL(file_exists(row_file), "missing file: " << row_file);

    if (file_exists(mtx_file + ".index"))
        remove_file(mtx_file + ".index");

    const std::string idx_file = mtx_file + ".index";

    CHECK(mmutil::index::build_mmutil_index(mtx_file, idx_file));

    const std::string indv_file = output + ".indv.gz";
    write_tuple_file(indv_file, indv_vec);

    return Rcpp::List::create(Rcpp::_["mtx"] = mtx_file,
                              Rcpp::_["row"] = row_file,
                              Rcpp::_["col"] = col_file,
                              Rcpp::_["idx"] = idx_file,
                              Rcpp::_["indv"] = indv_file);
}
