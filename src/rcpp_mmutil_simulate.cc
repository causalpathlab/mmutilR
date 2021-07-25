#include "mmutil.hh"
#include "mmutil_simulate.hh"
#include "mmutil_filter.hh"

//' Simulation Poisson data
//'
//' @param Mu depth-adjusted mean matrix (M x n), M=#features and n=#indv
//' @param Rho column depth vector (N x 1), N=#cells
//' @param output header for ${output}.{mtx.gz,cols.gz,indv.gz}
//'
//' @return a list of file names: {output}.{mtx,rows,cols}.gz
//'
//' @examples
//' rr <- rgamma(20, 1, 1)
//' mm <- matrix(rgamma(10 * 2, 1, 1), 10, 2)
//' data.hdr <- "test_sim"
//' .files <- mmutilR::rcpp_mmutil_simulate_poisson(mm, rr, data.hdr)
//' Y <- Matrix::readMM(.files$mtx)
//' print(Y)
//' A <- read.table(.files$indv, col.names = c("col", "ind"))
//' head(A)
//' unlink(list.files(pattern = data.hdr))
//'
// [[Rcpp::export]]
Rcpp::List
rcpp_mmutil_simulate_poisson(const Eigen::MatrixXf mu,
                             const Eigen::VectorXf rho,
                             const std::string output)
{

    const Index max_row = mu.rows();
    const Index Nind = mu.cols();
    const Index Nsample = rho.rows();

    TLOG(Nind << " individuals, " << Nsample << " cells");

    // Partition columns into individuals
    std::minstd_rand rng{ std::random_device{}() };
    std::uniform_int_distribution<Index> unif_ind(0, Nind - 1);
    std::uniform_int_distribution<Index> unif_rho(0, Nsample - 1);

    auto _sample_ind = [&rng, &unif_ind](const Index) -> Index {
        return unif_ind(rng);
    };

    std::vector<Index> indv;
    indv.resize(Nsample);
    std::transform(std::begin(indv),
                   std::end(indv),
                   std::begin(indv),
                   _sample_ind);

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

    for (Index i = 0; i < partition.size(); ++i) {
        const Index n_i = partition[i].size();

        if (n_i < 1)
            continue;

        std::string _temp_file = output + "_temp_" + std::to_string(i) + ".gz";
        obgzf_stream ofs(_temp_file.c_str(), std::ios::out);

        // Sample rho values for this individual i
        Vec rho_i(n_i);
        for (Index j = 0; j < n_i; ++j) {
            rho_i(j) = rho(unif_rho(rng));
        }

        const Index nelem =
            sample_poisson_data(mu.col(i), rho_i, ncol, ofs, FS);

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

    ASSERT(file_exists(mtx_file), "missing file: " << mtx_file);
    ASSERT(file_exists(col_file), "missing file: " << col_file);
    ASSERT(file_exists(row_file), "missing file: " << row_file);

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
