#include <getopt.h>
#include <cstdio>

#include <random>
#include "mmutil.hh"
#include "mmutil_index.hh"
#include "mmutil_merge_col.hh"

#include "stat.hh"
#include "std_util.hh"
#include "bgzstream.hh"

#ifndef MMUTIL_SIMULATE_HH_
#define MMUTIL_SIMULATE_HH_

struct simulate_options_t {

    simulate_options_t()
    {
        mu_file = "";
        rho_file = "";
        col_file = "";
        out = "output";
    }

    std::string mu_file;
    std::string rho_file;
    std::string col_file;
    std::string out;
};

template <typename OPTIONS>
int
parse_simulate_options(const int argc,     //
                       const char *argv[], //
                       OPTIONS &options)
{
    const char *_usage = "\n"
                         "[Arguments]\n"
                         "--mu (-u)     : Depth-adjusted mean matrix (M x n)\n"
                         "--rho (-r)    : Column depth file (N x 1)\n"
                         "--col (-c)    : Column names\n"
                         "--out (-o)    : Output file header\n"
                         "\n"
                         "${out}.mtx.gz : (M x N) sparse matrix file\n"
                         "\n";

    const char *const short_opts = "u:c:r:o:";

    const option long_opts[] = { { "mu", required_argument, nullptr, 'u' },  //
                                 { "rho", required_argument, nullptr, 'r' }, //
                                 { "col", required_argument, nullptr, 'c' }, //
                                 { "out", required_argument, nullptr, 'o' }, //
                                 { nullptr, no_argument, nullptr, 0 } };

    while (true) {
        const auto opt = getopt_long(argc,                      //
                                     const_cast<char **>(argv), //
                                     short_opts,                //
                                     long_opts,                 //
                                     nullptr);

        if (-1 == opt)
            break;

        switch (opt) {
        case 'u':
            options.mu_file = std::string(optarg);
            break;
        case 'r':
            options.rho_file = std::string(optarg);
            break;
        case 'c':
            options.col_file = std::string(optarg);
            break;
        case 'o':
            options.out = std::string(optarg);
            break;
        case 'h': // -h or --help
        case '?': // Unrecognized option
            std::cerr << _usage << std::endl;
            return EXIT_FAILURE;
        default: //
                 ;
        }
    }

    ERR_RET(!file_exists(options.mu_file), "No MU file");
    ERR_RET(!file_exists(options.rho_file), "No RHO file");
    ERR_RET(!file_exists(options.col_file), "No COL file");

    return EXIT_SUCCESS;
}

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
            nnz++; // not exactly true...
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

template <typename OPTIONS>
int
simulate_mtx_matrix(const OPTIONS &options)
{

    // using namespace mmutil::io;
    // using namespace mmutil::index;

    const std::string mu_file = options.mu_file;
    const std::string rho_file = options.rho_file;
    const std::string col_file = options.col_file;
    const std::string output = options.out;

    //////////////////
    // column names //
    //////////////////

    std::vector<std::string> cols;
    CHECK(read_vector_file(col_file, cols));
    const Index Nsample = cols.size();

    Mat mu, rho;

    CHECK(read_data_file(mu_file, mu));
    CHECK(read_data_file(rho_file, rho));

    const Index max_row = mu.rows();
    const Index Nind = mu.cols();
    const Index Nrho = rho.size();

    TLOG("Read mu and rho with " << Nind << " individuals, " << Nrho
                                 << " cells");

    // Partition columns into individuals
    std::minstd_rand rng{ std::random_device{}() };
    std::uniform_int_distribution<Index> unif_ind(0, Nind - 1);
    std::uniform_int_distribution<Index> unif_rho(0, Nrho - 1);

    auto _sample_ind = [&rng, &unif_ind](const Index) -> Index {
        return unif_ind(rng);
    };

    std::vector<Index> indv;
    indv.resize(Nsample);
    std::transform(std::begin(indv),
                   std::end(indv),
                   std::begin(indv),
                   _sample_ind);

    const std::vector<std::vector<Index>> partition = make_index_vec_vec(indv);

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

    std::vector<std::string> out_indv;
    out_indv.reserve(Nsample);

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

        nnz += sample_poisson_data(mu.col(i), rho_i, ncol, ofs, FS);
        ncol += n_i;

        ofs.close();
        temp_files.emplace_back(_temp_file);

        for (Index j = 0; j < n_i; ++j) {
            out_indv.emplace_back("Ind" + zeropad(i + 1, Nind));
        }
    }

    //////////////////////////
    // step 2: combine them //
    //////////////////////////

    const std::string mtx_file = output + ".mtx.gz";
    obgzf_stream ofs(mtx_file.c_str(), std::ios::out);

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
            std::remove(f.c_str());
    }

    ofs.close();

    if (file_exists(mtx_file + ".index"))
        std::remove((mtx_file + ".index").c_str());

    CHECK(mmutil::index::build_mmutil_index(mtx_file, mtx_file + ".index"));

    write_vector_file(output + ".ind.gz", out_indv);

    return EXIT_SUCCESS;
}

#endif
