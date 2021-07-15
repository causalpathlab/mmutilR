#include <getopt.h>

#include "mmutil_annotate.hh"

#ifndef MMUTIL_ANNOTATE_COL_
#define MMUTIL_ANNOTATE_COL_

//////////////////////
// Argument parsing //
//////////////////////

template <typename T>
int
parse_annotation_options(const int argc,     //
                         const char *argv[], //
                         T &options)
{
    const char *_usage =
        "\n"
        "[Arguments]\n"
        "--mtx (-m)         : data MTX file\n"
        "--data (-m)        : data MTX file\n"
        "\n"
        "--svd_u            : SVD U alternative input (instead of MTX)\n"
        "--svd_d            : SVD D (X = U D V')\n"
        "--svd_v            : SVD V (X = U D V')\n"
        "\n"
        "--col (-c)         : data column file\n"
        "--feature (-f)     : data row file (features)\n"
        "--row (-f)         : data row file (features)\n"
        "--ann (-a)         : row annotation file; each line contains a tuple of feature and label\n"
        "--anti (-A)        : row anti-annotation file; each line contains a tuple of feature and label\n"
        "--qc (-q)          : row annotation file for Q/C; each line contains a tuple of feature and minimum score\n"
        "--out (-o)         : Output file header\n"
        "\n"
        "--standardize (-S) : Standardize data (default: false)\n"
        "--log_scale (-L)   : Data in a log-scale (default: false)\n"
        "--raw_scale (-R)   : Data in a raw-scale (default: true)\n"
        "\n"
        "--batch_size (-B)  : Batch size (default: 100000)\n"
        "--kappa_max (-K)   : maximum kappa value (default: 100)\n"
        "\n"
        "--em_iter (-i)     : EM iteration (default: 100)\n"
        "--em_tol (-t)      : EM convergence criterion (default: 1e-4)\n"
        "\n"
        "--verbose (-v)     : Set verbose (default: false)\n"
        "\n"
        "--randomize        : Randomized initialization (default: false)\n"
        "\n";

    const char *const short_opts =
        "m:U:D:V:c:f:a:A:o:I:B:K:M:SLRi:t:hbd:u:r:l:kv0";

    const option long_opts[] = {
        { "mtx", required_argument, nullptr, 'm' },        //
        { "data", required_argument, nullptr, 'm' },       //
        { "svd_u", required_argument, nullptr, 'U' },      //
        { "svd_d", required_argument, nullptr, 'D' },      //
        { "svd_v", required_argument, nullptr, 'V' },      //
        { "svd_U", required_argument, nullptr, 'U' },      //
        { "svd_D", required_argument, nullptr, 'D' },      //
        { "svd_v", required_argument, nullptr, 'V' },      //
        { "col", required_argument, nullptr, 'c' },        //
        { "row", required_argument, nullptr, 'f' },        //
        { "feature", required_argument, nullptr, 'f' },    //
        { "ann", required_argument, nullptr, 'a' },        //
        { "anti", required_argument, nullptr, 'A' },       //
        { "qc", required_argument, nullptr, 'q' },         //
        { "out", required_argument, nullptr, 'o' },        //
        { "standardize", no_argument, nullptr, 'S' },      //
        { "std", no_argument, nullptr, 'S' },              //
        { "log_scale", no_argument, nullptr, 'L' },        //
        { "raw_scale", no_argument, nullptr, 'R' },        //
        { "batch_size", required_argument, nullptr, 'B' }, //
        { "kappa_max", required_argument, nullptr, 'K' },  //
        { "em_iter", required_argument, nullptr, 'i' },    //
        { "em_tol", required_argument, nullptr, 't' },     //
        { "help", no_argument, nullptr, 'h' },             //
        { "verbose", no_argument, nullptr, 'v' },          //
        { "randomize", no_argument, nullptr, '0' },        //
        { "rand", no_argument, nullptr, '0' },             //
        { nullptr, no_argument, nullptr, 0 }
    };

    while (true) {
        const auto opt = getopt_long(argc,                      //
                                     const_cast<char **>(argv), //
                                     short_opts,                //
                                     long_opts,                 //
                                     nullptr);

        if (-1 == opt)
            break;

        switch (opt) {
        case 'm':
            options.mtx = std::string(optarg);
            break;

        case 'U':
            options.svd_u = std::string(optarg);
            break;

        case 'D':
            options.svd_d = std::string(optarg);
            break;

        case 'V':
            options.svd_v = std::string(optarg);
            break;

        case 'c':
            options.col = std::string(optarg);
            break;

        case 'f':
            options.row = std::string(optarg);
            break;

        case 'a':
            options.ann = std::string(optarg);
            break;

        case 'A':
            options.anti_ann = std::string(optarg);
            break;

        case 'q':
            options.qc_ann = std::string(optarg);
            break;

        case 'o':
            options.out = std::string(optarg);
            break;

        case 'i':
            options.max_em_iter = std::stoi(optarg);
            break;

        case 't':
            options.em_tol = std::stof(optarg);
            break;

        case 'B':
            options.batch_size = std::stoi(optarg);
            break;
        case 'S':
            options.do_standardize = true;
            break;
        case 'L':
            options.log_scale = true;
            options.raw_scale = false;
            break;
        case 'R':
            options.log_scale = false;
            options.raw_scale = true;
            break;
        case '0':
            options.randomize_init = true;
            break;

        case 'v': // -v or --verbose
            options.verbose = true;
            break;

        case 'K':
            options.kappa_max = std::stof(optarg);
            break;
        case 'h': // -h or --help
        case '?': // Unrecognized option
            std::cerr << _usage << std::endl;
            return EXIT_FAILURE;
        default: //
                 ;
        }
    }

    ERR_RET(!file_exists(options.col), "No COL data file");
    ERR_RET(!file_exists(options.row), "No ROW data file");

    if (!file_exists(options.mtx)) {
        ERR_RET(!file_exists(options.svd_u), "No SVD U data file");
        ERR_RET(!file_exists(options.svd_d), "No SVD D data file");
        ERR_RET(!file_exists(options.svd_v), "No SVD V data file");
    }

    return EXIT_SUCCESS;
}

#endif
