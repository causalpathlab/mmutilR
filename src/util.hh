#include <algorithm>
#include <cassert>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <Rcpp.h>

#ifndef UTIL_HH_
#define UTIL_HH_

std::string curr_time();
std::string zeropad(const int t, const int tmax);

#define TLOG(msg)                                                      \
    {                                                                  \
        Rcpp::Rcerr << "[" << curr_time() << "] " << msg << std::endl; \
    }
#define ELOG(msg)                                                              \
    {                                                                          \
        Rcpp::Rcerr << "[" << curr_time() << "] [Error] " << msg << std::endl; \
    }
#define WLOG(msg)                                                  \
    {                                                              \
        Rcpp::Rcerr << "[" << curr_time() << "] [Warning] " << msg \
                    << std::endl;                                  \
    }
#define ASSERT(cond, msg)                   \
    {                                       \
        if (!(cond)) {                      \
            ELOG(msg);                      \
            Rcpp::stop("assertion failed"); \
        }                                   \
    }

#define ASSERT_RET(cond, msg)    \
    {                            \
        if (!(cond)) {           \
            ELOG(msg);           \
            return EXIT_FAILURE; \
        }                        \
    }

#define CHECK(cond)                   \
    {                                 \
        if ((cond) != EXIT_SUCCESS) { \
            std::exit(1);             \
        }                             \
    }

#define CHK_ERR_EXIT(cond, msg)       \
    {                                 \
        if ((cond) != EXIT_SUCCESS) { \
            ELOG(msg);                \
            std::exit(1);             \
        }                             \
    }

#define CHK_ERR_RET(cond, msg)        \
    {                                 \
        if ((cond) != EXIT_SUCCESS) { \
            ELOG(msg);                \
            return EXIT_FAILURE;      \
        }                             \
    }

#define ERR_RET(cond, msg)       \
    {                            \
        if ((cond)) {            \
            ELOG(msg);           \
            return EXIT_FAILURE; \
        }                        \
    }

#endif
