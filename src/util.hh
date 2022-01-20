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

#define ERR_RET(cond, msg)       \
    {                            \
        if (cond) {              \
            ELOG(msg);           \
            return EXIT_FAILURE; \
        }                        \
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

#define ASSERT_RETM(cond, msg)             \
    {                                      \
        if (!(cond)) {                     \
            ELOG(msg);                     \
            return Rcpp::NumericMatrix(0); \
        }                                  \
    }

#define ASSERT_RETSM(cond, msg) \
    {                           \
        if (!(cond)) {          \
            ELOG(msg);          \
            return SpMat();     \
        }                       \
    }

#define ASSERT_RETL(cond, msg)           \
    {                                    \
        if (!(cond)) {                   \
            ELOG(msg);                   \
            return Rcpp::List::create(); \
        }                                \
    }

#define CHECK(cond)                   \
    {                                 \
        if ((cond) != EXIT_SUCCESS) { \
            Rcpp::stop("exit 1");     \
        }                             \
    }

#define CHK_ERR_EXIT(cond, msg)       \
    {                                 \
        if ((cond) != EXIT_SUCCESS) { \
            ELOG(msg);                \
            Rcpp::stop("exit 1");     \
        }                             \
    }

#define CHK_RET(cond)                 \
    {                                 \
        if ((cond) != EXIT_SUCCESS) { \
            return EXIT_FAILURE;      \
        }                             \
    }

#define CHK_BRK(cond)                 \
    {                                 \
        if ((cond) != EXIT_SUCCESS) { \
            break;                    \
        }                             \
    }

#define CHK_RETM(cond)                     \
    {                                      \
        if ((cond) != EXIT_SUCCESS) {      \
            return Rcpp::NumericMatrix(0); \
        }                                  \
    }

#define CHK_RETL(cond)                   \
    {                                    \
        if ((cond) != EXIT_SUCCESS) {    \
            return Rcpp::List::create(); \
        }                                \
    }

#define CHK_RET_(cond, msg)           \
    {                                 \
        if ((cond) != EXIT_SUCCESS) { \
            ELOG(msg);                \
            return EXIT_FAILURE;      \
        }                             \
    }

#define CHK_RETM_(cond, msg)               \
    {                                      \
        if ((cond) != EXIT_SUCCESS) {      \
            ELOG(msg);                     \
            return Rcpp::NumericMatrix(0); \
        }                                  \
    }

#define CHK_RETL_(cond, msg)             \
    {                                    \
        if ((cond) != EXIT_SUCCESS) {    \
            ELOG(msg);                   \
            return Rcpp::List::create(); \
        }                                \
    }

#endif
