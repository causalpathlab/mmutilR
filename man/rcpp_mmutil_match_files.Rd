% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{rcpp_mmutil_match_files}
\alias{rcpp_mmutil_match_files}
\title{Match the columns of two MTX files}
\usage{
rcpp_mmutil_match_files(
  src_mtx,
  tgt_mtx,
  knn,
  RANK,
  TAKE_LN = TRUE,
  TAU = 1,
  COL_NORM = 10000,
  EM_ITER = 10L,
  EM_TOL = 1e-04,
  LU_ITER = 5L,
  KNN_BILINK = 10L,
  KNN_NNLIST = 10L,
  row_weight_file = "",
  NUM_THREADS = 1L,
  BLOCK_SIZE = 10000L
)
}
\arguments{
\item{src_mtx}{source data file}

\item{tgt_mtx}{target data file}

\item{knn}{k-nearest neighbour}

\item{RANK}{SVD rank}

\item{TAKE_LN}{take log(1 + x) trans or not}

\item{TAU}{regularization parameter (default = 1)}

\item{COL_NORM}{column normalization (default: 1e4)}

\item{EM_ITER}{EM iteration for factorization (default: 10)}

\item{EM_TOL}{EM convergence (default: 1e-4)}

\item{LU_ITER}{LU iteration (default: 5)}

\item{KNN_BILINK}{num. of bidirectional links (default: 10)}

\item{KNN_NNLIST}{num. of nearest neighbor lists (default: 10)}

\item{row_weight_file}{row-wise weight file}

\item{NUM_THREADS}{number of threads for multi-core processing}

\item{BLOCK_SIZE}{number of columns per block}
}
\value{
a list of source, target, distance
}
\description{
Match the columns of two MTX files
}
\examples{
## Generate some data
rr <- rgamma(100, 1, 6) # 100 cells
mm <- matrix(rgamma(100 * 3, 1, 1), 100, 3)
dat <- mmutilR::rcpp_mmutil_simulate_poisson(mm, rr, "sim_test")
.matched <- mmutilR::rcpp_mmutil_match_files(dat$mtx, dat$mtx,
                                             knn=1, RANK=5)
## Do they match well?
mean(.matched$src.index == .matched$tgt.index)
summary(.matched$dist)
## clean up temp directory
unlink(list.files(pattern = "sim_test"))

}
