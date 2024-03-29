% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{rcpp_mmutil_simulate_poisson}
\alias{rcpp_mmutil_simulate_poisson}
\title{Simulation Poisson data based on Mu}
\usage{
rcpp_mmutil_simulate_poisson(
  mu,
  rho,
  output,
  r_indv = NULL,
  rseed = 42L,
  MAX_COL_WORD = 100L,
  COL_WORD_SEP = "@"
)
}
\arguments{
\item{mu}{depth-adjusted mean matrix (M x n)}

\item{rho}{column depth vector (N x 1), N= num. of cells}

\item{output}{header for ${output}.{mtx.gz,cols.gz,indv.gz}}

\item{r_indv}{N x 1 individual membership (1-based, \link{1 .. n})}

\item{rseed}{random seed}
}
\value{
a list of file names: {output}.{mtx,rows,cols}.gz
}
\description{
M= num. of features and n= num. of indv
}
