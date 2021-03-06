% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{rcpp_mmutil_aggregate}
\alias{rcpp_mmutil_aggregate}
\title{Create pseudo-bulk data by aggregating columns}
\usage{
rcpp_mmutil_aggregate(
  mtx_file,
  row_file,
  col_file,
  r_cols = NULL,
  r_indv = NULL,
  r_annot = NULL,
  r_annot_mat = NULL,
  r_lab_name = NULL,
  r_trt = NULL,
  r_V = NULL,
  a0 = 1,
  b0 = 1,
  eps = 1e-08,
  knn = 10L,
  KNN_BILINK = 10L,
  KNN_NNLIST = 10L,
  NUM_THREADS = 1L,
  IMPUTE_BY_KNN = TRUE
)
}
\arguments{
\item{mtx_file}{data file}

\item{row_file}{row file}

\item{col_file}{column file}

\item{r_cols}{cell (col) names}

\item{r_indv}{membership for the cells (\code{r_cols})}

\item{r_annot}{label annotation for the (\code{r_cols})}

\item{r_annot_mat}{label annotation matrix (cell x type) (default: NULL)}

\item{r_lab_name}{label names (default: everything in \code{r_annot})}

\item{r_trt}{treatment assignment (default: NULL)}

\item{r_V}{SVD factors (default: NULL)}

\item{a0}{hyperparameter for gamma(a0, b0) (default: 1)}

\item{b0}{hyperparameter for gamma(a0, b0) (default: 1)}

\item{eps}{small number (default: 1e-8)}

\item{knn}{k-NN matching}

\item{KNN_BILINK}{# of bidirectional links (default: 10)}

\item{KNN_NNLIST}{# nearest neighbor lists (default: 10)}

\item{NUM_THREADS}{number of threads for multi-core processing}

\item{IMPUTE_BY_KNN}{imputation by kNN alone (default: TRUE)}
}
\value{
a list of inference results
}
\description{
Create pseudo-bulk data by aggregating columns
}
\examples{
options(stringsAsFactors = FALSE)
## combine two different mu matrices
set.seed(1)
rr <- rgamma(1000, 1, 1) # 1000 cells
mm.1 <- matrix(rgamma(100 * 3, 1, 1), 100, 3)
mm.1[1:10, ] <- rgamma(5, 1, .1)
mm.2 <- matrix(rgamma(100 * 3, 1, 1), 100, 3)
mm.2[11:20, ] <- rgamma(5, 1, .1)
mm <- cbind(mm.1, mm.2)
dat <- mmutilR::rcpp_mmutil_simulate_poisson(mm, rr, "sim_test")
rows <- read.table(dat$row)$V1
cols <- read.table(dat$col)$V1
## marker feature
markers <- list(
  annot.1 = list(
    ct1 = rows[1:10],
    ct2 = rows[11:20]
  )
)
## annotation on the MTX file
out <- mmutilR::rcpp_mmutil_annotate_columns(
       row_file = dat$row, col_file = dat$col,
       mtx_file = dat$mtx, pos_labels = markers)
annot <- out$annotation
## prepare column to individual
.ind <- read.table(dat$indv, col.names = c("col", "ind"))
.annot.ind <- .ind$ind[match(annot$col, .ind$col)]
## aggregate
agg <- mmutilR::rcpp_mmutil_aggregate(mtx_file = dat$mtx,
                                      row_file = dat$row,
                                      col_file = dat$col,
                                      r_cols = annot$col,
                                      r_indv = .annot.ind,
                                      r_annot = annot$argmax,
                                      r_lab_name = c("ct1", "ct2"))
## show average marker features
print(round(agg$mean[1:20, ]))
unlink(list.files(pattern = "sim_test"))
## Case-control simulation
.sim <- mmutilR::simulate_gamma_glm()
.dat <- mmutilR::rcpp_mmutil_simulate_poisson(.sim$obs.mu,
                                             .sim$rho,
                                             "sim_test")
## find column-wise annotation
.annot <- read.table(.dat$indv,
                     col.names = c("col", "ind"))
.annot$trt <- .sim$W[match(.annot$ind, 1:length(.sim$W))]
.annot$ct <- "ct1"
## simple PCA
.pca <- mmutilR::rcpp_mmutil_pca(.dat$mtx, 10)
.agg <- mmutilR::rcpp_mmutil_aggregate(mtx_file = .dat$mtx,
                                       row_file = .dat$row,
                                       col_file = .dat$col,
                                       r_cols = .annot$col,
                                       r_indv = .annot$ind,
                                       r_annot = .annot$ct,
                                       r_lab_name = "ct1",
                                       r_trt = .annot$trt,
                                       r_V = .pca$V,
                                       knn = 50,
                                       IMPUTE_BY_KNN = TRUE)
par(mfrow=c(1,3))
for(k in sample(.sim$causal, 3)) {
    y0 <- .agg$resid.mu[k, .sim$W == 0]
    y1 <- .agg$resid.mu[k, .sim$W == 1]
    boxplot(y0, y1)
}
## clean up temp directory
unlink(list.files(pattern = "sim_test"))

}
