#' A wrapper function to read a sparse submatrix
#'
#' @param mtx.file
#' @param sub.cols
#' @param memory.idx
#' @param memory.idx.file
#' @param verbose
#'
#' @return a sparse matrix
#'
read.sparse <- function(mtx.file,
                        sub.cols = NULL,
                        memory.idx = NULL,
                        memory.idx.file = paste0(mtx.file,".index"),
                        verbose = FALSE,
                        NUM_THREADS = 1) {

    if(is.null(memory.idx)) {
        stopifnot(file.exists(memory.idx.file))
        memory.idx <- rcpp_mmutil_read_index(memory.idx.file)
    }

    if(is.null(sub.cols)) {
        .info <- rcpp_mmutil_info(mtx.file)
        sub.cols <- 1:.info$max.col
    }

    .in <-
        rcpp_mmutil_read_columns_sparse(mtx.file,
                                        memory.idx,
                                        sub.cols,
                                        verbose,
                                        NUM_THREADS)

    Matrix::sparseMatrix(i=.in$row, j=.in$col, x=.in$val,
                         dims = c(.in$max.row, .in$max.col))
}

#' A wrapper function to read a dense submatrix
#'
#' @param mtx.file
#' @param sub.cols
#' @param memory.idx
#' @param memory.idx.file
#' @param verbose
#'
#' @return a dense matrix
#'
read.dense <- function(mtx.file,
                       sub.cols = NULL,
                       memory.idx = NULL,
                       memory.idx.file = paste0(mtx.file,".index"),
                       verbose = FALSE,
                       NUM_THREADS = 1) {

    if(is.null(memory.idx)) {
        stopifnot(file.exists(memory.idx.file))
        memory.idx <- rcpp_mmutil_read_index(memory.idx.file)
    }

    if(is.null(sub.cols)) {
        .info <- rcpp_mmutil_info(mtx.file)
        sub.cols <- 1:.info$max.col
    }

    rcpp_mmutil_read_columns(mtx.file,
                             memory.idx,
                             sub.cols,
                             verbose,
                             NUM_THREADS)    
}
