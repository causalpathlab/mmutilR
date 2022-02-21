#' A wrapper function to read a sparse submatrix
#'
#' @param mtx.file matrix market file
#' @param sub.cols column index (default: all)
#' @param memory.idx memory locations
#' @param memory.idx.file memory location file
#' @param verbose verbosity
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
        if(!file.exists(memory.idx.file))
            rcpp_mmutil_build_index(mtx.file, memory.idx.file)

        memory.idx <- rcpp_mmutil_read_index(memory.idx.file)
    }

    if(is.null(sub.cols)) {
        .info <- rcpp_mmutil_info(mtx.file)
        sub.cols <- 1:.info$max.col
    }

    stopifnot(length(sub.cols) == length(unique(sub.cols)))

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
#' @param mtx.file matrix market file
#' @param sub.cols column index (default: all)
#' @param memory.idx memory locations
#' @param memory.idx.file memory location file
#' @param verbose verbosity
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
        if(!file.exists(memory.idx.file))
            rcpp_mmutil_build_index(mtx.file, memory.idx.file)

        memory.idx <- rcpp_mmutil_read_index(memory.idx.file)
    }

    if(is.null(sub.cols)) {
        .info <- rcpp_mmutil_info(mtx.file)
        sub.cols <- 1:.info$max.col
    }

    stopifnot(length(sub.cols) == length(unique(sub.cols)))

    rcpp_mmutil_read_columns(mtx.file,
                             memory.idx,
                             sub.cols,
                             verbose,
                             NUM_THREADS)    
}
