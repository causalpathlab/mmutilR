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

#' Read a vector of string names
#' @param .file file name
#' @return a vector of strings
read.vec <- function(.file) {

    .ends.with <- function(x, pat) {
        .len <- nchar(x)
        .end <- base::substr(x, .len - nchar(pat) + 1, .len)
        return(.end == pat)
    }

    if(.ends.with(.file, ".gz")) {
        con <- gzfile(.file)
        ret <- readLines(con)
        close(con)
    } else {
        ret <- readLines(.file)
    }
    return(ret)
}

#' Create a list of MTX-related files
#' @param .hdr file set header name
#' @return a list of file names
fileset.list <- function(.hdr){
    .names <- c("mtx","row","col","idx")
    .out <- paste0(.hdr, c(".mtx.gz", ".rows.gz", ".cols.gz", ".mtx.gz.index"))
    .out <- as.list(.out)
    names(.out) <- .names
    .out
}

#' Write matrix market file set
#'
#' @param out.mtx a sparse matrix
#' @param out.rows a vector of rows
#' @param out.cols a vector of columns
#' @param output output file set header
#' 
write.sparse <- function(out.mtx, out.rows, out.cols, output){
    
    out.mtx.file <- paste0(output, ".mtx.gz")
    out.cols.file <- paste0(output, ".cols.gz")
    out.rows.file <- paste0(output, ".rows.gz")

    .write.vec <- function(.vec, .file) {
        con <- gzfile(.file)
        cat(.vec, file=con, sep="\n")
        close(con)
    }

    rcpp_mmutil_write_mtx(out.mtx, out.mtx.file)
    .write.vec(out.cols, out.cols.file)
    .write.vec(out.rows, out.rows.file)

    return(fileset.list(output))
}

#' A wrapper function that concatenates two files vertically
#' 
#' @param top.hdr a file header for the top files
#' @param bottom.hdr a file header for the bottom files
#' @param out.hdr output file header
#'
#' @return a list of the resulting file names
vcat.sparse <- function(top.hdr, bottom.hdr, out.hdr){

    dir.create(dirname(out.hdr), recursive = TRUE, showWarnings = FALSE)

    out.mtx.file <- paste0(out.hdr, ".mtx.gz")
    out.cols.file <- paste0(out.hdr, ".cols.gz")
    out.rows.file <- paste0(out.hdr, ".rows.gz")

    top.mtx.file <- paste0(top.hdr, ".mtx.gz")
    top.rows.file <- paste0(top.hdr, ".rows.gz")
    top.cols.file <- paste0(top.hdr, ".cols.gz")
    bottom.mtx.file <- paste0(bottom.hdr, ".mtx.gz")
    bottom.rows.file <- paste0(bottom.hdr, ".rows.gz")
    bottom.cols.file <- paste0(bottom.hdr, ".cols.gz")

    .files <- c(top.mtx.file,
                top.rows.file,
                top.cols.file,
                bottom.mtx.file,
                bottom.rows.file,
                bottom.cols.file)

    stopifnot(all(file.exists(.files)))

    top.mtx <- read.sparse(top.mtx.file)
    bottom.mtx <- read.sparse(bottom.mtx.file)

    top.cols <- read.vec(top.cols.file)
    top.rows <- read.vec(top.rows.file)

    bottom.cols <- read.vec(bottom.cols.file)
    bottom.rows <- read.vec(bottom.rows.file)

    .top.idx <- which(top.cols %in% bottom.cols)
    .bottom.idx <- match(top.cols[.top.idx], bottom.cols)

    ## no overlapping row names
    stopifnot(!any(top.rows %in% bottom.rows))

    out.mtx <- rbind(top.mtx[, .top.idx, drop = FALSE],
                     bottom.mtx[, .bottom.idx, drop = FALSE])

    out.rows <- c(top.rows, bottom.rows)
    out.cols <- top.cols[.top.idx]
    write.sparse(out.mtx, out.rows, out.cols, out.hdr)
}

