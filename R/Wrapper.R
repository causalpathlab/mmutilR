read.sparse <- function(...){
    read.mtx.sparse(...)
}

read.dense <- function(...){
    read.mtx.dense(...)
}

#' A wrapper function to read a sparse submatrix
#'
#' @param mtx.file matrix market file
#' @param row.file row file (default: NULL)
#' @param col.file column file (default: NULL)
#' @param sub.cols column index (default: NULL, all)
#' @param memory.idx memory locations
#' @param memory.idx.file memory location file
#' @param max.row.word maximum number of words per each row (default: 2)
#' @param row.word.sep row word separator character (default: "_")
#' @param max.col.word maximum number of words per each col (default: 100)
#' @param col.word.sep column word separator character (default: "@")
#' @param verbose verbosity
#'
#' @return a sparse matrix
#'
read.mtx.sparse <- function(mtx.file,
                            row.file = NULL,
                            col.file = NULL,
                            sub.cols = NULL,
                            memory.idx = NULL,
                            memory.idx.file = paste0(mtx.file,".index"),
                            verbose = FALSE,
                            NUM_THREADS = 1,
                            max.row.word = 2,
                            row.word.sep = "_",
                            max.col.word = 100,
                            col.word.sep = "@") {

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

    ret <- Matrix::sparseMatrix(i=.in$row, j=.in$col, x=.in$val,
                                dims = c(.in$max.row, .in$max.col))

    if(!is.null(row.file)){
        .rows <- rcpp_mmutil_rownames(row.file,
                                      MAX_ROW_WORD = max.row.word,
                                      ROW_WORD_SEP = row.word.sep)
        if(length(.rows) == nrow(ret)){
            rownames(ret) <- .rows
        }
    }

    if(!is.null(col.file)){
        .cols <- rcpp_mmutil_colnames(col.file,
                                      MAX_COL_WORD = max.col.word,
                                      COL_WORD_SEP = col.word.sep)
        if(length(.cols) == ncol(ret)){
            colnames(ret) <- .cols
        }
    }

    return(ret)
}

#' A wrapper function to read a dense submatrix
#'
#' @param mtx.file matrix market file
#' @param row.file row file (default: NULL)
#' @param col.file column file (default: NULL)
#' @param sub.cols column index (default: NULL, all)
#' @param memory.idx memory locations
#' @param memory.idx.file memory location file
#' @param max.row.word maximum number of words per each row (default: 2)
#' @param row.word.sep row word separator character (default: "_")
#' @param max.col.word maximum number of words per each col (default: 100)
#' @param col.word.sep column word separator character (default: "@")
#' @param verbose verbosity
#'
#' @return a dense matrix
#'
read.mtx.dense <- function(mtx.file,
                           row.file = NULL,
                           col.file = NULL,
                           sub.cols = NULL,
                           memory.idx = NULL,
                           memory.idx.file = paste0(mtx.file,".index"),
                           verbose = FALSE,
                           NUM_THREADS = 1,
                           max.row.word = 2,
                           row.word.sep = "_",
                           max.col.word = 100,
                           col.word.sep = "@") {

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

    ret <- rcpp_mmutil_read_columns(mtx.file,
                                    memory.idx,
                                    sub.cols,
                                    verbose,
                                    NUM_THREADS)

    if(!is.null(row.file)){
        .rows <- rcpp_mmutil_rownames(row.file,
                                      MAX_ROW_WORD = max.row.word,
                                      ROW_WORD_SEP = row.word.sep)
        if(length(.rows) == nrow(ret)){
            rownames(ret) <- .rows
        }
    }

    if(!is.null(col.file)){
        .cols <- rcpp_mmutil_colnames(col.file,
                                      MAX_COL_WORD = max.col.word,
                                      COL_WORD_SEP = col.word.sep)
        if(length(.cols) == ncol(ret)){
            colnames(ret) <- .cols
        }
    }

    return(ret)
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
