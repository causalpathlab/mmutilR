#' Counterfactual confounder adjustment by
#' by Pairwise INdividual Effect matching
#'
#' @param mtx.data fileset.header ($mtx, $row, $col, $idx)
#' @param celltype celltype/cluster assignments (cells x 2 mapping, cells x 1, or just a single string)
#' @param celltype.mat celltype/cluster assignment matrix (cells x cell types)
#' @param cell2indv cell-level individual assignments (cells x 2), cell -> indv
#' @param eps small number (default: 1e-8)
#' @param knn number of neighbours `k` in kNN for matching
#' @param a0 hyperparameter for gamma(a0, b0) (default: 1)
#' @param b0 hyperparameter for gamma(a0, b0) (default: 1)
#' @param num.threads number of threads for multi-core processing
#' @param .rank SVD rank for spectral matching
#' @param .take.ln take log(1 + x) trans or not
#' @param .pca.reg regularization parameter (default = 1)
#' @param .col.norm column normalization for SVD
#' @param .em.iter EM iteration for factorization (default: 0)
#' @param .em.tol EM convergence (default: 1e-4)
#'
#' @return a list of sufficient statistics matrices
#'
#' @examples
#'
#' sim.data <- make.sc.deg("temp",
#'                               nind = 20,
#'                               ngene = 100,
#'                               ncausal = 3,
#'                               ncovar.conf = 3,
#'                               ncovar.batch = 0,
#'                               ncell.ind = 10,
#'                               pve.1 = .3,
#'                               pve.c = .5,
#'                               pve.a = .5,
#'                               rseed = 13,
#'                               exposure.type = "continuous")
#'
#' mtx.data <- sim.data$data
#'
#' cell2indv <- read.table(sim.data$data$indv,
#'                         header=FALSE,
#'                         col.names=c("cell","indv"))
#'
#' nind <- length(sim.data$indv$W)
#'
#' .pine <- make.pine(mtx.data,
#'                    "bulk",
#'                    cell2indv,
#'                    knn.cell = 50,
#'                    knn.indv = 1)
#'
#' .names <- lapply(colnames(.pine$delta), strsplit, split="[_]")
#' .names <- lapply(.names, function(x) unlist(x))
#' .pairs <- data.frame(do.call(rbind, .names), stringsAsFactors=FALSE)
#' colnames(.pairs) <- c("src","tgt","ct")
#'
#' W <- sim.data$indv$W
#' w.src <- W[as.integer(.pairs$src)]
#' w.tgt <- W[as.integer(.pairs$tgt)]
#' w.delta <- w.src - w.tgt
#'
#' .pine$delta[.pine$delta < 0] <- NA # numerical errors
#'
#' ncausal <- length(sim.data$indv$causal)
#' ngene <- nrow(.pine$delta)
#'
#' par(mfrow=c(2, ncausal))
#' for(k in sim.data$indv$causal){
#'     plot(w.delta, .pine$delta[k, ], xlab = "dW", ylab = "dY")
#' }
#'
#' for(k in sample(setdiff(1:ngene, sim.data$indv$causal), ncausal)){
#'     plot(w.delta, .pine$delta[k, ], xlab = "dW", ylab = "dY")
#' }
#'
#' unlink(list.files(pattern = "temp"))
#'
make.pine <- function(mtx.data,
                      celltype,
                      cell2indv,
                      knn.cell = 50,
                      knn.indv = 1,
                      celltype.mat = NULL,
                      .rank = 10,
                      .take.ln = TRUE,
                      .pca.reg = 1,
                      .col.norm = 1e4 ,
                      .em.iter = 0,
                      .em.tol = 1e-4,
                      num.threads = 1,
                      remove.dup = TRUE,
                      ...) {

    .input <- check.cocoa.input(mtx.data, celltype, cell2indv, NULL, celltype.mat)

    cells <- .input$cells
    individuals <- .input$individuals
    celltype.vec <- .input$celltype.vec
    celltype.lab <- .input$celltype.lab
    treatments <- .input$treatments

    message("Running PCA...")

    .pca <- rcpp_mmutil_pca(mtx_file = mtx.data$mtx,
                            RANK=.rank,
                            TAKE_LN = .take.ln,
                            TAU = .pca.reg,
                            COL_NORM = .col.norm,
                            EM_ITER = .em.iter,
                            EM_TOL = .em.tol)

    message("Estimating sufficient statistics by matching...")

    .stat <- rcpp_mmutil_aggregate_pairwise(mtx_file = mtx.data$mtx,
                                            row_file = mtx.data$row,
                                            col_file = mtx.data$col,
                                            r_cols = cells,
                                            r_indv = individuals,
                                            r_annot = celltype.vec,
                                            r_annot_mat = celltype.mat,
                                            r_lab_name = celltype.lab,
                                            r_V = .pca$V,
                                            knn_cell = knn.cell,
                                            knn_indv = knn.indv,
                                            NUM_THREADS = num.threads,
                                            ...)

    message("Finished PINE statistics preparation")

    if(remove.dup){
        .stat <- pine.remove.duplicated(.stat)
        message("Removed duplicated pairs")
    }
    return(.stat)
}

#' Check duplicated pairs of individuals in the matching results
#' and take the nearest ones
#' 
#' @param input the whole result of `make.pine`
#'
#' @return filtered `input` results
#' 
pine.remove.duplicated <- function(input){

    left <- pmin(input$knn$obs.index,
                 input$knn$matched.index)
    right <- pmax(input$knn$obs.index,
                  input$knn$matched.index)

    dd <- input$knn$dist
    dd.order <- order(dd)
    left.sorted <- left[dd.order]
    right.sorted <- right[dd.order]

    ij <- paste0(left.sorted, ".", right.sorted)
    ij.uniq <- unique(ij, fromLast=FALSE)

    if(length(ij) == length(ij.uniq)){
        warning("all the pairs are already unique")
        return(input)
    }

    message(paste0("reducing ", length(ij), " to ", length(ij.uniq)))

    ij.pos <- match(ij.uniq, ij)
    knn.pos <- dd.order[ij.pos]

    output <- list()
    for(k in names(input)){
        if(k != "knn" && is.matrix(input[[k]])){
            output[[k]] <- input[[k]][, knn.pos, drop = FALSE]
        }
    }

    output[["knn"]] <- list()

    for(k in names(input[["knn"]])){
        output[["knn"]][[k]] <- input[["knn"]][[k]][knn.pos]
    }

    return(output)
}

#' Counterfactual confounder adjustment for individual-level
#' differential expression analysis
#'
#' @param mtx.data fileset.header ($mtx, $row, $col, $idx)
#' @param celltype celltype/cluster assignments (cells x 2 mapping, cells x 1, or just a single string)
#' @param celltype.mat celltype/cluster assignment matrix (cells x cell types)
#' @param cell2indv cell-level individual assignments (cells x 2), cell -> indv
#' @param indv2exp individual treatment/exposure assignments (indiv x 2), indv -> exposure
#' @param knn number of neighbours `k` in kNN for matching
#' @param a0 hyperparameter for gamma(a0, b0) (default: 1)
#' @param b0 hyperparameter for gamma(a0, b0) (default: 1)
#' @param eps small number (default: 1e-8)
#' @param num.threads number of threads for multi-core processing
#' @param .rank SVD rank for spectral matching
#' @param .take.ln take log(1 + x) trans or not
#' @param .pca.reg regularization parameter (default = 1)
#' @param .col.norm column normalization for SVD
#' @param .em.iter EM iteration for factorization (default: 0)
#' @param .em.tol EM convergence (default: 1e-4)
#'
#' @return a list of sufficient statistics matrices
#'
#' @details
#'
#' If treatment/exposure variables are assigned, it will calculate
#' counterfactual individual effects (`resid.mu`).  Otherwise, it will
#' only compute average effect estimations (`mu`).
#'
#' @examples
#' 
#' sim.data <- make.sc.deg("temp",
#'                         nind = 20,
#'                         ngene = 100,
#'                         ncausal = 3,
#'                         ncovar.conf = 3,
#'                         ncovar.batch = 0,
#'                           ncell.ind = 10,
#'                           pve.1 = .3,
#'                           pve.c = .5,
#'                           pve.a = .5,
#'                           rseed = 13,
#'                           exposure.type = "binary")
#' 
#' mtx.data <- sim.data$data
#' 
#' cell2indv <- read.table(sim.data$data$indv,
#'                         header=FALSE,
#'                         col.names=c("cell","indv"))
#' 
#' nind <- length(sim.data$indv$W)
#' indv2exp <- data.frame(indv=1:nind, exp = sim.data$indv$W)
#' 
#' .cocoa <- make.cocoa(mtx.data, "bulk", cell2indv, indv2exp, knn = 50)
#' .stat <- make.cocoa(mtx.data, "bulk", cell2indv)
#' 
#' ncausal <- length(sim.data$indv$causal)
#' ngene <- nrow(.cocoa$resid.mu)
#' 
#' par(mfrow=c(2, ncausal))
#' W <- sim.data$indv$W
#' for(k in sim.data$indv$causal){
#'     y0 <- .cocoa$resid.mu[k, W == 0]
#'     y1 <- .cocoa$resid.mu[k, W == 1]
#'     boxplot(y0, y1, main=k)
#' }
#' 
#' W <- sim.data$indv$W
#' for(k in sim.data$indv$causal){
#'     y0 <- .stat$mu[k, W == 0]
#'     y1 <- .stat$mu[k, W == 1]
#'     boxplot(y0, y1, main=k)
#' }
#'
#' par(mfrow=c(2, ncausal))
#' for(k in sample(setdiff(1:ngene, sim.data$indv$causal), ncausal)){
#'     y0 <- .cocoa$resid.mu[k, W == 0]
#'     y1 <- .cocoa$resid.mu[k, W == 1]
#'     boxplot(y0, y1, main=k)
#' }
#' for(k in sample(setdiff(1:ngene, sim.data$indv$causal), ncausal)){
#'     y0 <- .stat$mu[k, W == 0]
#'     y1 <- .stat$mu[k, W == 1]
#'     boxplot(y0, y1, main=k)
#' }
#' 
#' unlink(list.files(pattern = "temp"))
#'
make.cocoa <- function(mtx.data,
                       celltype,
                       cell2indv,
                       indv2exp = NULL,
                       knn = 10,
                       celltype.mat = NULL,
                       .rank = 10,
                       .take.ln = TRUE,
                       .pca.reg = 1,
                       .col.norm = 1e4 ,
                       .em.iter = 0,
                       .em.tol = 1e-4,
                       num.threads = 1,
                       ...) {

    .input <- check.cocoa.input(mtx.data, celltype, cell2indv, indv2exp, celltype.mat)

    cells <- .input$cells
    individuals <- .input$individuals
    celltype.vec <- .input$celltype.vec
    celltype.mat <- .input$celltype.mat
    celltype.lab <- .input$celltype.lab
    treatments <- .input$treatments

    if(!is.null(treatments)) {

        message("Running PCA...")

        .pca <- rcpp_mmutil_pca(mtx_file = mtx.data$mtx,
                                RANK=.rank,
                                TAKE_LN = .take.ln,
                                TAU = .pca.reg,
                                COL_NORM = .col.norm,
                                EM_ITER = .em.iter,
                                EM_TOL = .em.tol)

        message("Estimating sufficient statistics by matching...")

        .stat <- rcpp_mmutil_aggregate(mtx_file = mtx.data$mtx,
                                       row_file = mtx.data$row,
                                       col_file = mtx.data$col,
                                       r_cols = cells,
                                       r_indv = individuals,
                                       r_annot = celltype.vec,
                                       r_annot_mat = celltype.mat,
                                       r_lab_name = celltype.lab,
                                       r_trt = treatments,
                                       r_V = .pca$V,
                                       knn = knn,
                                       IMPUTE_BY_KNN = TRUE,
                                       NUM_THREADS = num.threads,
                                       ...)

        .stat$indv2exp <- indv2exp

    } else {
        
        .stat <- rcpp_mmutil_aggregate(mtx_file = mtx.data$mtx,
                                       row_file = mtx.data$row,
                                       col_file = mtx.data$col,
                                       r_cols = cells,
                                       r_indv = individuals,
                                       r_annot = celltype.vec,
                                       r_annot_mat = celltype.mat,
                                       r_lab_name = celltype.lab,
                                       NUM_THREADS = num.threads,
                                       ...)

    }

    message("Finished CoCoA statistics preparation")
    return(.stat)
}

#' Check CoCoA input
#'
#' @param mtx.data fileset.header ($mtx, $row, $col, $idx)
#' @param celltype celltype/cluster assignments (cells x 2 mapping, cells x 1, or just a single string)
#' @param celltype.mat celltype/cluster assignment matrix (cells x cell types)
#' @param cell2indv cell-level individual assignments (cells x 2), cell -> indv
#' @param indv2exp individual treatment/exposure assignments (indiv x 2), indv -> exposure
#'
check.cocoa.input <- function(mtx.data,
                              celltype,
                              cell2indv,
                              indv2exp = NULL,
                              celltype.mat = NULL) {

    #####################################
    ## match cell -> indv -> treatment ##
    #####################################
    cells <- readLines(mtx.data$col)
    ncells <- length(cells)

    .match <- function(dict, .cells) {
        .order <- match(.cells, unlist(dict[,1]))
        out <- as.character(unlist(dict[.order,2]))
        out[is.na(out)] <- "NA"
        return(out)
    }
    individuals <- .match(cell2indv, cells)

    if(is.null(indv2exp)){
        treatments <- NULL
    } else {
        treatments <- .match(indv2exp, individuals) 
    }
    #############################
    ## match cell -> cell type ##
    #############################
    if(is.matrix(celltype.mat)){
        ## use probabilistic assignment matrix
        celltype.lab <- colnames(celltype.mat)
        if(is.null(celltype.lab)){
            celltype.lab <- paste0("ct.", 1:ncol(celltype.mat))
        }
        stopifnot(nrow(celltype.mat) == ncells)
        celltype.vec <- NULL
    } else {
        if(length(celltype) == 1) {
            celltype.vec <- rep(celltype, ncells)
        } else if(ncol(celltype) > 1) {
            celltype.vec <- .match(celltype, cells)
        } else {
            celltype.vec <- celltype
        }
    }
    stopifnot(length(celltype.vec) == ncells)
    celltype.lab = sort(unique(celltype.vec))

    if(length(celltype.lab) > ncells/5) {
        warning("too many cell type names?")
    }

    list(cells = cells,
         individuals = individuals,
         celltype.vec = celltype.vec,
         celltype.mat = celltype.mat,
         celltype.lab = celltype.lab,
         treatments = treatments)
}
