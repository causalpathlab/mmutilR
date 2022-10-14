#' Simulate single-cell MTX data for DEG analysis
#'
#' @param file.header file set header
#' @param nind num of individuals
#' @param ngenes num of genes/features
#' @param ncausal num of causal genes
#' @param ncovar.conf num of confounding covariates
#' @param ncovar.batch num of confounding batch variables
#' @param ngenes.covar num of genes affected by covariates
#' @param num.mixtures num of cell mixtures
#' @param ncell.ind num of cells per individual
#' @param pve.1 variance of treatment/disease effect
#' @param pve.c variance of confounding effect
#' @param pve.a variance of confounders to the assignment
#' @param pve.r variance of reverse causation
#' @param rho.a rho ~ gamma(a, b)
#' @param rho.b rho ~ gamma(a, b)
#' @param rseed random seed
#' @param exposure.type "binary" or "continuous"
#'
#' @return simulation results
#'
#' @details
#'
#' The simulation result list will have two lists:
#'
#' `data`:
#'
#' * a matrix market data file `data$mtx`
#' * a file with row names `data$row`
#' * a file with column names `data$col`
#' * an indexing file for the columns `data$idx`
#' * a mapping file between column and individual names "indv"
#'
#' `indv`:
#'
#' * `obs.mu` observed (noisy) gene x individual matrix
#' * `clean.mu` clean gene x individual matrix
#' * `X` confounder x individual matrix
#' * `W` individual-level treatment assignment
#' * `rho` sequencing depth
#' * `causal` causal genes
#'
make.sc.deg.data <- function(file.header,
                             nind = 40,
                             ngenes = 1000,
                             ncausal = 5,
                             ncovar.conf = 3,
                             ncovar.batch = 0,
                             ncell.ind = 10,
                             ngenes.covar = ngenes,
                             num.mixtures = 1,
                             pve.1 = 0.3,
                             pve.c = 0.5,
                             pve.a = 0.5,
                             pve.r = 0,
                             rho.a = 2,
                             rho.b = 2,
                             rseed = 13,
                             exposure.type = c("binary","continuous")){

    .sim <- simulate_indv_glm(nind = nind,
                              ngenes = ngenes,
                              ncausal = ncausal,
                              ncovar.conf = ncovar.conf,
                              ncovar.batch = ncovar.batch,
                              ngenes.covar = ngenes.covar,
                              num.mixtures = num.mixtures,
                              pve.1 = pve.1,
                              pve.c = pve.c,
                              pve.r = pve.r,
                              pve.a = pve.a,
                              rseed = rseed,
                              exposure.type = exposure.type)

    ncells <- ncell.ind * nind

    dir.create(dirname(file.header), recursive = TRUE, showWarnings = FALSE)
    .data <- rcpp_mmutil_simulate_poisson_mixture(.sim$mu.list,
                                                  ncells,
                                                  file.header,
                                                  gam_alpha=rho.a,
                                                  gam_beta=rho.b)

    list(indv = .sim$indv,
         causal = .sim$causal,
         reverse = .sim$reverse,
         data = .data)
}

#' Simulate individual-level eQTL data
#'
#' The goal is to have unbiased estimates of Y ~ X
#'
#' However, possible confounding effect models lurking:
#'
#' Y1 ~ X + U
#' Y0 ~ Y1 + U
#'
#' @param X genotype matrix (individual x SNPs)
#' @param h2 heritability (proportion of variance of Y explained by genetic X)
#' @param n.causal.snps X variables directly affecting on Y
#' @param n.causal.genes Y variables directly regulated by X
#' @param pve.y.by.u0 proportion of variance of Y explained by U0
#' @param n.u0 number of covariates on Y
#' @param pve.u1.by.x proportion of variance of U1 explained by X
#' @param pve.y.by.u1 proportion of variance of Y explained by U1
#' @param n.u1 number of covariates on Y
#' @param pve.interaction proportion of variance of Y explained by interaction
#' @param n.interaction number of genes interacting with the causal genes
#' @param n.genes total number of genes (Y variables)
#'
#' @param rho.a rho ~ gamma(a, b)
#' @param rho.b rho ~ gamma(a, b)
#' @param ncell.ind number of cells per individual
#'
#' @return simulation results
#'
#' @details
#'
#' The simulation result list will have two lists:
#'
#' `data`:
#'
#' * `data$mtx`: a matrix market data file
#' * `data$row`: a file with row names
#' * `data$col`: a file with column names
#' * `data$idx`: an indexing file for the columns
#' * `data$indv`: a mapping file between column and individual names
#'
#' `indv`:
#' * `indv$y`: observed (noisy) individual x gene matrix
#' * `indv$x`: observed individual x variants genotype matrix
#' * `indv$causal.snps`: causal variants (X variables)
#' * `indv$causal.genes`: causal genes (Y variables)
#'
make.sc.eqtl <- function(file.header,
                         X, h2,
                         n.causal.snps = 1,
                         n.causal.genes = 5,
                         pve.y.by.u0 = .3,
                         n.u0 = 3,
                         pve.u1.by.x = .8,
                         pve.y.by.u1 = .3,
                         n.u1 = 3,
                         pve.interaction = 0.5,
                         n.interaction = 0,
                         n.genes = 50,
                         n.covar.genes = n.genes,
                         rho.a = 2,
                         rho.b = 2,
                         ncell.ind = 10,
                         ind.prob = NULL){

    .sim <- simulate_indv_eqtl(X = X, h2 = h2,
                               n.causal.snps = n.causal.snps,
                               n.causal.genes = n.causal.genes,
                               pve.y.by.u0 = pve.y.by.u0,
                               n.u0 = n.u0,
                               pve.u1.by.x = pve.u1.by.x,
                               pve.y.by.u1 = pve.y.by.u1,
                               n.u1 = n.u1,
                               pve.interaction = pve.interaction,
                               n.interaction = n.interaction,
                               n.genes = n.genes,
                               n.covar.genes = n.genes)

    n.ind <- nrow(.sim$x)

    if(is.null(ind.prob)){
        .prob <- rep(1/n.ind, n.ind)
    } else {
        .prob <- ind.prob
    }

    stopifnot(length(.prob) == n.ind)

    .ind <- sample(n.ind,
                   n.ind * ncell.ind,
                   replace=TRUE,
                   prob=.prob)

    ######################
    ## Sequencing depth ##
    ######################

    .rr <- rgamma(ncell.ind * n.ind, shape=rho.a, scale=1/rho.b)
    y <- apply(.sim$y, 2, scale) # just for numerical stability
    y[y > 8] <- 8
    y[y < -8] <- -8
    .mu <- exp(y)

    dir.create(dirname(file.header), recursive = TRUE, showWarnings = FALSE)

    .data <- rcpp_mmutil_simulate_poisson(t(.mu), .rr, file.header, .ind)

    list(indv = .sim, data = .data)
}

#' Simulate individual-level eQTL data
#'
#' @param X genotype matrix (individual x SNPs)
#' @param h2 heritability (proportion of variance of Y explained by genetic X)
#' @param n.causal.snps X variables directly affecting on Y
#' @param n.causal.genes Y variables directly regulated by X
#' @param pve.y.by.u0 proportion of variance of Y explained by U0
#' @param n.u0 number of covariates on Y
#' @param pve.u1.by.x proportion of variance of U1 explained by X
#' @param pve.y.by.u1 proportion of variance of Y explained by U1
#' @param n.u1 number of covariates on Y
#' @param pve.interaction proportion of variance of Y explained by interaction
#' @param n.interaction number of genes interacting with the causal genes
#' @param n.genes total number of genes (Y variables)
#' @param n.covar.genes number of genes affected by covariates
#'
#' @return simulation results
#'
simulate_indv_eqtl <- function(X, h2,
                               n.causal.snps,
                               n.causal.genes,
                               pve.y.by.u0,
                               n.u0,
                               pve.u1.by.x,
                               pve.y.by.u1,
                               n.u1,
                               pve.interaction,
                               n.interaction,
                               n.genes,
                               n.covar.genes){

    if(!is.matrix(X)) { X <- as.matrix(X) }

    .scale <- function(.mat) {
        apply(.mat, 2, scale)
    }

    .rnorm <- function(d1, d2) {
        matrix(rnorm(d1*d2), d1, d2)
    }

    `%c%` <- function(mat, cols){
        mat[, cols, drop = FALSE]
    }

    n.ind <- nrow(X)

    causal.genes <- sample(n.genes, min(n.genes, n.causal.genes))

    n.causal.snps <- min(n.causal.snps, floor(ncol(X)/2))
    causal.snps <- sample(ncol(X), n.causal.snps)

    non.causal <- setdiff(1:ncol(X), causal.snps)
    u1.snps <- sample(non.causal, n.causal.snps)

    ######################
    ## check parameters ##
    ######################

    pve.y.tot <- h2 + pve.y.by.u0 + pve.y.by.u1
    stopifnot(pve.y.tot >= 0 & pve.y.tot <= 1)

    y.err <- .rnorm(n.ind, n.genes)

    #############
    ## U0 -> Y ##
    #############
    u0 <- .rnorm(n.ind, n.u0)
    y.by.u0 <- u0 %*% .rnorm(n.u0, n.genes)

    if(n.covar.genes < n.genes){
        ## some genes are not affected by covariates
        n0 <- n.genes - n.covar.genes
        y.by.u0[, sample(n.genes, n0)] <- .rnorm(n.ind, n0)
    }

    #############
    ## U1 -> Y ##
    #############
    xx <- X %c% u1.snps
    xx[is.na(xx)] <- 0
    .beta.u <- .rnorm(ncol(xx), n.u1)
    u1 <- (.scale(.rnorm(n.ind, n.u1)) * sqrt(1 - pve.u1.by.x) +
           .scale(xx %*% .beta.u) * sqrt(pve.u1.by.x))
    .beta.yu <- .rnorm(ncol(u1), n.genes)
    y.by.u1 <- u1 %*% .beta.yu

    if(n.covar.genes < n.genes){
        ## some genes are not affected by covariates
        n0 <- n.genes - n.covar.genes
        y.by.u1[, sample(n.genes, n0)] <- .rnorm(n.ind, n0)
    }

    ############
    ## X -> Y ##
    ############
    xx.causal <- X %c% causal.snps
    xx.causal[is.na(xx.causal)] <- 0

    ## null effect
    non.causal <- setdiff(1:ncol(X), causal.snps)
    y.by.x <- lapply(1:n.genes, function(j) {
        xx <- X %c% sample(non.causal, n.causal.snps)
        xx <- apply(xx, 2, sample) # permute
        xx %*% .rnorm(n.causal.snps, 1)
    })
    y.by.x <- as.matrix(do.call(cbind, y.by.x))

    ## direct effect
    xy <- .rnorm(n.causal.snps, n.causal.genes)
    y.by.x[, causal.genes] <- (xx.causal %*% xy)

    ## introduce interactions between genes
    if(n.interaction > 0) {
        non.causal <- setdiff(1:n.genes, causal.genes)
        interacting <- sample(non.causal, n.interaction)
        y.to.y <- .rnorm(n.causal.genes, n.interaction)
        y.by.x[, interacting] <- (y.by.x %c% causal.genes) %*% y.to.y
    }

    ## stochastic version of Y
    y.obs <- (.scale(y.by.x) * sqrt(h2) +
              .scale(y.by.u0) * sqrt(pve.y.by.u0) +
              .scale(y.by.u1) * sqrt(pve.y.by.u1) +
              .scale(y.err) * sqrt(1 - pve.y.tot))

    list(y = y.obs, x = X, y.true = y.by.x,
         u1.snps = u1.snps,
         causal.snps = causal.snps,
         causal.genes = causal.genes,
         u0 = u0, u1 = u1)
}

#' Simulate individual-level effects by GLM
#'
#' @param nind num of individuals
#' @param ngene num of genes/features
#' @param ncausal num of causal genes
#' @param nreverse num of anti-causal genes
#' @param ncovar.conf num of confounding covariates
#' @param ncovar.batch num of confounding batch variables
#' @param ngene.covar num of genes affected by covariates
#' @param num.mixtures num of cell mixtures
#' @param pve.1 variance of treatment/disease effect
#' @param pve.c variance of confounding effect
#' @param pve.a variance of confounders to the assignment
#' @param pve.r variance of reverse causation
#' @param rseed random seed
#' @param exposure.type "binary" or "continuous"
#'
simulate_indv_glm <- function(nind = 40,
                              ngenes = 1000,
                              ncausal = 5,
                              nreverse = 0,
                              ncovar.conf = 3,
                              ncovar.batch = 0,
                              ngenes.covar = ngenes,
                              num.mixtures = 1,
                              pve.1 = 0.3,
                              pve.c = 0.5,
                              pve.a = 0.5,
                              pve.r = 0,
                              rseed = 13,
                              exposure.type = c("binary","continuous")){

    exposure.type <- match.arg(exposure.type)

    stopifnot((pve.1 + pve.c) < 1)
    ## stopifnot((ncovar.conf + ncovar.batch) > 0)

    ## simple concat
    `%&%` <- function(a,b) {
        paste0(a,b)
    }

    ## random standard Normal
    ##  n1
    ##  n2
    .rnorm <- function(n1, n2) {
        matrix(rnorm(n1 * n2), nrow = n1, ncol = n2)
    }

    ## random from collection
    ##  n1
    ##  n2
    .rand <- function(n1, n2, .sample = c(-1, 1)) {
        matrix(sample(.sample, n1 * n2, TRUE), nrow = n1, ncol = n2)
    }

    ## just a zero matrix
    ##  n1
    ##  n2
    .zero <- function(n1, n2) {
        matrix(0, nrow = n1, ncol = n2)
    }

    .sigmoid <- function(x) {
        1/(1 + exp(-x))
    }

    .scale <- function(x) {
        .sd <- pmax(apply(x, 2, sd), 1e-8)
        sweep(x, 2, .sd, `/`)
    }

    sample.biased.assignment <- function(nind,
                                         ncovar.conf,
                                         ncovar.batch,
                                         nreverse,
                                         pve.r,
                                         pve.bias,
                                         exposure.type){

        ## (1) sample confounding variables: U ~ epsilon
        if(ncovar.conf > 0) {
            cv <- .rnorm(nind, ncovar.conf) # covariates
        } else {
            cv <- .zero(nind, 1) # empty
        }

        if(ncovar.batch > 0) {
            uv <- .rnorm(nind, ncovar.batch) # covariates
        } else {
            uv <- .zero(nind, 1) # empty
        }

        ## (2) sample reverse causation genes: mu(reverse) -> X
        if(nreverse > 0){
            mu.rev <- .scale(.rnorm(nind, nreverse))
        } else {
            mu.rev <- .rnorm(nind, 1) * 0
        }

        ## (3) sample composite treatment assignment: W ~ mu(reverse) + U + epsilon
        random.assign <- .scale(.rnorm(nind, 1))
        cv.rev <- cbind(cv, mu.rev)
        biased.assign <- (random.assign * sqrt(1 - pve.bias - pve.r) +
                          .scale(cv %*% .rnorm(ncol(cv), 1)) * sqrt(pve.bias) +
                          .scale(mu.rev %*% .rnorm(ncol(mu.rev), 1)) %*% sqrt(pve.r))

        if(exposure.type == "binary"){
            ww <- rbinom(prob=.sigmoid(biased.assign), n=nind, size=1)
            ww <- matrix(ww, ncol = 1)
        } else {
            ww <- biased.assign
        }

        list(assignment = ww,
             covariates = cbind(cv, uv),
             mu.reverse = mu.rev)
    }

    causal <- sample(ngenes, ncausal) # causal genes
    reverse <- c()
    if(nreverse > 0){  # reverse causation genes
        reverse <- sample(setdiff(1:ngenes, causal), nreverse)
    }

    ## Take biased assignments
    ass <- sample.biased.assignment(nind,
                                    ncovar.conf,
                                    ncovar.batch,
                                    nreverse,
                                    pve.r,
                                    pve.a,
                                    exposure.type)

    ## causal effects invariant across mixture components
    tau <- .rnorm(1, ncausal)

    sample.mu <- function(k){
        ## a. non-causal genes w/o influence from the treatment assignment
        ln.mu.tau.k <- .rand(nind, ngenes, unlist(ass$assignment))
        ## b. causal genes exert invariant effects
        ln.mu.tau.k[, causal] <- ass$assignment %*% tau
        ## c. influence from the context-specific covariates
        beta.covar <- .rnorm(ncol(ass$covariates), ngenes)
        ln.mu.covar.k <- ass$covariates %*% beta.covar
        if(ngenes.covar < ngenes){
            .genes <- sample(ngenes, ngenes.covar)
            n0 <- ngenes - ngenes.covar
            ln.mu.covar.k[, - .genes] <- .rnorm(nrow(ln.mu.covar.k), n0)
        }
        ## d. unstructured noise
        ln.mu.eps.k <- .rnorm(nind, ngenes)
        ln.mu.k <- (.scale(ln.mu.tau.k) * sqrt(pve.1) +
                    .scale(ln.mu.covar.k) * sqrt(pve.c) +
                    .scale(ln.mu.eps.k) * sqrt(1 - pve.1 - pve.c))
        ## e. reverse causation
        if(nreverse > 0){
            ln.mu.k[, reverse] <- .scale(ass$mu.reverse)
        }
        exp(t(ln.mu.k))
    }

    list(indv = ass,
         causal = causal,
         reverse = reverse,
         mu.list = lapply(1:num.mixtures, sample.mu))
}
