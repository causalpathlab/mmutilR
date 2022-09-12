#' Simulate single-cell MTX data for DEG analysis
#'
#' @param file.header file set header
#' @param nind num of individuals
#' @param ngene num of genes/features
#' @param ncausal num of causal genes
#' @param ncovar.conf num of confounding covariates
#' @param ncovar.batch num of confounding batch variables
#' @param ncell.ind num of cells per individual
#' @param pve.1 variance of treatment/disease effect
#' @param pve.c variance of confounding effect
#' @param pve.a variance of confounders to the assignment
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
make.sc.deg <- function(file.header, 
                        nind = 40,
                        ngene = 1000,
                        ncausal = 5,
                        ncovar.conf = 1,
                        ncovar.batch = 3,
                        ncell.ind = 10,
                        pve.1 = 0.3,
                        pve.c = 0.5,
                        pve.a = 0.5,
                        rho.a = 2,
                        rho.b = 2,
                        rseed = 13,
                        exposure.type = c("binary","continuous")){

    .sim <- simulate_indv_glm(nind,
                              ngene,
                              ncausal,
                              ncovar.conf,
                              ncovar.batch,
                              ncell.ind,
                              pve.1,
                              pve.c,
                              pve.a,
                              rho.a,
                              rho.b,
                              rseed,
                              exposure.type)

    dir.create(dirname(file.header), recursive = TRUE, showWarnings = FALSE)

    .dat <- rcpp_mmutil_simulate_poisson(.sim$obs.mu,
                                         .sim$rho,
                                         file.header)

    list(indv = .sim, data = .dat)
}


#' Simulate individual-level eQTL data
#'
#' Possible confounding effect model:
#'
#' X -> U1 -> Y : harmful if U1 was adjusted (epigenetics, gene regulation, trans-effects)
#'      U0 -> Y : okay to adjust to boost the power (demographic variables, environments)
#'
#' Another rare possibility (we don't consider here):
#'
#' X -> Y -> Z : harmful if Z was adjusted (downstream phenotypes mediated by genes)
#'
#' @param X genotype matrix (individual x SNPs)
#' @param h2 heritability (proportion of variance of Y explained by genetic X)
#' @param pve.u1.by.x (proportion of variance of U1 explained by X)
#' @param pve.y.by.u1 (proportion of variance of Y explained by U1)
#' @param pve.y.by.u0 (proportion of variance of Y explained by U0)
#' @param n.causal.snps #(X directly affecting on Y)
#' @param n.causal.genes #(Y regulated by X)
#' @param n.u1 #(U1 variables)
#' @param n.u0 #(U0 variables)
#' @param n.genes total number of genes (Y variables)
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
#' * `indv$obs.mu`: observed (noisy) gene x individual matrix
#' * `indv$clean.mu`: clean gene x individual matrix
#' * `indv$X`: confounder x individual matrix
#' * `indv$W`: individual-level treatment assignment,
#' * `indv$rho`: sequencing depth
#' * `indv$causal`: causal genes
#'
make.sc.eqtl <- function(file.header,
                         X, h2,
                         pve.u1.by.x = .4,
                         pve.y.by.u1 = .2,
                         pve.y.by.u0 = .3,
                         n.causal.snps = 1,
                         n.causal.genes = 5,
                         n.u1 = 3,
                         n.u0 = 3,
                         n.genes = 50,
                         rho.a = 2,
                         rho.b = 2,
                         ncell.ind = 10,
                         ind.prob = NULL){

    .sim <- simulate_indv_eqtl(X = X, h2 = h2,
                               pve.u1.by.x = pve.u1.by.x,
                               pve.y.by.u1 = pve.y.by.u1,
                               pve.y.by.u0 = pve.y.by.u0,
                               n.causal.snps = n.causal.snps,
                               n.causal.genes = n.causal.genes,
                               n.genes = n.genes)

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

    .dat <- rcpp_mmutil_simulate_poisson(t(.mu), .rr, file.header, .ind)

    list(indv = .sim, data = .dat)
}

#' Simulate individual-level eQTL data
#'
#' @param X genotype matrix (individual x SNPs)
#' @param h2 heritability (proportion of variance of Y explained by genetic X)
#' @param pve.u1.by.x (proportion of variance of U1 explained by X)
#' @param pve.y.by.u1 (proportion of variance of Y explained by U1)
#' @param pve.y.by.u0 (proportion of variance of Y explained by U0)
#' @param n.causal.snps #(X directly affecting on Y)
#' @param n.causal.genes #(Y regulated by X)
#' @param n.u1 #(U1 variables)
#' @param n.u0 #(U0 variables)
#' @param n.genes total number of genes (Y variables)
#'
#' @return simulation results
#'
#' @details
#'
#' Possible confounding effect model:
#'
#' * X -> U1 -> Y : harmful if U1 was adjusted (epigenetics, gene regulation, trans-effects)
#'
#' * U0 -> Y : okay to adjust to boost the power (demographic variables, environments)
#'
#' Another rare possibility (we don't consider here):
#'
#' * X -> Y -> Z : harmful if Z was adjusted (downstream phenotypes mediated by genes)
#'
simulate_indv_eqtl <- function(X, h2,
                               pve.u1.by.x,
                               pve.y.by.u1,
                               pve.y.by.u0,
                               n.causal.snps,
                               n.causal.genes,
                               n.u1,
                               n.u0,
                               n.genes){

    if(!is.matrix(X)) { X <- as.matrix(X) }

    .scale <- function(.mat) {
        apply(.mat, 2, scale)
    }

    .rnorm <- function(d1, d2) {
        matrix(rnorm(d1*d2), d1, d2)
    }

    n.causal.snps <- min(n.causal.snps, floor(ncol(X)/2))
    causal.snps <- sample(ncol(X), n.causal.snps)
    non.causal.snps <- sample(setdiff(1:ncol(X), causal.snps), n.causal.snps)
    causal.genes <- sample(n.genes, min(n.genes, n.causal.genes))

    n.ind <- nrow(X)

    ######################
    ## check parameters ##
    ######################

    stopifnot(pve.u1.by.x > 0 & pve.u1.by.x < 1)
    pve.y.tot <- h2 + pve.y.by.u1 + pve.y.by.u0
    stopifnot(pve.y.tot > 0 & pve.y.tot < 1)

    ##################
    ## X -> U1 -> Y ##
    ##################
    xx.conf <- X[, non.causal.snps, drop=FALSE]
    xx.conf[is.na(xx.conf)] <- 0

    u1.by.x <- xx.conf %*% .rnorm(ncol(xx.conf), n.u1)
    u1 <- (.scale(u1.by.x) * sqrt(pve.u1.by.x) +
           .scale(.rnorm(n.ind, n.u1)) * sqrt(1-pve.u1.by.x))

    y.by.u1 <- u1 %*% .rnorm(n.u1, n.genes)

    #############
    ## U0 -> Y ##
    #############
    u0 <- .rnorm(n.ind, n.u0)
    y.by.u0 <- u0 %*% .rnorm(n.u0, n.genes)

    ############
    ## X -> Y ##
    ############
    xx.causal <- X[, causal.snps, drop=FALSE]
    xx.causal[is.na(xx.causal)] <- 0

    y.by.x <- .rnorm(n.ind, n.genes)
    xy <- .rnorm(n.causal.snps, n.causal.genes)
    y.by.x[, causal.genes] <- xx.causal %*% xy

    y.err <- .rnorm(n.ind, n.genes)

    y.obs <- (.scale(y.by.x) * sqrt(h2) +
              .scale(y.by.u1) * sqrt(pve.y.by.u1) +
              .scale(y.by.u0) * sqrt(pve.y.by.u0) +
              .scale(y.err) * sqrt(1 - pve.y.tot))

    list(y = y.obs, x = X, y.true = y.by.x,
         causal.snps = causal.snps,
         causal.genes = causal.genes,
         u1 = u1, u0 = u0)
}

#' Simulate individual-level effects by GLM
#'
#' @param nind num of individuals
#' @param ngene num of genes/features
#' @param ncausal num of causal genes
#' @param ncovar.conf num of confounding covariates
#' @param ncovar.batch num of confounding batch variables
#' @param ncell.ind num of cells per individual
#' @param pve.1 variance of treatment/disease effect
#' @param pve.c variance of confounding effect
#' @param pve.a variance of confounders to the assignment
#' @param rho.a rho ~ gamma(a, b)
#' @param rho.b rho ~ gamma(a, b)
#' @param rseed random seed
#' @param exposure.type "binary" or "continuous"
#'
simulate_indv_glm <- function(nind = 40,
                              ngene = 1000,
                              ncausal = 5,
                              ncovar.conf = 1,
                              ncovar.batch = 3,
                              ncell.ind = 10,
                              pve.1 = 0.3,
                              pve.c = 0.5,
                              pve.a = 0.5,
                              rho.a = 2,
                              rho.b = 2,
                              rseed = 13,
                              exposure.type = c("binary","continuous")){

    exposure.type <- match.arg(exposure.type)

    set.seed(rseed)
    stopifnot((pve.1 + pve.c) < 1)

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

    ## sample model parameters
    ##  nind number of individuals/samples
    ##  ncovar.conf number of covar shared
    ##  ncovar.batch number of covar on mu, batch effect
    ##  ngenes number of genes/features
    ##  ncausal number of causal genes
    sample.seed.data <- function(nind, ncovar.conf, ncovar.batch, pve) {

        if(ncovar.conf > 0) {
            xx <- .rnorm(nind, ncovar.conf) # covariates
        } else {
            xx <- .zero(nind, 1) # empty
        }

        if(ncovar.batch > 0) {
            xx.mu <- .rnorm(nind, ncovar.batch) # covariates
        } else {
            xx.mu <- .zero(nind, 1) # empty
        }

        ## Biased assignment mechanism
        .delta <- .rnorm(ncol(xx), 1) / sqrt(ncol(xx))

        true.logit <- .rnorm(nind, 1)

        logit <- .scale(xx %*% .delta) * sqrt(pve)
        logit <- logit + true.logit * sqrt(1 - pve)

        if(exposure.type == "binary"){
            ww <- rbinom(prob=.sigmoid(logit), n=nind, size=1)
        } else {
            ww <- logit
        }

        list(w = ww, lib = true.logit, x = xx, x.mu = xx.mu)
    }

    .param <- sample.seed.data(nind, ncovar.conf, ncovar.batch, pve.a)

    nn <- length(.param$w)

    xx <- .param$x       # covariates shared --> confounding
    xx.mu <- .param$x.mu # covariates on mu only --> non-confounding
    ww <- .param$w       # stochastic assignment

    causal <- sample(ngene, ncausal) # causal genes

    ## Treatment effects ##

    sample.w.rand <- function(j) {
        ## make sure that we don't create unlabeled positive genes
        r <- rnorm(length(ww))
        r <- .scale(matrix(residuals(lm(r ~ ww)), ncol=1))
        return(matrix(r * rnorm(1), ncol=1))
    }

    ln.mu.w <- sapply(1:ngene, sample.w.rand)
    ln.mu.w[, causal] <- sapply(1:ncausal, function(j) ww * rnorm(1))

    #########################
    ## confounding effects ##
    #########################

    ln.mu.x <- xx %*% .rand(ncol(xx), ngene)

    #########################
    ## other batch effects ##
    #########################

    .batch <- xx.mu %*% .rand(ncol(xx.mu), ngene)

    ln.mu.x <- ln.mu.x + .batch

    ########################
    ## unstructured noise ##
    ########################

    ln.mu.eps <- .rnorm(nn, ngene)

    #############################
    ## combine all the effects ##
    #############################

    ln.mu <-
        .scale(ln.mu.w) * sqrt(pve.1) +
        .scale(ln.mu.x) * sqrt(pve.c) +
        .scale(ln.mu.eps) * sqrt(1 - pve.1 - pve.c)

    ln.mu <- .scale(ln.mu)  # numerical stability
    ln.mu[ln.mu > 8] <- 8   #
    ln.mu[ln.mu < -8] <- -8 #

    mu <- exp(ln.mu)

    ##########################
    ## unconfounded signals ##
    ##########################

    clean.ln.mu <- .scale(ln.mu.w) * sqrt(pve.1) +
        .scale(ln.mu.eps) * sqrt(1 - pve.1)

    clean.mu <- exp(clean.ln.mu)

    ######################
    ## sequencing depth ##
    ######################

    rr <- rgamma(ncell.ind * nn, shape=rho.a, scale=1/rho.b)

    cells <- 1:length(rr)

    list(obs.mu = t(mu),
         clean.mu = t(clean.mu),
         X = t(xx),
         W = ww,
         rho = rr,
         causal = sort(causal))
}
