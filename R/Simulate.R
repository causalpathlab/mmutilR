#' Simulate single-cell MTX data
#'
#' @param file.header file set header
#' @param nind # individuals
#' @param ngene # genes/features
#' @param ncausal # causal genes
#' @param ncovar.conf # confounding covariates
#' @param ncovar.batch # confounding batch variables
#' @param ncell.ind # cells per individual
#' @param pve.1 variance of treatment/disease effect
#' @param pve.c variance of confounding effect
#' @param pve.a variance of confounders to the assignment
#' @param rho.a rho ~ gamma(a, b)
#' @param rho.b rho ~ gamma(a, b)
#' @param rseed random seed
#' @param exposure.type "binary" or "continuous"
#'
#' @return
#' 
simulate.data <- function(file.header, ...) {
    .sim <- simulate_gamma_glm(...)
    .dat <- rcpp_mmutil_simulate_poisson(.sim$obs.mu,
                                         .sim$rho,
                                         file.header)

    list(indv = .sim, data = .dat)
}

#' Simulate individual-level effects by GLM
#'
#' @param nind # individuals
#' @param ngene # genes/features
#' @param ncausal # causal genes
#' @param ncovar.conf # confounding covariates
#' @param ncovar.batch # confounding batch variables
#' @param ncell.ind # cells per individual
#' @param pve.1 variance of treatment/disease effect
#' @param pve.c variance of confounding effect
#' @param pve.a variance of confounders to the assignment
#' @param rho.a rho ~ gamma(a, b)
#' @param rho.b rho ~ gamma(a, b)
#' @param rseed random seed
#' @param exposure.type "binary" or "continuous"
#'
simulate_gamma_glm <- function(nind = 40,
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
