% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/Simulate.R
\name{simulate_indv_glm}
\alias{simulate_indv_glm}
\title{Simulate individual-level effects by GLM}
\usage{
simulate_indv_glm(
  nind = 40,
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
  exposure.type = c("binary", "continuous"),
  smudge = 1
)
}
\arguments{
\item{nind}{num of individuals}

\item{ncausal}{num of causal genes}

\item{nreverse}{num of anti-causal genes}

\item{ncovar.conf}{num of confounding covariates}

\item{ncovar.batch}{num of confounding batch variables}

\item{num.mixtures}{num of cell mixtures}

\item{pve.1}{variance of treatment/disease effect}

\item{pve.c}{variance of confounding effect}

\item{pve.a}{variance of confounders to the assignment}

\item{pve.r}{variance of reverse causation}

\item{rseed}{random seed}

\item{exposure.type}{"binary" or "continuous"}

\item{ngene}{num of genes/features}

\item{ngene.covar}{num of genes affected by covariates}
}
\description{
Simulate individual-level effects by GLM
}
