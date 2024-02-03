#include "mmutil.hh"
#include "rcpp_util.hh"

//' Build sparse matrix from triplets
//'
//' @param A_ij_list list(src.index, tgt.index, [weights])
//' @param Nrow number of rows
//' @param Ncol number of columns
//' @param symmetrize symmetrize A matrix
//'
//' @return a sparse matrix A
//'
// [[Rcpp::export]]
Eigen::SparseMatrix<float>
rcpp_build_sparse_mat(const Rcpp::List A_ij_list,
                      const std::size_t Nrow,
                      const std::size_t Ncol,
                      const bool symmetrize = true)
{
    Eigen::SparseMatrix<float> A;

    rcpp::util::build_sparse_mat(Rcpp::List(A_ij_list), Nrow, Ncol, A);

    Eigen::SparseMatrix<float> ret;
    if (symmetrize && A.rows() == A.cols()) {
        const Eigen::SparseMatrix<float> At = A.transpose();
        ret = (A + At) * 0.5;
    } else {
        return A;
    }

    return ret;
}
