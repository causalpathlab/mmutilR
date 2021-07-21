#include "mmutil.hh"
#include "mmutil_annotate.hh"

// Read label definition
//
// list(lab1 = c("feature1", "feature2", "feature3"),
//      lab2 = c("feature4", "feature5", "feature6"))
template <typename T1, typename T2>
void
read_label_defn(Rcpp::List input, std::vector<std::tuple<T1, T2>> &ret)
{
    if (input.size() == 0)
        return;

    Rcpp::CharacterVector labels(input.names());

    for (Index i = 0; i < input.size(); ++i) {
        const T2 lab_i = Rcpp::as<T2>(labels[i]);
        Rcpp::StringVector features = input[i];
        for (auto f : features) {
            ret.emplace_back(std::make_tuple<>(f, lab_i));
            // TLOG(f << " -> " << lab_i);
        }
    }
}

// Read label definition
//
// input : named vector
template <typename T1, typename T2>
void
read_qc_defn(Rcpp::NumericVector input, std::vector<std::tuple<T1, T2>> &ret)
{
    if (input.size() == 0)
        return;

    Rcpp::CharacterVector features(input.names());

    for (Index i = 0; i < input.size(); ++i) {
        const T1 feature = Rcpp::as<T1>(features[i]);
        const T2 threshold = static_cast<T2>(input[i]);
        ret.emplace_back(std::make_tuple<>(feature, threshold));
    }
}

// read label names
std::vector<std::string>
read_label_names(Rcpp::List input)
{
    std::vector<std::string> ret;
    if (input.size() == 0)
        return ret;

    Rcpp::CharacterVector labels(input.names());
    return std::vector<std::string>(labels.begin(), labels.end());
}

auto
read_annotation_input(
    const std::string row_file,
    Rcpp::List pos_labels,
    Rcpp::Nullable<Rcpp::List> r_neg_labels = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> r_qc_labels = R_NilValue)
{

    std::vector<std::string> rows;
    CHECK(read_vector_file(row_file, rows));

    auto row_pos = make_position_dict<std::string, Index>(rows);

    using pair_vector_t = std::vector<std::tuple<std::string, std::string>>;

    const Index num_annot = pos_labels.size();

    std::vector<std::string> _labels_tot;

    std::vector<std::shared_ptr<pair_vector_t>> pos_pairs;
    pos_pairs.reserve(num_annot);

    std::vector<std::shared_ptr<pair_vector_t>> neg_pairs;
    neg_pairs.reserve(num_annot);

    for (Index a = 0; a < num_annot; ++a) {
        pos_pairs.emplace_back(std::make_shared<pair_vector_t>());
        neg_pairs.emplace_back(std::make_shared<pair_vector_t>());
    }

    for (Index a = 0; a < num_annot; ++a) {
        pair_vector_t &a_pos_pairs = *pos_pairs.at(a).get();
        read_label_defn(pos_labels[a], a_pos_pairs);
        auto temp = read_label_names(pos_labels[a]);
        _labels_tot.insert(std::end(_labels_tot),
                           std::begin(temp),
                           std::end(temp));
    }

    if (r_neg_labels.isNotNull()) {

        Rcpp::List neg_labels(r_neg_labels);

        for (Index a = 0; a < neg_labels.size(); ++a) {
            pair_vector_t &a_neg_pairs = *neg_pairs.at(a).get();
            read_label_defn(neg_labels[a], a_neg_pairs);
            auto temp = read_label_names(neg_labels[a]);
            _labels_tot.insert(std::end(_labels_tot),
                               std::begin(temp),
                               std::end(temp));
        }
    }

    std::vector<std::tuple<std::string, Scalar>> qc_pairs;

    if (r_qc_labels.isNotNull()) {
        Rcpp::NumericVector qc_labels(r_qc_labels);
        read_qc_defn(qc_labels, qc_pairs);
    }

    std::vector<std::string> labels;
    std::unordered_map<std::string, Index> label_pos;
    std::tie(std::ignore, labels, label_pos) =
        make_indexed_vector<std::string, Index>(_labels_tot);

    /////////////////////////////////////
    // Build annotation model and stat //
    /////////////////////////////////////

    std::vector<std::shared_ptr<annotation_stat_t>> stat_vector;

    for (Index a = 0; a < num_annot; ++a) {
        Mat l1, l0;
        SpMat lq;

        pair_vector_t &a_pos_pairs = *pos_pairs.at(a).get();
        pair_vector_t &a_neg_pairs = *neg_pairs.at(a).get();

        std::tie(l1, l0, lq) = read_annotation_matched(row_pos,
                                                       label_pos,
                                                       a_pos_pairs,
                                                       a_neg_pairs,
                                                       qc_pairs);

        stat_vector.emplace_back(
            std::make_shared<annotation_stat_t>(l1, l0, lq));

        annotation_stat_t &stat = *stat_vector.at(a).get();
        stat.labels.insert(std::end(stat.labels),
                           std::begin(labels),
                           std::end(labels));
    }

    return std::make_tuple<>(stat_vector, labels, rows);
}

//' Annotate columns by marker feature information
//'
//' @param row_file row file
//' @param col_file column file
//' @param pos_labels markers
//' @param neg_labels anti-markers
//' @param qc_labels feature-specific threshold
//' @param mtx_file data file
//' @param param_kappa_max
//'
//' @return a list of inference results
//'
//' @examples
//' options(stringsAsFactors = FALSE)
//' ## combine two different mu matrices
//' rr <- rgamma(1000, 1, 1) # 1000 cells
//' mm.1 <- matrix(rgamma(100 * 3, 1, 1), 100, 3)
//' mm.1[1:10, ] <- rgamma(5, 1, .1)
//' mm.2 <- matrix(rgamma(100 * 3, 1, 1), 100, 3)
//' mm.2[11:20, ] <- rgamma(5, 1, .1)
//' mm <- cbind(mm.1, mm.2)
//' dat <- mmutilR::rcpp_simulate_poisson_data(mm, rr, "sim_test")
//' rows <- read.table(dat$row)$V1
//' cols <- read.table(dat$col)$V1
//' ## marker feature
//' markers <- list(
//'   annot.1 = list(
//'     ct1 = rows[1:10],
//'     ct2 = rows[11:20]
//'   )
//' )
//' ## annotation on the MTX file
//' out <- mmutilR::rcpp_annotate_columns(dat$row, dat$col,
//'        mtx_file = dat$mtx, pos_labels = markers)
//' annot <- out$annotation
//' .pca <- mmutilR::rcpp_mmutil_pca(dat$mtx, 3, TAKE_LN = TRUE)
//' out.df <- data.frame(col = as.integer(annot$col),
//'                      argmax = annot$argmax)
//' out.df <- cbind(out.df, PC=.pca$V)
//' plot(out.df$PC.1, out.df$PC.2, xlab = "PC1", ylab = "PC2")
//' ct1 <- which(out.df$argmax == "ct1")
//' points(out.df$PC.1[ct1], out.df$PC.2[ct1], pch = 19, col = 2)
//' ct2 <- which(out.df$argmax == "ct2")
//' points(out.df$PC.1[ct2], out.df$PC.2[ct2], pch = 19, col = 3)
//' ## annotation on the PC results
//' out.2 <- mmutilR::rcpp_annotate_columns(dat$row, dat$col,
//'          pos_labels = markers,
//'          r_U = .pca$U, r_D = .pca$D, r_V = .pca$V)
//' annot <- out.2$annotation
//' out.df <- data.frame(col = as.integer(annot$col),
//'                      argmax = annot$argmax)
//' out.df <- cbind(out.df, PC=.pca$V)
//' plot(out.df$PC.1, out.df$PC.2, xlab = "PC1", ylab = "PC2")
//' ct1 <- which(out.df$argmax == "ct1")
//' points(out.df$PC.1[ct1], out.df$PC.2[ct1], pch = 19, col = 2)
//' ct2 <- which(out.df$argmax == "ct2")
//' points(out.df$PC.1[ct2], out.df$PC.2[ct2], pch = 19, col = 3)
//' unlink(list.files(pattern = "sim_test"))
//'
// [[Rcpp::export]]
Rcpp::List
rcpp_annotate_columns(
    const std::string row_file,
    const std::string col_file,
    Rcpp::List pos_labels,
    Rcpp::Nullable<Rcpp::List> r_neg_labels = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> r_qc_labels = R_NilValue,
    const std::string mtx_file = "",
    Rcpp::Nullable<const Eigen::MatrixXf> r_U = R_NilValue,
    Rcpp::Nullable<const Eigen::MatrixXf> r_D = R_NilValue,
    Rcpp::Nullable<const Eigen::MatrixXf> r_V = R_NilValue,
    const double KAPPA_MAX = 100.,
    const bool TAKE_LN = false,
    const std::size_t BATCH_SIZE = 10000,
    const std::size_t EM_ITER = 100,
    const double EM_TOL = 1e-4,
    const bool VERBOSE = false,
    const bool DO_STD = false)
{

    //////////////////////////////
    // parsing input arguments  //
    //////////////////////////////

    std::vector<std::string> columns;
    CHECK(read_vector_file(col_file, columns));

    annotation_options_t options;
    options.mtx_file = mtx_file;
    options.kappa_max = KAPPA_MAX;
    options.log_scale = TAKE_LN;
    options.batch_size = BATCH_SIZE;
    options.max_em_iter = EM_ITER;
    options.em_tol = EM_TOL;
    options.verbose = VERBOSE;
    options.do_standardize = DO_STD;

    std::vector<std::shared_ptr<annotation_stat_t>> stat_vector;
    std::vector<std::string> labels;
    std::vector<std::string> rows;

    std::tie(stat_vector, labels, rows) =
        read_annotation_input(row_file, pos_labels, r_neg_labels, r_qc_labels);

    TLOG("Parsed label annotations");

    ////////////////////////
    // training the model //
    ////////////////////////

    const std::size_t num_annot = stat_vector.size();

    std::vector<std::shared_ptr<annotation_model_t>> model_vector;
    std::vector<Scalar> max_prob;
    std::vector<Scalar> max_score;
    std::vector<std::string> argmax;
    Mat Pr;

    if (file_exists(mtx_file)) {

        TLOG("Training on the MTX file");

        mm_data_loader_t data_loader(options);

        std::tie(model_vector, argmax, max_prob, max_score, Pr) =
            train_model(stat_vector, data_loader, options);

    } else if (r_U.isNotNull() && r_D.isNotNull() && r_V.isNotNull()) {

        TLOG("Training on the SVD results");

        const Mat U = Rcpp::as<Mat>(r_U);
        const Mat D = Rcpp::as<Mat>(r_D);
        const Mat V = Rcpp::as<Mat>(r_V);

        svd_data_loader_t data_loader(U, D, V, options.log_scale);
        std::tie(model_vector, argmax, max_prob, max_score, Pr) =
            train_model(stat_vector, data_loader, options);
    } else {
        ELOG("No proper data set was given");
        return Rcpp::List::create();
    }

    TLOG("Organizing results...");

    ////////////////////
    // output results //
    ////////////////////

    Rcpp::List param_list = Rcpp::List::create(num_annot);
    for (Index a = 0; a < num_annot; ++a) {
        annotation_model_t &annot = *model_vector.at(a).get();
        annotation_stat_t &stat = *stat_vector.at(a).get();
        Mat unc = stat.unc_stat * stat.nsize.cwiseInverse().asDiagonal();
        Mat unc_anti =
            stat.unc_stat_anti * stat.nsize.cwiseInverse().asDiagonal();
        Rcpp::List a_param =
            Rcpp::List::create(Rcpp::_["mu"] = annot.mu,
                               Rcpp::_["mu.anti"] = annot.mu_anti,
                               Rcpp::_["unc"] = unc,
                               Rcpp::_["unc.anti"] = unc_anti);
        param_list[a] = a_param;
    }

    annotation_stat_t &stat = *stat_vector.at(0).get();
    std::vector<Index> &subrow = stat.subrow;

    Rcpp::StringVector markers;
    std::for_each(subrow.begin(), subrow.end(), [&](const auto r) {
        markers.push_back(rows.at(r));
    });

    Rcpp::List annot_tab =
        Rcpp::List::create(Rcpp::_["col"] = columns,
                           Rcpp::_["argmax"] = argmax,
                           Rcpp::_["max.prob"] = max_prob,
                           Rcpp::_["max.ln.prob"] = max_score);

    TLOG("Done");

    Rcpp::StringVector r_labels(labels.begin(), labels.end());

    return Rcpp::List::create(Rcpp::_["parameters"] = param_list,
                              Rcpp::_["markers"] = markers,
                              Rcpp::_["labels"] = r_labels,
                              Rcpp::_["annotation"] = annot_tab,
                              Rcpp::_["P.annot"] = Pr);
}
