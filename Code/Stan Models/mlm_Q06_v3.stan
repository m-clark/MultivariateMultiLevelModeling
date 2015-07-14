data {
  int<lower=1> n_stu;                             // number of students
  int<lower=1> S;                                 // number of potential scores per student
  int<lower=1> n_obs_tot;                         // total number of observed outcomes
  vector[n_obs_tot] y_obs;                        // observed outcomes, long format
  int n_obs_stu[n_stu];                        // number of observed outcomes per student
  int idx_start[n_stu];                        // start index
  int idx_end[n_stu];                          // end index
  matrix[n_obs_tot, S] miss_ind;                  // stacked missing indicator matrices
}                                              

parameters {                                   
  row_vector[S] mu;                               // score means (wide form)

  cholesky_factor_corr[S] cholCorrSigma;          // chol decomp of corr matrix for Sigma (residuals)
  vector<lower=0>[S] scaleSigma;                  // scale for Sigma
}                                              

model {                                        
  matrix[S,S] cholCovSigma;                       // scaled chol decomp of Sigma

  // priors 
  scaleSigma ~ cauchy(0, 5);
  cholCorrSigma ~ lkj_corr_cholesky(2.0);
  cholCovSigma <- diag_matrix(scaleSigma) * cholCorrSigma;

  mu ~ normal(0, 10);

  // likelihood
  for (n in 1:n_stu) {
    matrix[n_obs_stu[n_stu], S] M;
    M <- block(miss_ind, idx_start[n_stu], 1, n_obs_stu[n_stu], S);
    segment(y_obs, idx_start[n_stu], n_obs_stu[n_stu]) ~ multi_normal_cholesky(M * to_vector(mu), M * cholCovSigma * M' );
    }
}

generated quantities{
  // Residual dispersion matrix
  cov_matrix[S] Sigma;                            

  Sigma <- quad_form_diag(tcrossprod(cholCorrSigma), scaleSigma);
}
