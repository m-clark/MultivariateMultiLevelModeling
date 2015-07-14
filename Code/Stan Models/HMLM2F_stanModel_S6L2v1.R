# Multivariate multi-level model with factor structure: HMLM2F
stanModel = "
data {
  int<lower=1> N;                                                    // number of students (total)
  int<lower=1> J;                                                    // number of classes
  int<lower=1> S;                                                    // number of potential scores per student
  int<lower=1> L;                                                    // number of (classroom-level) latent traits
  int<lower=1,upper=J> jj[N];                                        // student identifier
  row_vector[S] y[N];                                                // Outcomes (wide form) 
}

parameters {
  row_vector[S] mu;                                                  // score means (wide form)

  cholesky_factor_corr[S] cholCorrSigma;                             // chol decomp of corr matrix for Sigma (residuals)
  vector<lower=0>[S] scaleSigma;                                     // scale for Sigma

  cholesky_factor_corr[L] cholCorrTau;                               // chol decomp of corr matrix for Tau
  vector<lower=0>[L] scaleTau;                                       // scale for Tau
  matrix[J,L] eta; 
  
  real<lower=0> lambda_2_1;                                          // Unknown factor loadings:
  real<lower=0> lambda_3_1;
  real<lower=0> lambda_5_2;
  real<lower=0> lambda_6_2;
}

model {
  matrix[S,S] cholCovSigma;                                          // scaled chol decomp of Sigma
  matrix[L,L] cholCovTau;                                            // scaled chol decomp of Tau
  matrix[J,S] lambdaeta; 
  matrix[N,S] termeta; 
  matrix[N,S] X; 
  matrix[N,S] exp_y;

  // priors for dispersion matrices
  scaleSigma ~ cauchy(0, 5);
  cholCorrSigma ~ lkj_corr_cholesky(2.0);

  scaleTau ~ cauchy(0, 5);
  cholCorrTau ~ lkj_corr_cholesky(2.0);

  cholCovSigma <- diag_matrix(scaleSigma) * cholCorrSigma;
  cholCovTau <- diag_matrix(scaleTau) * cholCorrTau;

  // other priors
  mu ~ normal(0, 10);
  lambda_2_1 ~ normal(0, 1);
  lambda_3_1 ~ normal(0, 1);
  lambda_5_2 ~ normal(0, 1);
  lambda_6_2 ~ normal(0, 1);

  // Latent-trait effects (wide form)
  for (j in 1:J) 
    eta[j] ~ multi_normal_cholesky(rep_vector(0, L), cholCovTau);
  
  // Weighted latent-trait effects (wide form)
  for (j in 1:J) {
    lambdaeta[j,1] <- 1 * eta[j,1];
    lambdaeta[j,2] <- lambda_2_1 * eta[j,1];
    lambdaeta[j,3] <- lambda_3_1 * eta[j,1];
    lambdaeta[j,4] <- 1 * eta[j,2];
    lambdaeta[j,5] <- lambda_5_2 * eta[j,2];
    lambdaeta[j,6] <- lambda_6_2 * eta[j,2];
  }
  
  // Full term containing classroom latent effects (wide form)
  for (n in 1:N) 
    for (s in 1:S) 
      termeta[n,s] <- lambdaeta[jj[n],s];
  
  // Full term containing linear predictor
  for (n in 1:N)
    for (s in 1:S) 
      X[n,s] <- mu[s];
  
  exp_y <- X + termeta;
      
  // likelihood
  for (n in 1:N)
    y[n] ~ multi_normal_cholesky(exp_y[n], cholCovSigma);
}

generated quantities{
  cov_matrix[S] Sigma;                                               // Residual vcov matrix
  cov_matrix[L] Tau;                                                 // Factor vcov matrix

  Sigma <- quad_form_diag(tcrossprod(cholCorrSigma), scaleSigma);
  Tau <- quad_form_diag(tcrossprod(cholCorrTau), scaleTau);
}
"
