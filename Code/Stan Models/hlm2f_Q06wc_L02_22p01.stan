data {
  int<lower=1> N;                                 // total number of students
  int<lower=1> J;                                 // number of classes
  int<lower=1> S;                                 // number of potential scores per student
  int<lower=1> L;                                 // number of (classroom-level) latent traits
  int<lower=0> P;                                 // number of L2 and L3 covariates
  int<lower=1,upper=J> jj[N];                     // student classroom identifier
  matrix[N,P] X;                             	    // covariables
  row_vector[S] y[N];                             // outcomes (wide form) 
}                                              

parameters {                                   
  row_vector[S] mu;                               // score means (wide form)
  vector[P] gamma;                                // covariate effects
  matrix[J,L] eta_wide;                           // latent traits (wide form)

  cholesky_factor_corr[S] cholCorrSigma;          // chol decomp of corr matrix for Sigma (residuals)
  vector<lower=0>[S] scaleSigma;                  // scale for Sigma
  cholesky_factor_corr[L] cholCorrTau;            // chol decomp of corr matrix for Tau
  vector<lower=0>[L] scaleTau;                    // scale for Tau

  // Unknown factor loadings
  real lambda_2_1; 
  real lambda_3_1;                    
  real lambda_5_2;                    
  real lambda_6_2;                    
}                                              

transformed parameters {          
  // Factor loadings matrix 
  matrix[S,L] Lambda; 
  Lambda[1,1] <- 1;            Lambda[1,2] <- 0; 
  Lambda[2,1] <- lambda_2_1;   Lambda[2,2] <- 0; 
  Lambda[3,1] <- lambda_3_1;   Lambda[3,2] <- 0; 
  Lambda[4,1] <- 0;            Lambda[4,2] <- 1; 
  Lambda[5,1] <- 0;            Lambda[5,2] <- lambda_5_2; 
  Lambda[6,1] <- 0;            Lambda[6,2] <- lambda_6_2; 
}                                              
 
model {                                        
  matrix[S,S] cholCovSigma;                       // scaled chol decomp of Sigma
  matrix[L,L] cholCovTau;                         // scaled chol decomp of Tau
  matrix[J,S] lambdaeta;                          // Weighted random effects
  matrix[N,S] termeta;                            // Term containing random effects
  matrix[N,S] Fix1_wide;                          // Fixed effects contributions (wide form)
  vector[N] Fix2;
  matrix[N,S] Fix2_wide;
  matrix[N,S] exp_y;

  // priors for dispersion matrices
  scaleSigma ~ cauchy(0, 5);
  scaleTau ~ cauchy(0, 5);
  cholCorrSigma ~ lkj_corr_cholesky(2.0);
  cholCorrTau ~ lkj_corr_cholesky(2.0);
  cholCovSigma <- diag_matrix(scaleSigma) * cholCorrSigma;
  cholCovTau <- diag_matrix(scaleTau) * cholCorrTau;

  // other priors
  mu ~ normal(0, 10);
  gamma ~ normal(0, 10);
  lambda_2_1 ~ normal(0, 1);
  lambda_3_1 ~ normal(0, 1);
  lambda_5_2 ~ normal(0, 1);
  lambda_6_2 ~ normal(0, 1);

  // linear predictor
  Fix2 <- X * gamma; // linear predictor containing the intercept
  for (n in 1:N) 
    for (s in 1:S) {
      Fix1_wide[n,s] <- mu[s];
      Fix2_wide[n,s] <- Fix2[n];
  }
  
  // latent-trait effects (wide form)
  for (j in 1:J) 
    eta_wide[j] ~ multi_normal_cholesky(rep_vector(0, L), cholCovTau);

  // weighted latent-trait effects (wide form)
  for (j in 1:J)
    lambdaeta[j] <- to_row_vector(Lambda * to_vector(eta_wide[j]));

  // classroom-level term (wide form)
  for (n in 1:N) 
    for (s in 1:S) 
      termeta[n,s] <- lambdaeta[jj[n],s]; 

  // expected value
  exp_y <- Fix1_wide + Fix2_wide + termeta;
      
  // likelihood
  for (n in 1:N)
    y[n] ~ multi_normal_cholesky(exp_y[n], cholCovSigma);
}

generated quantities{
  cov_matrix[S] Sigma;                            // Residual vcov matrix
  cov_matrix[L] Tau;                              // Factor vcov matrix

  Sigma <- quad_form_diag(tcrossprod(cholCorrSigma), scaleSigma);
  Tau <- quad_form_diag(tcrossprod(cholCorrTau), scaleTau);
}
