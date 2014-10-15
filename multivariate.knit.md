---
title: "Multivariate Model"
date: "14 October, 2014"
output: 
  html_document:
    toc: true
---


# Preamble
All necessary files are sourced here.  Not shown.




# Basic Info
The following will show a simple multilevel model to serve as a starting point to a multivariate setting. The other file can be found here ADD FILE LOCATION HERE.

# Generate Data
First we generate the data given Andres' function which is sourced in the preamble.  In the following we will have 3 scores (S) per student, of which there are 50 students (n) in each of 10 classrooms (J). $$\tau$$ regards the covariance matrix for the random effects, while $$\sigma$$ regards the residual covariance matrix.



```r
S = 3
n = 50
J = 10
Tau = matrix(c(1, 0, 0, 
               0, 1, 0,
               0, 0, 1), nrow = 3, byrow = TRUE)

Sigma = matrix(c(1, 0, 0,  
                 0, 2^2, 0,
                 0, 0, 3^2), nrow = 3, byrow = TRUE)

Mu = c(70, 80, 90)

schooldat = dataGenHMLM2(S=S, n=n, J=J, mu=Mu,   
                         Tau = Tau, 
                         Sigma =  Sigma)
str(schooldat)
```

```
## 'data.frame':	1500 obs. of  12 variables:
##  $ S     : int  1 2 3 1 2 3 1 2 3 1 ...
##  $ n     : int  1 1 1 2 2 2 3 3 3 4 ...
##  $ J     : int  1 1 1 1 1 1 1 1 1 1 ...
##  $ idL2  : int  1 1 1 2 2 2 3 3 3 4 ...
##  $ idL3  : int  1 1 1 1 1 1 1 1 1 1 ...
##  $ mu    : num  70 80 90 70 80 90 70 80 90 70 ...
##  $ e     : num  -0.5147 -0.9018 -0.4725 -0.0302 0.3661 ...
##  $ y     : num  67.1 80.2 88.7 67.6 81.4 ...
##  $ beta_j: num  67.6 81.1 89.2 67.6 81.1 ...
##  $ ind1  : num  1 0 0 1 0 0 1 0 0 1 ...
##  $ ind2  : num  0 1 0 0 1 0 0 1 0 0 ...
##  $ ind3  : num  0 0 1 0 0 1 0 0 1 0 ...
##  - attr(*, "out.attrs")=List of 2
##   ..$ dim     : int  3 50 10
##   ..$ dimnames:List of 3
##   .. ..$ Var1: chr  "Var1=1" "Var1=2" "Var1=3"
##   .. ..$ Var2: chr  "Var2= 1" "Var2= 2" "Var2= 3" "Var2= 4" ...
##   .. ..$ Var3: chr  "Var3= 1" "Var3= 2" "Var3= 3" "Var3= 4" ...
```

## Reshape for input to Stan model
We're going to put the data in wide format, as this will likely make it easier, and probably quicker, to process in Stan.


```r
library(reshape2)
schoolSlim = schooldat[,c('S','n','J', 'y')]
schoolWide = dcast(schoolSlim, J+n~S)
head(schoolWide)
```

```
##   J n     1     2     3
## 1 1 1 67.09 80.16 88.74
## 2 1 2 67.57 81.43 91.99
## 3 1 3 67.02 79.57 89.31
## 4 1 4 67.31 81.26 93.70
## 5 1 5 67.29 79.78 92.68
## 6 1 6 67.89 79.37 92.97
```

```r
psych::describe(schoolWide)
```

```
##   vars   n  mean     sd median trimmed    mad   min   max  range     skew kurtosis      se
## J    1 500  5.50  2.875   5.50    5.50  3.706  1.00 10.00  9.000  0.00000 -1.23134 0.12858
## n    2 500 25.50 14.445  25.50   25.50 18.532  1.00 50.00 49.000  0.00000 -1.20815 0.64601
## 1    3 500 70.06  1.386  70.18   70.15  1.284 64.79 73.45  8.656 -0.61843  0.47467 0.06198
## 2    4 500 79.93  2.038  79.87   79.89  2.063 74.23 86.87 12.641  0.21668  0.05466 0.09115
## 3    5 500 90.04  3.209  89.85   90.00  3.198 79.89 98.74 18.854  0.06452 -0.25917 0.14350
```

# Stan code for univariate model

First we create the model in Stan. 

## Model code

```r
stanmodelcode = '
data {
  int<lower=1> N;                             // number of students per class * number of classes
  int<lower=1> J;                             // number of classes
  int<lower=1> S;                             // number of scores
  matrix<lower=0>[N, S] y;                    // Response: test scores
  int<lower=1,upper=J> classroom[N];          // student classroom
}

parameters {
  row_vector[S] Intercept;                    // score means
  cholesky_factor_corr[S] cholTau;            // chol decomp of corr matrix for RE
  cholesky_factor_corr[S]  cholSigma;         // chol decomp of residual matrix
  vector<lower=0>[S] scaleTau;                // scale for Tau
  vector<lower=0>[S] scaleSigma;              // scale for Sigma
  matrix[J,S] beta;                           // classroom effects
}

transformed parameters {

}

model {
  matrix[S,S] cholmatTau;                     // scaled chol decomp of Tau
  matrix[S,S] cholmatSigma;                   // scaled chol decomp of Sigma
  matrix[N,S] yhat;                           // Linear predictor
  vector[S] zerovec;                          // mean of RE

  zerovec <- rep_vector(0, S);
  
  // priors
  Intercept ~ normal(80, 10);                 
  cholTau ~ lkj_corr_cholesky(2.0);
  cholSigma ~ lkj_corr_cholesky(2.0);
  scaleTau ~ cauchy(0, 2.5);
  scaleSigma ~ cauchy(0, 2.5);

  // model calculations

  cholmatTau <- diag_matrix(scaleTau) * cholTau;
  cholmatSigma <- diag_matrix(scaleSigma) * cholSigma;


  for (j in 1:J){
    beta[j] ~ multi_normal_cholesky(zerovec, cholmatTau);
  }


  // likelihood
  for(n in 1:N){
    yhat[n] <- Intercept + beta[classroom[n]];
    y[n] ~ multi_normal_cholesky(yhat[n], cholmatSigma);
  }
}

generated quantities{
  matrix[S,S] Tau;
  matrix[S,S] Sigma;

  Tau <- tcrossprod(cholTau);
  Sigma <- tcrossprod(cholSigma);
}
'
```



## Test run
Next comes a test run.  This is simply to check for compilation, typos, and if there are coding issues that might result in slow convergence.  The run is kept small, and while we glance at the output, it really is of not too much interest at this point.

### Create Stan data list

First we create the data list containing everything Stan needs to run.


```r
stanTest = list(J=J, N=n*J, y=schoolWide[,c('1','2','3')], classroom=schoolWide$J, S=S)
```

Next comes a test run.  This is simply to check for compilation, typos, and if there are coding issues that might result in slow convergence.  The run is kept small, and while we glance at the output, it really is of not too much interest at this point.

### Run the model


```r
testiter = 2000
testwu = 1000
testthin = 10
testchains = 2


library(rstan)
fitTest = stan(model_code = stanmodelcode, model_name = "test",
               data = stanTest, iter = testiter, warmup=testwu, thin=testthin, 
               chains = testchains, verbose = F, refresh=100)
```

```
## 
## TRANSLATING MODEL 'test' FROM Stan CODE TO C++ CODE NOW.
## COMPILING THE C++ CODE FOR MODEL 'test' NOW.
## cygwin warning:
##   MS-DOS style path detected: C:/PROGRA~1/R/R-31~1.1/etc/x64/Makeconf
##   Preferred POSIX equivalent is: /cygdrive/c/PROGRA~1/R/R-31~1.1/etc/x64/Makeconf
##   CYGWIN environment variable option "nodosfilewarning" turns off this warning.
##   Consult the user's guide for more details about POSIX paths:
##     http://cygwin.com/cygwin-ug-net/using.html#using-pathnames
## In file included from C:/Program Files/R/R-3.1.1/library/rstan/include/rstan/rstaninc.hpp:3:0,
##                  from file312020f06068.cpp:724:
## C:/Program Files/R/R-3.1.1/library/rstan/include/rstan/stan_fit.hpp: In function 'int rstan::{anonymous}::sampler_command(rstan::stan_args&, Model&, Rcpp::List&, const std::vector<long long unsigned int>&, const std::vector<std::basic_string<char> >&, RNG_t&) [with Model = model31205e561e2c_test_namespace::model31205e561e2c_test, RNG_t = boost::random::additive_combine_engine<boost::random::linear_congruential_engine<unsigned int, 40014u, 0u, 2147483563u>, boost::random::linear_congruential_engine<unsigned int, 40692u, 0u, 2147483399u> >, Rcpp::List = Rcpp::Vector<19>]':
## C:/Program Files/R/R-3.1.1/library/rstan/include/rstan/stan_fit.hpp:1357:7:   instantiated from 'SEXPREC* rstan::stan_fit<Model, RNG_t>::call_sampler(SEXP) [with Model = model31205e561e2c_test_namespace::model31205e561e2c_test, RNG_t = boost::random::additive_combine_engine<boost::random::linear_congruential_engine<unsigned int, 40014u, 0u, 2147483563u>, boost::random::linear_congruential_engine<unsigned int, 40692u, 0u, 2147483399u> >, SEXP = SEXPREC*]'
## file312020f06068.cpp:735:116:   instantiated from here
## C:/Program Files/R/R-3.1.1/library/rstan/include/rstan/stan_fit.hpp:778:15: warning: unused variable 'return_code' [-Wunused-variable]
## C:/Program Files/R/R-3.1.1/library/rstan/include/rstan/stan_fit.hpp:1357:7:   instantiated from 'SEXPREC* rstan::stan_fit<Model, RNG_t>::call_sampler(SEXP) [with Model = model31205e561e2c_test_namespace::model31205e561e2c_test, RNG_t = boost::random::additive_combine_engine<boost::random::linear_congruential_engine<unsigned int, 40014u, 0u, 2147483563u>, boost::random::linear_congruential_engine<unsigned int, 40692u, 0u, 2147483399u> >, SEXP = SEXPREC*]'
## file312020f06068.cpp:735:116:   instantiated from here
## C:/Program Files/R/R-3.1.1/library/rstan/include/rstan/stan_fit.hpp:842:15: warning: unused variable 'return_code' [-Wunused-variable]
## In file included from C:/Program Files/R/R-3.1.1/library/rstan/include//stansrc/stan/prob/distributions/multivariate/continuous.hpp:14:0,
##                  from C:/Program Files/R/R-3.1.1/library/rstan/include//stansrc/stan/prob/distributions/multivariate.hpp:4,
##                  from C:/Program Files/R/R-3.1.1/library/rstan/include//stansrc/stan/prob/distributions.hpp:5,
##                  from C:/Program Files/R/R-3.1.1/library/rstan/include//stansrc/stan/model/model_header.hpp:37,
##                  from file312020f06068.cpp:8:
## C:/Program Files/R/R-3.1.1/library/rstan/include//stansrc/stan/prob/distributions/multivariate/continuous/lkj_corr.hpp: In function 'typename boost::math::tools::promote_args<T_lambda, T_cut>::type stan::prob::lkj_corr_cholesky_log(const Eigen::Matrix<T_covar, -0x00000000000000001, -0x00000000000000001>&, const T_shape&) [with bool propto = true, T_covar = stan::agrad::var, T_shape = double, typename boost::math::tools::promote_args<T_lambda, T_cut>::type = stan::agrad::var]':
## file312020f06068.cpp:305:13:   instantiated from 'T__ model31205e561e2c_test_namespace::model31205e561e2c_test::log_prob(std::vector<T__>&, std::vector<int>&, std::ostream*) const [with bool propto__ = true, bool jacobian__ = true, T__ = stan::agrad::var, std::ostream = std::basic_ostream<char>]'
## C:/Program Files/R/R-3.1.1/library/rstan/include//stansrc/stan/model/util.hpp:100:70:   instantiated from 'double stan::model::log_prob_grad(const M&, std::vector<double>&, std::vector<int>&, std::vector<double>&, std::ostream*) [with bool propto = true, bool jacobian_adjust_transform = true, M = model31205e561e2c_test_namespace::model31205e561e2c_test, std::ostream = std::basic_ostream<char>]'
## C:/Program Files/R/R-3.1.1/library/rstan/include/rstan/stan_fit.hpp:1325:9:   instantiated from 'SEXPREC* rstan::stan_fit<Model, RNG_t>::grad_log_prob(SEXP, SEXP) [with Model = model31205e561e2c_test_namespace::model31205e561e2c_test, RNG_t = boost::random::additive_combine_engine<boost::random::linear_congruential_engine<unsigned int, 40014u, 0u, 2147483563u>, boost::random::linear_congruential_engine<unsigned int, 40692u, 0u, 2147483399u> >, SEXP = SEXPREC*]'
## file312020f06068.cpp:751:116:   instantiated from here
## C:/Program Files/R/R-3.1.1/library/rstan/include//stansrc/stan/prob/distributions/multivariate/continuous/lkj_corr.hpp:77:25: warning: comparison between signed and unsigned integer expressions [-Wsign-compare]
## C:/Program Files/R/R-3.1.1/library/rstan/include//stansrc/stan/prob/distributions/multivariate/continuous/lkj_corr.hpp: In function 'T_shape stan::prob::do_lkj_constant(const T_shape&, const unsigned int&) [with T_shape = double]':
## C:/Program Files/R/R-3.1.1/library/rstan/include//stansrc/stan/prob/distributions/multivariate/continuous/lkj_corr.hpp:71:9:   instantiated from 'typename boost::math::tools::promote_args<T_lambda, T_cut>::type stan::prob::lkj_corr_cholesky_log(const Eigen::Matrix<T_covar, -0x00000000000000001, -0x00000000000000001>&, const T_shape&) [with bool propto = true, T_covar = stan::agrad::var, T_shape = double, typename boost::math::tools::promote_args<T_lambda, T_cut>::type = stan::agrad::var]'
## file312020f06068.cpp:305:13:   instantiated from 'T__ model31205e561e2c_test_namespace::model31205e561e2c_test::log_prob(std::vector<T__>&, std::vector<int>&, std::ostream*) const [with bool propto__ = true, bool jacobian__ = true, T__ = stan::agrad::var, std::ostream = std::basic_ostream<char>]'
## C:/Program Files/R/R-3.1.1/library/rstan/include//stansrc/stan/model/util.hpp:100:70:   instantiated from 'double stan::model::log_prob_grad(const M&, std::vector<double>&, std::vector<int>&, std::vector<double>&, std::ostream*) [with bool propto = true, bool jacobian_adjust_transform = true, M = model31205e561e2c_test_namespace::model31205e561e2c_test, std::ostream = std::basic_ostream<char>]'
## C:/Program Files/R/R-3.1.1/library/rstan/include/rstan/stan_fit.hpp:1325:9:   instantiated from 'SEXPREC* rstan::stan_fit<Model, RNG_t>::grad_log_prob(SEXP, SEXP) [with Model = model31205e561e2c_test_namespace::model31205e561e2c_test, RNG_t = boost::random::additive_combine_engine<boost::random::linear_congruential_engine<unsigned int, 40014u, 0u, 2147483563u>, boost::random::linear_congruential_engine<unsigned int, 40692u, 0u, 2147483399u> >, SEXP = SEXPREC*]'
## file312020f06068.cpp:751:116:   instantiated from here
## C:/Program Files/R/R-3.1.1/library/rstan/include//stansrc/stan/prob/distributions/multivariate/continuous/lkj_corr.hpp:26:24: warning: comparison between signed and unsigned integer expressions [-Wsign-compare]
## C:/Program Files/R/R-3.1.1/library/rstan/include//stansrc/stan/prob/distributions/multivariate/continuous/lkj_corr.hpp:71:9:   instantiated from 'typename boost::math::tools::promote_args<T_lambda, T_cut>::type stan::prob::lkj_corr_cholesky_log(const Eigen::Matrix<T_covar, -0x00000000000000001, -0x00000000000000001>&, const T_shape&) [with bool propto = true, T_covar = stan::agrad::var, T_shape = double, typename boost::math::tools::promote_args<T_lambda, T_cut>::type = stan::agrad::var]'
## file312020f06068.cpp:305:13:   instantiated from 'T__ model31205e561e2c_test_namespace::model31205e561e2c_test::log_prob(std::vector<T__>&, std::vector<int>&, std::ostream*) const [with bool propto__ = true, bool jacobian__ = true, T__ = stan::agrad::var, std::ostream = std::basic_ostream<char>]'
## C:/Program Files/R/R-3.1.1/library/rstan/include//stansrc/stan/model/util.hpp:100:70:   instantiated from 'double stan::model::log_prob_grad(const M&, std::vector<double>&, std::vector<int>&, std::vector<double>&, std::ostream*) [with bool propto = true, bool jacobian_adjust_transform = true, M = model31205e561e2c_test_namespace::model31205e561e2c_test, std::ostream = std::basic_ostream<char>]'
## C:/Program Files/R/R-3.1.1/library/rstan/include/rstan/stan_fit.hpp:1325:9:   instantiated from 'SEXPREC* rstan::stan_fit<Model, RNG_t>::grad_log_prob(SEXP, SEXP) [with Model = model31205e561e2c_test_namespace::model31205e561e2c_test, RNG_t = boost::random::additive_combine_engine<boost::random::linear_congruential_engine<unsigned int, 40014u, 0u, 2147483563u>, boost::random::linear_congruential_engine<unsigned int, 40692u, 0u, 2147483399u> >, SEXP = SEXPREC*]'
## file312020f06068.cpp:751:116:   instantiated from here
## C:/Program Files/R/R-3.1.1/library/rstan/include//stansrc/stan/prob/distributions/multivariate/continuous/lkj_corr.hpp:37:25: warning: comparison between signed and unsigned integer expressions [-Wsign-compare]
## C:/Program Files/R/R-3.1.1/library/rstan/include//stansrc/stan/prob/distributions/multivariate/continuous/lkj_corr.hpp: In function 'typename boost::math::tools::promote_args<T_lambda, T_cut>::type stan::prob::lkj_corr_cholesky_log(const Eigen::Matrix<T_covar, -0x00000000000000001, -0x00000000000000001>&, const T_shape&) [with bool propto = false, T_covar = double, T_shape = double, typename boost::math::tools::promote_args<T_lambda, T_cut>::type = double]':
## file312020f06068.cpp:305:13:   instantiated from 'T__ model31205e561e2c_test_namespace::model31205e561e2c_test::log_prob(std::vector<T__>&, std::vector<int>&, std::ostream*) const [with bool propto__ = false, bool jacobian__ = true, T__ = double, std::ostream = std::basic_ostream<char>]'
## C:/Program Files/R/R-3.1.1/library/rstan/include//stansrc/stan/model/util.hpp:228:61:   instantiated from 'void stan::model::finite_diff_grad(const M&, std::vector<double>&, std::vector<int>&, std::vector<double>&, double, std::ostream*) [with bool propto = false, bool jacobian_adjust_transform = true, M = model31205e561e2c_test_namespace::model31205e561e2c_test, std::ostream = std::basic_ostream<char>]'
## C:/Program Files/R/R-3.1.1/library/rstan/include//stansrc/stan/model/util.hpp:281:7:   instantiated from 'int stan::model::test_gradients(const M&, std::vector<double>&, std::vector<int>&, double, double, std::ostream&, std::ostream*) [with bool propto = true, bool jacobian_adjust_transform = true, M = model31205e561e2c_test_namespace::model31205e561e2c_test, std::ostream = std::basic_ostream<char>]'
## C:/Program Files/R/R-3.1.1/library/rstan/include/rstan/stan_fit.hpp:703:111:   instantiated from 'int rstan::{anonymous}::sampler_command(rstan::stan_args&, Model&, Rcpp::List&, const std::vector<long long unsigned int>&, const std::vector<std::basic_string<char> >&, RNG_t&) [with Model = model31205e561e2c_test_namespace::model31205e561e2c_test, RNG_t = boost::random::additive_combine_engine<boost::random::linear_congruential_engine<unsigned int, 40014u, 0u, 2147483563u>, boost::random::linear_congruential_engine<unsigned int, 40692u, 0u, 2147483399u> >, Rcpp::List = Rcpp::Vector<19>]'
## C:/Program Files/R/R-3.1.1/library/rstan/include/rstan/stan_fit.hpp:1357:7:   instantiated from 'SEXPREC* rstan::stan_fit<Model, RNG_t>::call_sampler(SEXP) [with Model = model31205e561e2c_test_namespace::model31205e561e2c_test, RNG_t = boost::random::additive_combine_engine<boost::random::linear_congruential_engine<unsigned int, 40014u, 0u, 2147483563u>, boost::random::linear_congruential_engine<unsigned int, 40692u, 0u, 2147483399u> >, SEXP = SEXPREC*]'
## file312020f06068.cpp:735:116:   instantiated from here
## C:/Program Files/R/R-3.1.1/library/rstan/include//stansrc/stan/prob/distributions/multivariate/continuous/lkj_corr.hpp:77:25: warning: comparison between signed and unsigned integer expressions [-Wsign-compare]
## C:/Program Files/R/R-3.1.1/library/rstan/include//stansrc/stan/agrad/rev/var_stack.hpp: At global scope:
## C:/Program Files/R/R-3.1.1/library/rstan/include//stansrc/stan/agrad/rev/var_stack.hpp:49:17: warning: 'void stan::agrad::free_memory()' defined but not used [-Wunused-function]
## 
## SAMPLING FOR MODEL 'test' NOW (CHAIN 1).
## 
## Iteration:    1 / 2000 [  0%]  (Warmup)
## Iteration:  100 / 2000 [  5%]  (Warmup)
## Iteration:  200 / 2000 [ 10%]  (Warmup)
## Iteration:  300 / 2000 [ 15%]  (Warmup)
## Iteration:  400 / 2000 [ 20%]  (Warmup)
## Iteration:  500 / 2000 [ 25%]  (Warmup)
## Iteration:  600 / 2000 [ 30%]  (Warmup)
## Iteration:  700 / 2000 [ 35%]  (Warmup)
## Iteration:  800 / 2000 [ 40%]  (Warmup)
## Iteration:  900 / 2000 [ 45%]  (Warmup)
## Iteration: 1000 / 2000 [ 50%]  (Warmup)
## Iteration: 1001 / 2000 [ 50%]  (Sampling)
## Iteration: 1100 / 2000 [ 55%]  (Sampling)
## Iteration: 1200 / 2000 [ 60%]  (Sampling)
## Iteration: 1300 / 2000 [ 65%]  (Sampling)
## Iteration: 1400 / 2000 [ 70%]  (Sampling)
## Iteration: 1500 / 2000 [ 75%]  (Sampling)
## Iteration: 1600 / 2000 [ 80%]  (Sampling)
## Iteration: 1700 / 2000 [ 85%]  (Sampling)
## Iteration: 1800 / 2000 [ 90%]  (Sampling)
## Iteration: 1900 / 2000 [ 95%]  (Sampling)
## Iteration: 2000 / 2000 [100%]  (Sampling)
## #  Elapsed Time: 138.494 seconds (Warm-up)
## #                36.199 seconds (Sampling)
## #                174.693 seconds (Total)
## 
## 
## SAMPLING FOR MODEL 'test' NOW (CHAIN 2).
## 
## Iteration:    1 / 2000 [  0%]  (Warmup)
## Iteration:  100 / 2000 [  5%]  (Warmup)
## Iteration:  200 / 2000 [ 10%]  (Warmup)
## Iteration:  300 / 2000 [ 15%]  (Warmup)
## Iteration:  400 / 2000 [ 20%]  (Warmup)
## Iteration:  500 / 2000 [ 25%]  (Warmup)
## Iteration:  600 / 2000 [ 30%]  (Warmup)
## Iteration:  700 / 2000 [ 35%]  (Warmup)
## Iteration:  800 / 2000 [ 40%]  (Warmup)
## Iteration:  900 / 2000 [ 45%]  (Warmup)
## Iteration: 1000 / 2000 [ 50%]  (Warmup)
## Iteration: 1001 / 2000 [ 50%]  (Sampling)
## Iteration: 1100 / 2000 [ 55%]  (Sampling)
## Iteration: 1200 / 2000 [ 60%]  (Sampling)
## Iteration: 1300 / 2000 [ 65%]  (Sampling)
## Iteration: 1400 / 2000 [ 70%]  (Sampling)
## Iteration: 1500 / 2000 [ 75%]  (Sampling)
## Iteration: 1600 / 2000 [ 80%]  (Sampling)
## Iteration: 1700 / 2000 [ 85%]  (Sampling)
## Iteration: 1800 / 2000 [ 90%]  (Sampling)
## Iteration: 1900 / 2000 [ 95%]  (Sampling)
## Iteration: 2000 / 2000 [100%]  (Sampling)
## #  Elapsed Time: 137.051 seconds (Warm-up)
## #                32.115 seconds (Sampling)
## #                169.166 seconds (Total)
```

```r
### Summarize
print(fitTest, digits_summary=3, pars=c('Intercept','beta','Tau', 'Sigma', 'scaleTau', 'scaleSigma'),
      probs = c(.025, .5, .975))
```

```
## Inference for Stan model: test.
## 2 chains, each with iter=2000; warmup=1000; thin=10; 
## post-warmup draws per chain=100, total post-warmup draws=200.
## 
##                 mean se_mean    sd   2.5%    50%  97.5% n_eff  Rhat
## Intercept[1]  70.431   0.031 0.322 69.806 70.417 71.076   105 1.001
## Intercept[2]  79.979   0.023 0.308 79.428 79.980 80.536   185 1.001
## Intercept[3]  89.440   0.030 0.372 88.689 89.447 90.241   151 0.994
## beta[1,1]      0.656   0.036 0.361 -0.087  0.667  1.311   101 1.017
## beta[1,2]      0.240   0.030 0.389 -0.453  0.286  0.981   166 0.994
## beta[1,3]      0.351   0.037 0.505 -0.491  0.369  1.361   183 0.996
## beta[2,1]     -0.079   0.035 0.344 -0.726 -0.095  0.564    95 1.010
## beta[2,2]      0.319   0.029 0.386 -0.365  0.284  1.043   180 1.004
## beta[2,3]     -0.872   0.044 0.507 -1.848 -0.819 -0.142   135 0.995
## beta[3,1]     -0.196   0.033 0.346 -0.842 -0.191  0.455   111 1.001
## beta[3,2]     -0.160   0.030 0.382 -0.812 -0.165  0.611   158 1.004
## beta[3,3]      0.740   0.039 0.546 -0.244  0.665  1.718   197 0.995
## beta[4,1]      0.602   0.032 0.362 -0.128  0.623  1.329   127 1.003
## beta[4,2]     -0.331   0.029 0.409 -1.199 -0.358  0.494   200 0.999
## beta[4,3]     -1.387   0.038 0.517 -2.315 -1.337 -0.440   181 0.996
## beta[5,1]      0.654   0.036 0.352 -0.002  0.700  1.382    98 1.002
## beta[5,2]     -1.091   0.029 0.407 -1.824 -1.088 -0.328   200 0.994
## beta[5,3]      0.859   0.038 0.497 -0.199  0.943  1.728   167 1.000
## beta[6,1]     -0.920   0.032 0.361 -1.651 -0.908 -0.178   127 0.995
## beta[6,2]      0.115   0.029 0.409 -0.759  0.112  0.859   200 1.001
## beta[6,3]      0.430   0.036 0.506 -0.423  0.429  1.450   196 0.997
## beta[7,1]      1.061   0.032 0.349  0.338  1.062  1.788   121 1.001
## beta[7,2]      0.669   0.031 0.401 -0.174  0.641  1.422   164 1.005
## beta[7,3]     -0.084   0.041 0.518 -0.948 -0.086  0.903   163 1.004
## beta[8,1]     -0.237   0.030 0.337 -0.909 -0.212  0.372   130 1.001
## beta[8,2]      1.284   0.027 0.381  0.594  1.287  2.022   200 1.007
## beta[8,3]     -0.400   0.037 0.523 -1.432 -0.402  0.650   200 0.992
## beta[9,1]     -1.435   0.034 0.365 -2.272 -1.418 -0.715   116 0.998
## beta[9,2]     -1.110   0.029 0.390 -1.881 -1.116 -0.352   177 0.997
## beta[9,3]      0.686   0.037 0.521 -0.263  0.643  1.685   200 0.994
## beta[10,1]    -0.274   0.032 0.338 -0.990 -0.261  0.319   113 1.008
## beta[10,2]     0.307   0.030 0.420 -0.514  0.328  1.050   200 1.001
## beta[10,3]     0.086   0.032 0.458 -0.616  0.119  0.969   200 1.002
## Tau[1,1]       1.000   0.000 0.000  1.000  1.000  1.000   200   NaN
## Tau[1,2]       0.086   0.019 0.273 -0.440  0.104  0.566   200 0.992
## Tau[1,3]      -0.136   0.023 0.297 -0.709 -0.143  0.416   165 1.008
## Tau[2,1]       0.086   0.019 0.273 -0.440  0.104  0.566   200 0.992
## Tau[2,2]       1.000   0.000 0.000  1.000  1.000  1.000   200 0.990
## Tau[2,3]      -0.205   0.026 0.293 -0.730 -0.219  0.338   127 1.002
## Tau[3,1]      -0.136   0.023 0.297 -0.709 -0.143  0.416   165 1.008
## Tau[3,2]      -0.205   0.026 0.293 -0.730 -0.219  0.338   127 1.002
## Tau[3,3]       1.000   0.000 0.000  1.000  1.000  1.000   200 0.990
## Sigma[1,1]     1.000   0.000 0.000  1.000  1.000  1.000   200   NaN
## Sigma[1,2]     0.054   0.003 0.043 -0.027  0.051  0.139   200 0.997
## Sigma[1,3]    -0.040   0.003 0.046 -0.144 -0.038  0.035   200 0.999
## Sigma[2,1]     0.054   0.003 0.043 -0.027  0.051  0.139   200 0.997
## Sigma[2,2]     1.000   0.000 0.000  1.000  1.000  1.000   136 0.990
## Sigma[2,3]     0.020   0.003 0.042 -0.069  0.021  0.094   200 0.991
## Sigma[3,1]    -0.040   0.003 0.046 -0.144 -0.038  0.035   200 0.999
## Sigma[3,2]     0.020   0.003 0.042 -0.069  0.021  0.094   200 0.991
## Sigma[3,3]     1.000   0.000 0.000  1.000  1.000  1.000   153 0.990
## scaleTau[1]    0.969   0.019 0.270  0.609  0.916  1.567   200 1.000
## scaleTau[2]    0.935   0.021 0.285  0.485  0.875  1.631   185 1.001
## scaleTau[3]    0.994   0.024 0.330  0.459  0.941  1.751   183 1.000
## scaleSigma[1]  1.037   0.002 0.033  0.967  1.037  1.107   200 0.991
## scaleSigma[2]  2.027   0.004 0.063  1.913  2.019  2.154   200 0.997
## scaleSigma[3]  2.981   0.007 0.092  2.822  2.978  3.161   184 0.993
## 
## Samples were drawn using NUTS(diag_e) at Tue Oct 14 16:11:57 2014.
## For each parameter, n_eff is a crude measure of effective sample size,
## and Rhat is the potential scale reduction factor on split chains (at 
## convergence, Rhat=1).
```

```r
# pairs(fitTest, pars=c('Intercept','mu','Tau', 'Sigma', 'scaleTau', 'scaleSigma'))
traceplot(fitTest)
```

<img src="./multivariate_files/figure-html/stanTest1.png" title="plot of chunk stanTest" alt="plot of chunk stanTest" width="672" /><img src="./multivariate_files/figure-html/stanTest2.png" title="plot of chunk stanTest" alt="plot of chunk stanTest" width="672" /><img src="./multivariate_files/figure-html/stanTest3.png" title="plot of chunk stanTest" alt="plot of chunk stanTest" width="672" /><img src="./multivariate_files/figure-html/stanTest4.png" title="plot of chunk stanTest" alt="plot of chunk stanTest" width="672" /><img src="./multivariate_files/figure-html/stanTest5.png" title="plot of chunk stanTest" alt="plot of chunk stanTest" width="672" /><img src="./multivariate_files/figure-html/stanTest6.png" title="plot of chunk stanTest" alt="plot of chunk stanTest" width="672" /><img src="./multivariate_files/figure-html/stanTest7.png" title="plot of chunk stanTest" alt="plot of chunk stanTest" width="672" /><img src="./multivariate_files/figure-html/stanTest8.png" title="plot of chunk stanTest" alt="plot of chunk stanTest" width="672" /><img src="./multivariate_files/figure-html/stanTest9.png" title="plot of chunk stanTest" alt="plot of chunk stanTest" width="672" /><img src="./multivariate_files/figure-html/stanTest10.png" title="plot of chunk stanTest" alt="plot of chunk stanTest" width="672" />

```r
library(lme4)
lme4res = sapply(schoolWide[,c('1','2','3')], function(y) lmer(y~1|J, data=schoolWide))
sapply(lme4res, summary, simplify=F)
```

```
## $`1`
## Linear mixed model fit by REML ['lmerMod']
## Formula: y ~ 1 | J
##    Data: schoolWide
## 
## REML criterion at convergence: 1410
## 
## Scaled residuals: 
##     Min      1Q  Median      3Q     Max 
## -2.7699 -0.6857  0.0014  0.6502  2.9436 
## 
## Random effects:
##  Groups   Name        Variance Std.Dev.
##  J        (Intercept) 1.127    1.062   
##  Residual             0.905    0.951   
## Number of obs: 500, groups:  J, 10
## 
## Fixed effects:
##             Estimate Std. Error t value
## (Intercept)   70.056      0.338     207
## 
## $`2`
## Linear mixed model fit by REML ['lmerMod']
## Formula: y ~ 1 | J
##    Data: schoolWide
## 
## REML criterion at convergence: 2062
## 
## Scaled residuals: 
##    Min     1Q Median     3Q    Max 
## -2.688 -0.657  0.018  0.614  3.309 
## 
## Random effects:
##  Groups   Name        Variance Std.Dev.
##  J        (Intercept) 0.792    0.89    
##  Residual             3.440    1.85    
## Number of obs: 500, groups:  J, 10
## 
## Fixed effects:
##             Estimate Std. Error t value
## (Intercept)   79.935      0.293     272
## 
## $`3`
## Linear mixed model fit by REML ['lmerMod']
## Formula: y ~ 1 | J
##    Data: schoolWide
## 
## REML criterion at convergence: 2579
## 
## Scaled residuals: 
##    Min     1Q Median     3Q    Max 
## -3.180 -0.693 -0.056  0.676  2.806 
## 
## Random effects:
##  Groups   Name        Variance Std.Dev.
##  J        (Intercept) 0.375    0.612   
##  Residual             9.958    3.156   
## Number of obs: 500, groups:  J, 10
## 
## Fixed effects:
##             Estimate Std. Error t value
## (Intercept)    90.04       0.24     376
```

```r
sapply(lme4res, ranef)
```

```
## $`1.J`
##    (Intercept)
## 1     -2.62860
## 2      0.79878
## 3     -0.32441
## 4      0.23636
## 5      0.85511
## 6     -0.41025
## 7     -0.20807
## 8      0.79029
## 9      0.05277
## 10     0.83801
## 
## $`2.J`
##    (Intercept)
## 1       0.8469
## 2       1.3619
## 3       0.7585
## 4      -0.5515
## 5      -0.7316
## 6      -0.7185
## 7      -0.1603
## 8       0.7989
## 9      -1.0307
## 10     -0.5737
## 
## $`3.J`
##    (Intercept)
## 1      -0.1485
## 2      -0.1139
## 3      -0.7493
## 4       0.6941
## 5       0.2165
## 6       0.7592
## 7      -0.1474
## 8       0.3420
## 9      -0.4517
## 10     -0.4010
```

# Main model

Now we are ready for the main run, bumping up the iterations etc.


```r
###############################
### A parallelized approach ###
###############################
library(rstan)
iters = 22000
wu = 2000
thin = 20
chains = 4


library(parallel)
cl = makeCluster(chains)
clusterEvalQ(cl, library(rstan))
```

```
## [[1]]
##  [1] "rstan"     "inline"    "Rcpp"      "graphics"  "grDevices" "utils"     "datasets"  "methods"   "stats"     "base"     
## 
## [[2]]
##  [1] "rstan"     "inline"    "Rcpp"      "graphics"  "grDevices" "utils"     "datasets"  "methods"   "stats"     "base"     
## 
## [[3]]
##  [1] "rstan"     "inline"    "Rcpp"      "graphics"  "grDevices" "utils"     "datasets"  "methods"   "stats"     "base"     
## 
## [[4]]
##  [1] "rstan"     "inline"    "Rcpp"      "graphics"  "grDevices" "utils"     "datasets"  "methods"   "stats"     "base"
```

```r
clusterExport(cl, c('stanmodelcode', 'stanTest', 'fitTest', 'iters', 'wu', 'thin'))
p = proc.time()
parfit = parSapply(cl, 1:chains, function(i) stan(model_code = stanmodelcode, model_name = "schools", 
                                             fit = fitTest, data = stanTest, iter = iters, 
                                             warmup=wu, thin=thin, chains = 1, chain_id=i,
                                             verbose = T, refresh=4000),
                   simplify=F)
proc.time() - p
```

```
##    user  system elapsed 
##    0.98    0.12  911.23
```

```r
stopCluster(cl)

# combine the chains
fitMain = sflist2stanfit(parfit)




print(fitMain, pars= c('Intercept','beta','Tau', 'Sigma', 'scaleTau', 'scaleSigma'), digits=3,
      probs = c(.025, .5, 0.975))
```

```
## Inference for Stan model: test.
## 4 chains, each with iter=22000; warmup=2000; thin=20; 
## post-warmup draws per chain=1000, total post-warmup draws=4000.
## 
##                 mean se_mean    sd   2.5%    50%  97.5% n_eff  Rhat
## Intercept[1]  70.415   0.005 0.326 69.766 70.418 71.054  4000 1.000
## Intercept[2]  80.014   0.005 0.324 79.388 80.007 80.668  3769 1.002
## Intercept[3]  89.473   0.006 0.362 88.762 89.470 90.193  3907 1.000
## beta[1,1]      0.677   0.006 0.352 -0.003  0.667  1.387  4000 1.001
## beta[1,2]      0.215   0.007 0.405 -0.576  0.214  0.995  3820 1.000
## beta[1,3]      0.347   0.008 0.491 -0.584  0.333  1.342  3896 1.000
## beta[2,1]     -0.065   0.006 0.353 -0.781 -0.057  0.646  4000 1.000
## beta[2,2]      0.293   0.007 0.403 -0.517  0.292  1.106  3622 1.000
## beta[2,3]     -0.950   0.008 0.506 -1.966 -0.942  0.013  4000 1.000
## beta[3,1]     -0.181   0.006 0.350 -0.865 -0.181  0.510  3563 1.001
## beta[3,2]     -0.209   0.006 0.395 -1.016 -0.211  0.556  3992 1.000
## beta[3,3]      0.712   0.008 0.501 -0.222  0.704  1.740  4000 1.000
## beta[4,1]      0.605   0.006 0.354 -0.074  0.606  1.314  3713 1.001
## beta[4,2]     -0.375   0.007 0.414 -1.204 -0.366  0.440  3578 1.000
## beta[4,3]     -1.417   0.009 0.528 -2.511 -1.401 -0.447  3571 1.000
## beta[5,1]      0.670   0.006 0.350 -0.034  0.670  1.368  3887 1.000
## beta[5,2]     -1.087   0.007 0.409 -1.927 -1.074 -0.323  3820 1.001
## beta[5,3]      0.822   0.008 0.513 -0.150  0.808  1.859  4000 1.000
## beta[6,1]     -0.896   0.006 0.353 -1.619 -0.888 -0.197  3833 1.000
## beta[6,2]      0.081   0.007 0.405 -0.712  0.078  0.893  3564 1.001
## beta[6,3]      0.401   0.008 0.505 -0.592  0.390  1.396  4000 1.000
## beta[7,1]      1.082   0.006 0.352  0.402  1.077  1.800  3800 1.000
## beta[7,2]      0.649   0.007 0.414 -0.146  0.642  1.490  3839 1.000
## beta[7,3]     -0.091   0.008 0.499 -1.079 -0.094  0.923  4000 1.000
## beta[8,1]     -0.220   0.006 0.353 -0.892 -0.223  0.476  3909 1.000
## beta[8,2]      1.233   0.007 0.413  0.462  1.217  2.075  3758 1.001
## beta[8,3]     -0.387   0.008 0.497 -1.347 -0.378  0.570  3748 1.000
## beta[9,1]     -1.432   0.006 0.349 -2.136 -1.433 -0.723  3890 1.000
## beta[9,2]     -1.138   0.007 0.415 -1.952 -1.127 -0.348  3735 1.000
## beta[9,3]      0.651   0.008 0.503 -0.276  0.639  1.659  3858 1.001
## beta[10,1]    -0.253   0.006 0.346 -0.926 -0.254  0.414  3947 1.000
## beta[10,2]     0.275   0.007 0.404 -0.530  0.278  1.070  3610 1.001
## beta[10,3]     0.090   0.008 0.483 -0.889  0.086  1.022  4000 1.000
## Tau[1,1]       1.000   0.000 0.000  1.000  1.000  1.000  4000   NaN
## Tau[1,2]       0.118   0.005 0.286 -0.444  0.129  0.639  3886 0.999
## Tau[1,3]      -0.143   0.005 0.287 -0.660 -0.155  0.435  4000 0.999
## Tau[2,1]       0.118   0.005 0.286 -0.444  0.129  0.639  3886 0.999
## Tau[2,2]       1.000   0.000 0.000  1.000  1.000  1.000  4000 0.999
## Tau[2,3]      -0.208   0.005 0.290 -0.716 -0.230  0.391  3444 1.001
## Tau[3,1]      -0.143   0.005 0.287 -0.660 -0.155  0.435  4000 0.999
## Tau[3,2]      -0.208   0.005 0.290 -0.716 -0.230  0.391  3444 1.001
## Tau[3,3]       1.000   0.000 0.000  1.000  1.000  1.000  3702 0.999
## Sigma[1,1]     1.000   0.000 0.000  1.000  1.000  1.000  4000   NaN
## Sigma[1,2]     0.053   0.001 0.044 -0.035  0.053  0.138  3871 1.000
## Sigma[1,3]    -0.042   0.001 0.045 -0.133 -0.043  0.046  4000 1.000
## Sigma[2,1]     0.053   0.001 0.044 -0.035  0.053  0.138  3871 1.000
## Sigma[2,2]     1.000   0.000 0.000  1.000  1.000  1.000  4000 0.999
## Sigma[2,3]     0.018   0.001 0.046 -0.069  0.017  0.109  4000 1.001
## Sigma[3,1]    -0.042   0.001 0.045 -0.133 -0.043  0.046  4000 1.000
## Sigma[3,2]     0.018   0.001 0.046 -0.069  0.017  0.109  4000 1.001
## Sigma[3,3]     1.000   0.000 0.000  1.000  1.000  1.000  4000 0.999
## scaleTau[1]    0.969   0.005 0.288  0.581  0.912  1.673  3902 1.000
## scaleTau[2]    0.944   0.005 0.295  0.530  0.888  1.678  4000 0.999
## scaleTau[3]    1.003   0.005 0.343  0.516  0.941  1.793  4000 1.001
## scaleSigma[1]  1.042   0.001 0.033  0.981  1.042  1.108  4000 0.999
## scaleSigma[2]  2.027   0.001 0.065  1.902  2.027  2.158  3761 1.000
## scaleSigma[3]  2.983   0.002 0.096  2.805  2.979  3.180  3801 1.000
## 
## Samples were drawn using NUTS(diag_e) at Tue Oct 14 16:27:23 2014.
## For each parameter, n_eff is a crude measure of effective sample size,
## and Rhat is the potential scale reduction factor on split chains (at 
## convergence, Rhat=1).
```

```r
# examine some diagnostics
traceplot(fitMain, inc_warmup=F, pars=c('Intercept','beta','Tau', 'Sigma', 'scaleTau', 'scaleSigma'))
```

<img src="./multivariate_files/figure-html/stanMain11.png" title="plot of chunk stanMain1" alt="plot of chunk stanMain1" width="672" /><img src="./multivariate_files/figure-html/stanMain12.png" title="plot of chunk stanMain1" alt="plot of chunk stanMain1" width="672" /><img src="./multivariate_files/figure-html/stanMain13.png" title="plot of chunk stanMain1" alt="plot of chunk stanMain1" width="672" /><img src="./multivariate_files/figure-html/stanMain14.png" title="plot of chunk stanMain1" alt="plot of chunk stanMain1" width="672" /><img src="./multivariate_files/figure-html/stanMain15.png" title="plot of chunk stanMain1" alt="plot of chunk stanMain1" width="672" /><img src="./multivariate_files/figure-html/stanMain16.png" title="plot of chunk stanMain1" alt="plot of chunk stanMain1" width="672" /><img src="./multivariate_files/figure-html/stanMain17.png" title="plot of chunk stanMain1" alt="plot of chunk stanMain1" width="672" /><img src="./multivariate_files/figure-html/stanMain18.png" title="plot of chunk stanMain1" alt="plot of chunk stanMain1" width="672" />

```r
ainfo = get_adaptation_info(fitMain)
cat(ainfo[[1]])
```

```
## # Adaptation terminated
## # Step size = 0.148772
## # Diagonal elements of inverse mass matrix:
## # 0.101106, 0.0986579, 0.126008, 0.0821886, 0.101408, 0.115149, 0.00206179, 0.0022309, 0.00198141, 0.0726442, 0.0823817, 0.114313, 0.00106019, 0.000991059, 0.00114354, 0.11025, 0.117964, 0.116092, 0.121526, 0.116083, 0.118631, 0.115018, 0.120601, 0.117942, 0.122953, 0.165004, 0.161527, 0.149456, 0.154747, 0.166748, 0.150305, 0.15319, 0.159029, 0.152396, 0.141798, 0.26906, 0.245645, 0.242615, 0.254508, 0.267365, 0.246899, 0.239426, 0.223611, 0.265146, 0.236116
```

```r
samplerpar = get_sampler_params(fitMain)[[1]]
summary(samplerpar)
```

```
##  accept_stat__      stepsize__      treedepth__     n_leapfrog__    n_divergent__  
##  Min.   :0.0009   Min.   :0.0001   Min.   : 1.00   Min.   :   1.0   Min.   :0e+00  
##  1st Qu.:0.8640   1st Qu.:0.1488   1st Qu.: 4.00   1st Qu.:  15.0   1st Qu.:0e+00  
##  Median :0.9440   Median :0.1488   Median : 4.00   Median :  15.0   Median :0e+00  
##  Mean   :0.9006   Mean   :0.1521   Mean   : 4.52   Mean   :  25.8   Mean   :9e-04  
##  3rd Qu.:0.9829   3rd Qu.:0.1488   3rd Qu.: 5.00   3rd Qu.:  31.0   3rd Qu.:0e+00  
##  Max.   :1.0000   Max.   :0.4659   Max.   :11.00   Max.   :2047.0   Max.   :1e+00
```


# Use coda for additional inspection
## Density and Trace












