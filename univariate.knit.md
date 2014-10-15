---
title: "Univariate Model"
date: "13 October, 2014"
output: 
  html_document:
    toc: true
---




# Preamble
All necessary files are sourced here.  Not shown.




# Basic Info
The following will show a simple multilevel model to serve as a starting point to a multivariate setting. Comparisons of results will be made to output from the R pacakges <span style='color:blue'>_lme4_</span>.

# Generate Data
First we generate the data given Andres' function which is sourced in the preamble.  In the following we will have 3 scores (S) per student, of which there are 50 students (n) in each of 10 classrooms (J). $$\tau$$ regards the covariance matrix for the random effects, while $$\sigma$$ regards the residual covariance matrix.

Since in this example we are only dealing with the univariate setting, the data is then divided according to score of interest, each of which may hten serve as an example.


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
##  $ e     : num  0.909 1.288 4.332 -0.948 -0.856 ...
##  $ y     : num  69.8 80.2 94 68 78 ...
##  $ beta_j: num  68.9 78.9 89.6 68.9 78.9 ...
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

```r
# Create subsets
library(dplyr)
score1 = schooldat %>%
  filter(S==1)
score2 = schooldat %>%
  filter(S==2)
score3 = schooldat %>%
  filter(S==3)
```


# Stan code for univariate model

First we create the model in Stan.  One thing to note is the prior mean for the intercept is set to the overall mean, but otherwise we have a normal distribution for the intercept and random effect, and half-cauchy for the sd/variance parameters.

## Model code

```r
stanmodelcode = '
data {
  int<lower=1> N;                             // number of students per class * number of classes
  int<lower=1> J;                             // number of classes
  vector<lower=0>[N] y;                       // Response: test scores
  int<lower=1,upper=J> classroom[N];          // student classroom
}

parameters {
  real Intercept;                             // overall mean
  real<lower=0> sd_int;                       // sd for ints
  real<lower=0> sigma_y;                      // residual sd
  vector[J] mu;                               // classroom effects
}

transformed parameters {
  vector[N] yhat;                             // Linear predictor
  for (n in 1:N)
    yhat[n] <- mu[classroom[n]];
}

model {
  // priors
  Intercept ~ normal(80, 10);                 // example of weakly informative priors (and ignoring Matt trick for now);
  mu ~ normal(Intercept, sd_int);
  sd_int ~ cauchy(0, 2.5);
  sigma_y ~ cauchy(0, 2.5);

  // likelihood
  y ~ normal(yhat, sigma_y);
}

generated quantities{
  real<lower=0, upper=1>  ICC;
  
  ICC <- (sd_int^2)/(sd_int^2+sigma_y^2);
}
'
```



## Test run
Next comes a test run.  This is simply to check for compilation, typos, and if there are coding issues that might result in slow convergence.  The run is kept small, and while we glance at the output, it really is of not too much interest at this point.

### Create Stan data list

```r
score1StanTest = list(J=J, N=n*J, y=score1$y, classroom=score1$J)
```


```r
testiter = 2000
testwu = 1000
testthin = 10
testchains = 2


library(rstan)
fitTest = stan(model_code = stanmodelcode, model_name = "test",
               data = score1StanTest, iter = testiter, warmup=testwu, thin=testthin, 
               chains = testchains, verbose = F)
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
##                  from file31885f131391.cpp:544:
## C:/Program Files/R/R-3.1.1/library/rstan/include/rstan/stan_fit.hpp: In function 'int rstan::{anonymous}::sampler_command(rstan::stan_args&, Model&, Rcpp::List&, const std::vector<long long unsigned int>&, const std::vector<std::basic_string<char> >&, RNG_t&) [with Model = model318847c32880_test_namespace::model318847c32880_test, RNG_t = boost::random::additive_combine_engine<boost::random::linear_congruential_engine<unsigned int, 40014u, 0u, 2147483563u>, boost::random::linear_congruential_engine<unsigned int, 40692u, 0u, 2147483399u> >, Rcpp::List = Rcpp::Vector<19>]':
## C:/Program Files/R/R-3.1.1/library/rstan/include/rstan/stan_fit.hpp:1357:7:   instantiated from 'SEXPREC* rstan::stan_fit<Model, RNG_t>::call_sampler(SEXP) [with Model = model318847c32880_test_namespace::model318847c32880_test, RNG_t = boost::random::additive_combine_engine<boost::random::linear_congruential_engine<unsigned int, 40014u, 0u, 2147483563u>, boost::random::linear_congruential_engine<unsigned int, 40692u, 0u, 2147483399u> >, SEXP = SEXPREC*]'
## file31885f131391.cpp:555:116:   instantiated from here
## C:/Program Files/R/R-3.1.1/library/rstan/include/rstan/stan_fit.hpp:778:15: warning: unused variable 'return_code' [-Wunused-variable]
## C:/Program Files/R/R-3.1.1/library/rstan/include/rstan/stan_fit.hpp:1357:7:   instantiated from 'SEXPREC* rstan::stan_fit<Model, RNG_t>::call_sampler(SEXP) [with Model = model318847c32880_test_namespace::model318847c32880_test, RNG_t = boost::random::additive_combine_engine<boost::random::linear_congruential_engine<unsigned int, 40014u, 0u, 2147483563u>, boost::random::linear_congruential_engine<unsigned int, 40692u, 0u, 2147483399u> >, SEXP = SEXPREC*]'
## file31885f131391.cpp:555:116:   instantiated from here
## C:/Program Files/R/R-3.1.1/library/rstan/include/rstan/stan_fit.hpp:842:15: warning: unused variable 'return_code' [-Wunused-variable]
## C:/Program Files/R/R-3.1.1/library/rstan/include//stansrc/stan/agrad/rev/var_stack.hpp: At global scope:
## C:/Program Files/R/R-3.1.1/library/rstan/include//stansrc/stan/agrad/rev/var_stack.hpp:49:17: warning: 'void stan::agrad::free_memory()' defined but not used [-Wunused-function]
## 
## SAMPLING FOR MODEL 'test' NOW (CHAIN 1).
## 
## Iteration:    1 / 2000 [  0%]  (Warmup)
## Iteration:  200 / 2000 [ 10%]  (Warmup)
## Iteration:  400 / 2000 [ 20%]  (Warmup)
## Iteration:  600 / 2000 [ 30%]  (Warmup)
## Iteration:  800 / 2000 [ 40%]  (Warmup)
## Iteration: 1000 / 2000 [ 50%]  (Warmup)
## Iteration: 1001 / 2000 [ 50%]  (Sampling)
## Iteration: 1200 / 2000 [ 60%]  (Sampling)
## Iteration: 1400 / 2000 [ 70%]  (Sampling)
## Iteration: 1600 / 2000 [ 80%]  (Sampling)
## Iteration: 1800 / 2000 [ 90%]  (Sampling)
## Iteration: 2000 / 2000 [100%]  (Sampling)
## #  Elapsed Time: 0.309 seconds (Warm-up)
## #                0.184 seconds (Sampling)
## #                0.493 seconds (Total)
## 
## 
## SAMPLING FOR MODEL 'test' NOW (CHAIN 2).
## 
## Iteration:    1 / 2000 [  0%]  (Warmup)
## Iteration:  200 / 2000 [ 10%]  (Warmup)
## Iteration:  400 / 2000 [ 20%]  (Warmup)
## Iteration:  600 / 2000 [ 30%]  (Warmup)
## Iteration:  800 / 2000 [ 40%]  (Warmup)
## Iteration: 1000 / 2000 [ 50%]  (Warmup)
## Iteration: 1001 / 2000 [ 50%]  (Sampling)
## Iteration: 1200 / 2000 [ 60%]  (Sampling)
## Iteration: 1400 / 2000 [ 70%]  (Sampling)
## Iteration: 1600 / 2000 [ 80%]  (Sampling)
## Iteration: 1800 / 2000 [ 90%]  (Sampling)
## Iteration: 2000 / 2000 [100%]  (Sampling)
## #  Elapsed Time: 0.31 seconds (Warm-up)
## #                0.2 seconds (Sampling)
## #                0.51 seconds (Total)
```

```r
### Summarize
print(fitTest, digits_summary=3, pars=c('Intercept','mu','sigma_y', 'sd_int', 'ICC'),
      probs = c(.025, .5, .975))
```

```
## Inference for Stan model: test.
## 2 chains, each with iter=2000; warmup=1000; thin=10; 
## post-warmup draws per chain=100, total post-warmup draws=200.
## 
##             mean se_mean    sd   2.5%    50%  97.5% n_eff  Rhat
## Intercept 69.906   0.026 0.364 69.314 69.883 70.621   200 0.997
## mu[1]     68.803   0.011 0.146 68.528 68.782 69.118   166 0.992
## mu[2]     70.379   0.010 0.131 70.133 70.378 70.623   184 0.995
## mu[3]     71.782   0.010 0.148 71.536 71.761 72.067   200 1.001
## mu[4]     70.931   0.009 0.130 70.706 70.923 71.167   200 1.005
## mu[5]     69.796   0.010 0.133 69.515 69.797 70.030   175 0.992
## mu[6]     68.853   0.009 0.133 68.581 68.857 69.095   200 1.005
## mu[7]     69.242   0.010 0.135 68.986 69.238 69.490   200 1.009
## mu[8]     70.706   0.010 0.141 70.457 70.712 70.985   200 1.000
## mu[9]     69.701   0.009 0.134 69.467 69.704 69.947   200 1.009
## mu[10]    68.407   0.010 0.140 68.128 68.407 68.670   200 0.994
## sigma_y    0.989   0.002 0.033  0.923  0.986  1.051   200 1.007
## sd_int     1.181   0.022 0.299  0.777  1.103  1.912   180 1.001
## ICC        0.570   0.008 0.108  0.394  0.551  0.799   191 0.995
## 
## Samples were drawn using NUTS(diag_e) at Mon Oct 13 14:42:01 2014.
## For each parameter, n_eff is a crude measure of effective sample size,
## and Rhat is the potential scale reduction factor on split chains (at 
## convergence, Rhat=1).
```

```r
pairs(fitTest, pars=c('Intercept','sigma_y', 'sd_int', 'ICC'))
```

<img src="./univariate_files/figure-html/stanTest1.png" title="plot of chunk stanTest" alt="plot of chunk stanTest" width="672" />

```r
### Compare
library(lme4)
mod_lme = lmer(y~1+ (1|J), score1)
summary(mod_lme)
```

```
## Linear mixed model fit by REML ['lmerMod']
## Formula: y ~ 1 + (1 | J)
##    Data: score1
## 
## REML criterion at convergence: 1448
## 
## Scaled residuals: 
##     Min      1Q  Median      3Q     Max 
## -2.9508 -0.6226  0.0018  0.6530  2.8459 
## 
## Random effects:
##  Groups   Name        Variance Std.Dev.
##  J        (Intercept) 1.174    1.084   
##  Residual             0.979    0.989   
## Number of obs: 500, groups:  J, 10
## 
## Fixed effects:
##             Estimate Std. Error t value
## (Intercept)   69.854      0.346     202
```

```r
ranef(mod_lme)$J + fixef(mod_lme)
```

```
##    (Intercept)
## 1        68.78
## 2        70.39
## 3        71.76
## 4        70.91
## 5        69.80
## 6        68.85
## 7        69.25
## 8        70.70
## 9        69.70
## 10       68.41
```

```r
print(fitTest, digits_summary=3, pars='mu')
```

```
## Inference for Stan model: test.
## 2 chains, each with iter=2000; warmup=1000; thin=10; 
## post-warmup draws per chain=100, total post-warmup draws=200.
## 
##         mean se_mean    sd  2.5%   25%   50%   75% 97.5% n_eff  Rhat
## mu[1]  68.80   0.011 0.146 68.53 68.71 68.78 68.92 69.12   166 0.992
## mu[2]  70.38   0.010 0.131 70.13 70.29 70.38 70.47 70.62   184 0.995
## mu[3]  71.78   0.010 0.148 71.54 71.68 71.76 71.88 72.07   200 1.001
## mu[4]  70.93   0.009 0.130 70.71 70.83 70.92 71.02 71.17   200 1.005
## mu[5]  69.80   0.010 0.133 69.52 69.70 69.80 69.89 70.03   175 0.992
## mu[6]  68.85   0.009 0.133 68.58 68.77 68.86 68.94 69.09   200 1.005
## mu[7]  69.24   0.010 0.135 68.99 69.16 69.24 69.34 69.49   200 1.009
## mu[8]  70.71   0.010 0.141 70.46 70.60 70.71 70.80 70.98   200 1.000
## mu[9]  69.70   0.009 0.134 69.47 69.61 69.70 69.78 69.95   200 1.009
## mu[10] 68.41   0.010 0.140 68.13 68.31 68.41 68.50 68.67   200 0.994
## 
## Samples were drawn using NUTS(diag_e) at Mon Oct 13 14:42:01 2014.
## For each parameter, n_eff is a crude measure of effective sample size,
## and Rhat is the potential scale reduction factor on split chains (at 
## convergence, Rhat=1).
```

```r
### Diagnostic plots
traceplot(fitTest, pars=c('Intercept','mu','sigma_y', 'sd_int'))
```

<img src="./univariate_files/figure-html/stanTest2.png" title="plot of chunk stanTest" alt="plot of chunk stanTest" width="672" /><img src="./univariate_files/figure-html/stanTest3.png" title="plot of chunk stanTest" alt="plot of chunk stanTest" width="672" />

```r
traceplot(fitTest, pars=c('Intercept','mu','sigma_y', 'sd_int'), inc_warmup=F)
```

<img src="./univariate_files/figure-html/stanTest4.png" title="plot of chunk stanTest" alt="plot of chunk stanTest" width="672" /><img src="./univariate_files/figure-html/stanTest5.png" title="plot of chunk stanTest" alt="plot of chunk stanTest" width="672" />

Comparison to <span style='color:blue'>_lme4_</span> suggests the code is on the right track.

## Main run
Here we do the main run, bumping up the iterations and number of chains, though in parallel.













