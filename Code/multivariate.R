

### Relies on the following files
R.utils::sourceDirectory('Code/Functions')
# load()



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




library(reshape2)
schoolSlim = schooldat[,c('S','n','J', 'y')]
schoolWide = dcast(schoolSlim, J+n~S)
head(schoolWide)
psych::describe(schoolWide)



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



stanTest = list(J=J, N=n*J, y=schoolWide[,c('1','2','3')], classroom=schoolWide$J, S=S)



testiter = 2000
testwu = 1000
testthin = 10
testchains = 2

library(rstan)
suppressMessages({
fitTest = stan(model_code = stanmodelcode, model_name = "test",
               data = stanTest, iter = testiter, warmup=testwu, thin=testthin, 
               chains = testchains, verbose = F, refresh=100)
})

### Summarize
print(fitTest, digits_summary=3, pars=c('Intercept','beta','Tau', 'Sigma', 'scaleTau', 'scaleSigma'),
      probs = c(.025, .5, .975))

# pairs(fitTest, pars=c('Intercept','mu','Tau', 'Sigma', 'scaleTau', 'scaleSigma'))
traceplot(fitTest)

library(lme4)
lme4res = sapply(schoolWide[,c('1','2','3')], function(y) lmer(y~1|J, data=schoolWide))
sapply(lme4res, summary, simplify=F)
sapply(lme4res, ranef)



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
clusterExport(cl, c('stanmodelcode', 'stanTest', 'fitTest', 'iters', 'wu', 'thin'))
p = proc.time()
parfit = parSapply(cl, 1:chains, function(i) stan(model_code = stanmodelcode, model_name = "schools", 
                                             fit = fitTest, data = stanTest, iter = iters, 
                                             warmup=wu, thin=thin, chains = 1, chain_id=i,
                                             verbose = T, refresh=4000),
                   simplify=F)
proc.time() - p
stopCluster(cl)

# combine the chains
fitMain = sflist2stanfit(parfit)




print(fitMain, pars= c('Intercept','beta','Tau', 'Sigma', 'scaleTau', 'scaleSigma'), digits=3,
      probs = c(.025, .5, 0.975))

# examine some diagnostics
traceplot(fitMain, inc_warmup=F, pars=c('Intercept','beta','Tau', 'Sigma', 'scaleTau', 'scaleSigma'))
ainfo = get_adaptation_info(fitMain)
cat(ainfo[[1]])
samplerpar = get_sampler_params(fitMain)[[1]]
summary(samplerpar)



library(coda)
Int = as.mcmc(extract(fitMain, par='Intercept')$Intercept)
plot(Int)

betas = extract(fitMain, par='beta')$beta # array
str(betas)
plot(as.mcmc(betas[,,1]))



acf(Int)



R.utils::sourceDirectory('Code/Functions')
library(lme4)
ranefs = sapply(schoolWide[,c('1','2','3')], function(y) ranef(lmer(y~1|J, data=schoolWide))$J$'(Intercept)')
cor(ranefs) # spurious?

STest = 3
nTest = 50
JTest = 500
TauTest = matrix(c(1, 0, 0, 
                   0, 1, 0,
                   0, 0, 1), nrow = 3, byrow = TRUE)

SigmaTest = matrix(c(1, 0, 0,  
                     0, 2^2, 0,
                     0, 0, 3^2), nrow = 3, byrow = TRUE)

MuTest = c(70, 80, 90)

schooldatTest = dataGenHMLM2(S=STest, n=nTest, J=JTest, mu=MuTest,   
                             Tau = TauTest, 
                             Sigma =  SigmaTest)
str(schooldatTest)

library(reshape2)
schoolSlimTest = schooldatTest[,c('S','n','J', 'y')]
schoolWideTest = dcast(schoolSlimTest, J+n~S)

ranefs = sapply(schoolWideTest[,c('1','2','3')], 
                function(y) ranef(lmer(y~1|J, data=schoolWideTest))$J$'(Intercept)')
cor(ranefs)



R.utils::sourceDirectory('Code/Functions')
examineCor = function(J){
  R.utils::sourceDirectory('Code/Functions')
  # J is vector of J sizes
  STest = 3
  nTest = 50
  JTest = 500
  TauTest = matrix(c(1, 0, 0, 
                     0, 1, 0,
                     0, 0, 1), nrow = 3, byrow = TRUE)
  
  SigmaTest = matrix(c(1, 0, 0,  
                       0, 2^2, 0,
                       0, 0, 3^2), nrow = 3, byrow = TRUE)
  
  MuTest = c(70, 80, 90)
  
  schooldatList = sapply(J, function(j) dataGenHMLM2(S=STest, n=nTest, J=j, mu=MuTest,
                                                     Tau = TauTest, Sigma =  SigmaTest), 
                         simplify=F)
  
  require(reshape2)
  schoolSlimList = lapply(schooldatList, function(data) data[,c('S','n','J', 'y')])
  schoolWideList = suppressMessages({ lapply(schoolSlimList, function(data) dcast(data, J+n~S)) })

  require(lme4)
  ranefList = lapply(schoolWideList, function(data) 
    sapply(data[,c('1','2','3')], function(y) ranef(lmer(y~1|J, data=data))$J$'(Intercept)')
    )


  corList = lapply(ranefList, cor)
  meancor = sapply(corList, function(cormat) mean(cormat[lower.tri(cormat)]))
  
  out = list(corList, meancor)
  out
}
  
# debugonce(examineCor)
J = c(10, 25, 50, seq(100, 1000, 100))
test = examineCor(J) 
plot(J, test[[2]], ylab='Mean off-diagonal correlation')




# save.image()



### Subsequent files that rely on this one if any
# X.R
# Y.R

