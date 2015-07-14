# Analysis of HMLM2F simulated data using stan

library(rstan)

# Working directory, if applicable:
# setwd(".//code")

# Source data-generating and other related functions
source("Code/Functions/HMLM2F_fun2.R")

# Create loadings matrix
Lambda <- simpleLambda(nunk=c(2,2), lambda=c(.8,.5,.7,.5))

# Generate data 
simData <- HMLM2F.datGen(S=6, L=2, n=20, J=50, 
                         mu = c(0, -1, 1, -2, 2, 3),
                         Lambda = Lambda,
                         Tau = matrix(c(1, 0,
                                        0, 2), nrow = 2, byrow = TRUE),
                         Sigma = matrix(c(1, 0, 0, 0, 0, 0,  
                                          0, 2, 0, 0, 0, 0,
                                          0, 0, 3, 0, 0, 0,
                                          0, 0, 0, 3, 0, 0,
                                          0, 0, 0, 0, 2, 0, 
                                          0, 0, 0, 0, 0, 1), nrow = 6, byrow = TRUE) 
)

test = reshape2::dcast(simData, n+J~S, value.var = 'y')
colnames(test)[3:8] = paste0('y', 1:6)
head(test)


##################################################
# Create stan data list:
# Data generated in long format. Need to reshape wide by outcome, long by student-school.
library(reshape2)
simDatLong = simData[,c('S','n','J','idL2', 'y')]
head(simDatLong)
simDatWide = reshape2::dcast(simDatLong, J+n+idL2~S, value.var='y')
head(simDatWide)
# length(unique(simDatWide$n))
# length(unique(simDatWide$J))
# length(unique(simDatWide$idL2))
# psych::describe(simDatWide)

stanData = list(N=max(unique(simDatWide$idL2)), 
                J=max(unique(simDatWide$J)),
                S=max(unique(simDatLong$S)),
                L=ncol(Lambda),
                jj=simDatWide$J, 
                y=simDatWide[,c('1','2','3','4','5','6')])

##################################################
# stan model 
source("Code/Stan Models/HMLM2F_stanModel_S6L2v1.R")

##################################################
# Analysis 

# Test run (for compilation)
stanFit = stan(model_code = stanModel, data = stanData, 
               iter = 1000, warmup = 200, thin = 2, chains = 1, 
               model_name = "HMLM2F", 
               # fit = stanFit,
               verbose = F)

print(stanFit, digits_summary=2, probs = c(.025, .5, .975), pars = c('lambda_2_1', 'lambda_3_1',
                                                                     'lambda_5_2','lambda_6_2','mu', 
                                                                     'Sigma', 'Tau'))

# traceplot(stanFit, inc_warmup=F, pars = c('lambda_2_1', 'lambda_3_1',
#                                           'lambda_5_2','lambda_6_2','mu', 
#                                           'Sigma', 'Tau'))



library(parallel)
cl = makeCluster(4)
nchain = 4
clusterExport(cl, c('stanFit', 'stanModel', 'stanData', 'nchain'))
clusterEvalQ(cl, library(rstan))

p = proc.time()
stanFitMain = parSapply(cl, 1:nchain, function(chain) stan(model_code = stanModel, data = stanData, 
                                                           iter = 12000, warmup = 2000, thin = 20, chains = 1, chain_id=chain,
                                                           model_name = "HMLM2F", fit = stanFit))
(proc.time() - p)[3]/60
stopCluster(cl)



fit = sflist2stanfit(stanFitMain)
save(fit, simData, file = 'Data/bigRun.RData')

print(fit,digits_summary=2, probs = c(.025, .5, .975), pars = c('lambda_2_1', 'lambda_3_1',
                                                                 'lambda_5_2','lambda_6_2','mu', 
                                                                 'Sigma', 'Tau'))
traceplot(fit, inc_warmup=F, pars = c('lambda_2_1', 'lambda_3_1',
                                      'lambda_5_2','lambda_6_2','mu', 
                                      'Sigma', 'Tau'))


# comparison
Tau = matrix(get_posterior_mean(fit, 'Tau')[,5], 2, 2, byrow = T)
Sigma =  matrix(get_posterior_mean(fit, 'Sigma')[,5], 6, 6, byrow = T)
lambda = get_posterior_mean(fit, pars=c('lambda_2_1', 'lambda_3_1',
                                        'lambda_5_2','lambda_6_2'))[,5]
etas = matrix(get_posterior_mean(fit, pars=c('eta'))[,5], ncol=2, byrow=T)



library(lavaan)
cfamod = "
  F1 =~ y1 + y2 + y3
  F2 =~ y4 + y5 + y6 
"
cfafit <- cfa(cfamod, data=test, meanstructure = F)
summary(cfafit, fit.measures=F)
Tau
Sigma
lambda
