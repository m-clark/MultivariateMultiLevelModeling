setwd("/afs/crc.nd.edu/user/a/amarti38/Private/Projects/AM/Diss")

##################################################
# Load an prepare data 
rm(list = ls())
RdataFile <- "code//hlm2f_Q6E01.Rdata"
load(file = RdataFile)
# load(file = "code//hlm2f_Q6E01_20150608.Rdata")

# Data generated in long format. Need to reshape wide by outcome, long by student.
library(reshape2)
datLong <- hlm2f
datWide <- reshape2::dcast(datLong, J + n + idL2 + idL3 + a1 + a2 + w1 + w2 ~ S, value.var = 'y')
stanDataWide = list(N = length(unique(datWide$idL2)), # number of sutdents
                    J = length(unique(datWide$J)), # number of classrooms
                    S = length(unique(datLong$S)), # number of outcomes per student
                    L = ncol(True_Lambda), # number of latent traits
                    P = ncol(datWide[,c("a1","a2","w1","w2")]), # number of covariates
                    jj = datWide$J,  # classroom identifier
                    X = datWide[,c("a1","a2","w1","w2")], # covariate values
                    y = datWide[,as.character(c(1:max(datLong$S)))]) # outcome

##################################################
# Stan models
stan_model_list <- c(
  "code/hlm2f_Q06wc_L02_22p01.stan", 
  "code/hlm2f_Q06wc_L02_22p02.stan", 
  "code/hlm2f_Q06wc_L02_22p03.stan", 
  "code/hlm2f_Q06wc_L02_33p01.stan", 
  "code/hlm2f_Q06wc_L02_55p01.stan", 
  "code/hlm2f_Q06wc_L02_66p01.stan", 
  "code/hlm2f_Q06wm_L02_22p01.stan" 
)

##################################################
# Function to fit a stan model in parallel
# Better to check model compilation prior to using this function
stan_parallel_fit <-
  function(stan_model, stan_data) {
    library(parallel)
    envir_all_models = new.env()
    stanData = stan_data
    stanModel = stan_model
    n_iter = 22000 # iterations
    n_warmup = 2000 # warmup
    n_thin = 20 # thinning
    n_par_runs = 4 # parallel runs
#     n_iter = 100 # iterations
#     n_warmup = 50 # warmup
#     n_thin = 5 # thinning
#     n_par_runs = 2 # parallel runs
    cl = makeCluster(n_par_runs)
    clusterEvalQ(cl, library(rstan))
    clusterExport(
      cl,
      c('stanData', 'stanModel', 'n_iter', 'n_warmup', 'n_thin', 'n_par_runs'),
      envir = envir_all_models
    )
    parFit = parSapply(cl, 1:n_par_runs,
                       function(chain)
                         stan(
                           data = stanData,
                           file = stanModel,
                           chains = 1, # chains (per parallel run)
                           iter = n_iter, 
                           warmup = n_warmup, 
                           thin = n_thin,
                           chain_id = chain,
                           verbose = TRUE
                         ),
                       simplify = FALSE)
    stopCluster(cl)
    return(sflist2stanfit(parFit))
  }

##################################################
# Fit a single model
library(rstan)
a_model <- stan_model_list[7]
time_start = proc.time()
stan_fit <- stan_parallel_fit(stan_model = a_model, stan_data = stanDataWide)
time_fit <- (proc.time() - time_start)[3]/60 # processing time in minutes
str(stan_fit, 2) # Extract first 2 levels of object
stan_fit@model_pars

save(stan_fit, time_fit, 
     file = paste(substring(RdataFile, 1, nchar(RdataFile) - 6),
                  "_",
                  substr(a_model, 15, nchar(a_model) - 5),
                  ".Rdata",
                  sep = ""
                  )
     )

