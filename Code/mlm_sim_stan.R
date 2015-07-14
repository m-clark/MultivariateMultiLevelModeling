# Analysis of simulated mlm data using Stan 
# Multivariate outcome (nested within units)

rm(list = ls()) 

############################################################
# Data generating function for a multivariate linear model (mlm). 
# For example, multiple scores per student. 
# Known limitations of this function include: 
#   Continuous outcomes only 
#   No covariates
mlm_datGen = function(
  S, # number of potential scores for each unit
  n, # number of units (e.g., students)
  mu, # vector with S means
  Sigma,  # S x S dispersion matrix (across units)
  missingPrcnt = 0 # percent missing
) {  
  stopifnot(all(S == c(length(mu), dim(Sigma))))
  stopifnot(all(missingPrcnt >= 0, missingPrcnt <= 1))
  # Create data grid:
  dat <- expand.grid(S = 1:S, n = 1:n)
  # Intercept:
  dat$mu <- (rep(1,n) %x% diag(S)) %*% mu
  # Residuals
  # require(mvtnorm)
  e.wide <- mvtnorm::rmvnorm(n, mean = c(rep(0,S)), sigma = Sigma) 
  # Generating a unique error (row) vector for each student 
  # Each row in e.wide represents a student (i.e., wide format)
  dat$e <- as.vector(t(e.wide)) # Stack and add level-1 random effects
  dat$y <- dat$mu + dat$e # Add outcome
  # dat$mis <- ifelse(missingPrcnt > 0, rbinom(nrow(dat), 1, prob = missingPrcnt), 0)
  ifelse(missingPrcnt > 0, dat$mis <- rbinom(nrow(dat), 1, prob = missingPrcnt), dat$mis <- 0)
  dat$y_obs <- dat$y
  dat$y_obs[dat$mis == 1] <- NA
  # dat$obs <- 1; dat$obs[dat$mis == 1] <- 0
#   for (i in 1:nrow(dat)) 
#     if (dat$mis[i] == 1) dat$yobs[i] <- NA
  # Add outcome indicators
  for (d in unique(dat$S)) 
    dat[paste("ind", d, sep = "")] <- ifelse(dat$S == d, 1, 0)
  return(dat) 
}

############################################################
# Generate data and prepare data

True_mu <- seq(1:6)
True_Sigma <- diag(6)
# debugonce(mlm_datGen)
mlm <- mlm_datGen(S = 6, n = 100, mu = True_mu, Sigma = True_Sigma, missingPrcnt = 0.25)  
table(mlm$mis)

# Observed data, in database-like format:
mlm_obs <- na.omit(mlm[,c('S', 'n', 'y_obs', grep('ind', names(mlm), value=T))]) 

# Number of observed outcomes per student and indexes
n_obs_st <- aggregate(S ~ n, mlm_obs, length)[,"S"] 
end <- cumsum(n_obs_st)
strt <- end - n_obs_st + 1
obs_ns <- cbind(n_obs_st, strt, end)

# Stan data
stan_dat = list(
  n_stu = length(unique(mlm$n)), # number of students
  S = length(unique(mlm$S)), # number of potential outcomes per student
  n_obs_tot = nrow(mlm_obs), # number of observed outcomes
  y_obs = as.vector(mlm_obs[,c("y_obs")]), # observed data, long format
  n_obs_stu = obs_ns[,c('n_obs_st')], # number of observed outcomes per student
  idx_start = obs_ns[,c('strt')], # start index for observed outcomes per student
  idx_end = obs_ns[,c('end')], # end index for per student observed outcomes
  miss_ind = as.matrix(mlm_obs[,c(grep('ind', names(mlm), value=T))]) # stacked missing indicator matrices
)

############################################################
# Model and analysis
library(rstan)

# Test run, mainly to check compilation
stanFit = stan(
  file = 'code/mlm_Q06_v3.stan',
  data = stan_dat,
  iter = 20, warmup = 10, thin = 2, chains = 1,
  verbose = F
)
