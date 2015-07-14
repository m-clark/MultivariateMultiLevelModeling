# Functions pertaining to a multivariate 2-level HLM with factor structure at level 2 (HMLM2F)
# For example, scores of students within classroom, with interest in classroom-level latent traits

####################################################################################################
####################################################################################################
####################################################################################################

# Function to create a simple structure factor loadings matrix
simpleLambda = function(
  nunk, # vector containing number of unknowns for each latent trait (ie, for each column)
  lambdas # vector containing values to be allocated (not including 1s)
) {  
  stopifnot(sum(nunk)==length(lambdas)) # check consistency
  nrow = sum(nunk) + length(nunk) # number of rows
  ncol = length(nunk) # number of columns (i.e., latent traits)
  LAM = matrix(0, nrow=nrow, ncol=ncol) # Grid for Lamnda Matrix
  nunkplus1 <- nunk + rep(1,length(nunk))
  for (l in 1:length(nunk)) {
    x <- cumsum(nunkplus1)[l] - nunkplus1[l] + 2
    counter <- 0
    LAM[x-1+counter, l] <- 1
    while (counter < nunk[l]) {
      LAM[x + counter, l] <- lambdas[x + counter - l]
      counter <- counter + 1
    }
  }
  return(LAM)
}

# debugonce(simpleLambda)
# simpleLambda(nrow=6, nunk=c(2,1,0), lambda=c(.1,.2,.3))
# simpleLambda(nrow=8, nunk=c(3,2,0), lambda=c(.5,.5,.5,.3,.3))

# Lambda = matrix(c(1, 0, 0,
#                   .1, 0, 0,
#                   .1, 0, 0,
#                   0, 1, 0,
#                   0, .1, 0,
#                   0, 0, 1), nrow = 6, byrow = TRUE), 

# To do:
# May want to remove nrow from the parameters requested as it can be obtained from the other paramters.

####################################################################################################
####################################################################################################
####################################################################################################

# Data generating function
# Known limitations of this function include: 
#   Continuous outcomes only
#   Unconditional (no covariates)
#   Balanced (complete data), thus not differentiating potential scores from actually observed
HMLM2F.datGen = function(
  S, # number of potential scores for each unit
  L, # number of latent traits
  n, # number of level 1 units (e.g., students)
  J, # number of level 2 units (e.g., classrooms)
  mu, # vector with S means
  Lambda, # S x L factor loadings matrix
  Tau, # L x L dispersion matrix (across clusters)
  Sigma # S x S dispersion matrix (across units)
) {  
  # Some checks of the incoming information:
  stopifnot(S==length(mu), S==nrow(Lambda), S==nrow(Sigma), S==ncol(Sigma))
  stopifnot(L==ncol(Lambda), L==nrow(Tau), L==ncol(Tau))
  # Create data grid:
  dat <- expand.grid(S = 1:S, n = 1:n, J = 1:J)
  # Add case IDs
  dat$idL2 <- c(plyr::id(dat[c("J", "n")], drop = FALSE)) # level-2 (student) ID
  dat$idL3 <- dat$J # level-3 (classroom) ID
  # Create design matrices for the classroom-level model:
  A1 = rep(1,n) %x% diag(S) # Design matrix for the fixed effects
  A2 = rep(1,n) %x% diag(S) # Design matrix for the random effects
    # Note %x% is Kronecker product
    # A1 and A2 are the same (in this case) given that the model has no covariates
    # A1 and A2 are not pre-multiplied by any observed-scores indicator matrix given that complete data are assumed
  # Add mean (mu):
  dat$mu <- c(rep(1,J) %x% (A1 %*% mu)) 
  # Add random effects:
  require(MASS)
  eta.wide <- mvrnorm(J, mu = rep(0, nrow(Tau)), Sigma = Tau, empirical=T) # level-2 random effects
    # Note each row in eta.wide represents a classroom (i.e., wide format)
  term.eta <- NULL
  for (j in 1:J) {
    eta_j <- as.vector(t(eta.wide[j,])) 
      # Each row of eta.wide is vectorized into eta_j (i.e., long format)
    term.eta <- rbind(term.eta, A2 %*% Lambda %*% eta_j)
      # For each j, terms of eta_j are allocated 
      # Results across all J are stacked
  } 
  e.wide <- mvrnorm(n*J, mu = c(rep(0,S)), Sigma = Sigma, empirical=T) # level-1 random effects
    # Generating a unique error (row) vector for each student 
    # Each row in e.wide represents a student (i.e., wide format)
  dat$e <- as.vector(t(e.wide)) # Stack and add level-1 random effects
  dat$y <- c(dat$mu + term.eta + dat$e) # Add outcome
  dat$exp.y <- c(dat$mu + term.eta) # Add expected value of outcome
  # Add outcome indicators:
#   for(m in 1:S) {
#     dat[paste("ind", m, sep="")] <- ifelse(dat$S==m, 1, 0)
#   } 
  # Add Lambda column vectors, each one repeatedly stacked as necessary
#   for(l in 1:ncol(Lambda)) {
#     dat[paste("lambda", l, sep="")] <- c(rep(1,J) %x% (A2 %*% Lambda[,l])) 
#   } 
  #dat <- dat[, c('S', 'n', 'J', 'y', 'mu', 'exp.y_j', 'e')] # re-order columns
  return(dat) 
}

# debugonce(HMLM2F.datGen)
# hmlm2f_dat <- HMLM2F.datGen(S=6, L=3, n=25, J=100, 
#                             mu = c(100, -300, 500, -700, 900, -1100),
#                             Lambda = matrix(c(1, 0, 0,
#                                               .1, 0, 0,
#                                               .1, 0, 0,
#                                               0, 1, 0,
#                                               0, .1, 0,
#                                               0, 0, 1), nrow = 6, byrow = TRUE), 
#                             Tau = matrix(c(100, 0, 0, 
#                                            0, 200, 0,
#                                            0, 0, 300), nrow = 3, byrow = TRUE), 
#                             Sigma = matrix(c(10, 0, 0, 0, 0, 0,  
#                                              0, 20, 0, 0, 0, 0,
#                                              0, 0, 30, 0, 0, 0,
#                                              0, 0, 0, 40, 0, 0,
#                                              0, 0, 0, 0, 50, 0, 
#                                              0, 0, 0, 0, 0, 60), nrow = 6, byrow = TRUE) 
# )

# To do:
# Add ability to generate incomplete data?
# Add checks for simple structure of Lambda?
# Generate based on unknown lambdas only? (All other lambda's are either 0 or 1). 

####################################################################################################
####################################################################################################
####################################################################################################

# Function to prepare data sets to be used by hlm::hmlm2.exe
HMLM2F.dat4HLM = function(hmlm2f_dat) {  
  library(foreign) 
  write.dta(hmlm2f_dat, "C:/HLM/hmlm2f_dat_L1.dta") 
  hmlm2f_dat_L2 <- aggregate(hmlm2f_dat, by=list(hmlm2f_dat$idL2), FUN=mean)
  hmlm2f_dat_L2$Group.1 <- NULL
  write.dta(hmlm2f_dat_L2, "C:/HLM/hmlm2f_dat_L2.dta") 
  hmlm2f_dat_L3 <- aggregate(hmlm2f_dat, by=list(hmlm2f_dat$idL3), FUN=mean)
  hmlm2f_dat_L3$Group.1 <- NULL
  hmlm2f_dat_L3$nPS <- NULL
  hmlm2f_dat_L3$n <- NULL
  hmlm2f_dat_L3$idL2 <- NULL
  write.dta(hmlm2f_dat_L3, "C:/HLM/hmlm2f_dat_L3.dta") 
}

