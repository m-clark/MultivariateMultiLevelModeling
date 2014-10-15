# Data generating function for HMLM2: 
# 2-level, continuous multivariate outcome, unconditional, balanced, complete data
# For example, scores nested within students within classrooms  
dataGenHMLM2 = function(
  S, # number of potential scores for each unit
  n, # number of units (e.g., students)
  J, # number of clusters (e.g., classrooms)
  mu, # vector with S means
  Sigma, # S x S  dispersion matrix (across units)
  Tau # S x S  dispersion matrix (across clusters)
) {  
  stopifnot(S==length(mu), S==nrow(Sigma), S==ncol(Sigma), S==nrow(Tau), S==ncol(Tau))
  A1 = rep(1,n) %x% diag(S) # design matrix, classroom-level model 
  A2 = rep(1,n) %x% diag(S) # design matrix, classroom-level model 
  # Complete data assumed, so A1 and A2 need not be pre-multiplied by observed-scores indicator matrix. 
  dat <- expand.grid(1:S, 1:n, 1:J)
  colnames(dat) <- c("S", "n", "J")
  require(plyr)
  #dat$id1 <- as.integer(factor(paste(dat$J, dat$n))) # generate level-2 (student) ID
  dat$idL2 <- c(plyr::id(dat[c("J", "n")], drop = FALSE))
  dat$idL3 <- dat$J # generate level-3 (classroom) ID
  dat$mu <- c(rep(1,J) %x% (A1 %*% mu)) # create term containing mu
  require(mvtnorm)
  r.wide <- mvtnorm::rmvnorm(J, mean = rep(0, nrow(Tau)), sigma = Tau) # level-2 random effects
    # each row in r.wide represents a classroom
  term.r <- NULL
  for (j in 1:J) {
    r_j <- as.vector(t(r.wide[j,]))
    term.r <- rbind(term.r, A2 %*% r_j)
  } # each row of r.wide is vectorized, terms are then allocated using A2, results are stacked
  e.wide <- mvtnorm::rmvnorm(n*J, mean = c(rep(0,S)), sigma = Sigma) # level-1 random effects
    # Generating a unique error vector for each student in the study (all classrooms)
    # Each row in e.wide represents a student
  dat$e <- as.vector(t(e.wide)) # Stack level-1 random effects (as a multivariate outcome)
  dat$y <- c(dat$mu + term.r + dat$e)
  dat$beta_j <- c(dat$mu + term.r)
  for(S in 1:S) {
    dat[paste("ind",S,sep="")] <- ifelse(dat$S==S,1,0)
  } # Create outcome indicators 
  #dat <- dat[, c('S', 'n', 'J', 'y', 'e')] # re-order columns
  return(dat) 
}

# datHMLM2 <- dataGenHMLM2(S=3, n=20, J=50, mu = c(1000,3000,5000),
#   Tau = matrix(c(1, 0, 0, 
#                  0, 1, 0,
#                  0, 0, 1), nrow = 3, byrow = TRUE), 
#   Sigma = matrix(c(1, 0, 0,  
#                    0, 2^2, 0,
#                    0, 0, 3^2), nrow = 3, byrow = TRUE) 
# )

##################################################
# Save data for analysis using HMLM2
# library(foreign)
# # write.dta(datHMLM2, "C:/HLM/hmlm2ex/hmlm2sim01_L1.dta") 
# datHMLM2L2 <- aggregate(datHMLM2, by=list(datHMLM2$idL2), FUN=mean)
# # write.dta(datHMLM2L2, "C:/HLM/hmlm2ex/hmlm2sim01_L2.dta") 
# datHMLM2L3 <- aggregate(datHMLM2, by=list(datHMLM2$idL3), FUN=mean)
# # write.dta(datHMLM2L3, "C:/HLM/hmlm2ex/hmlm2sim01_L3.dta") 
# 
# ##################################################
# # To do:
# # Add ability to generate incomplete data.  
# 
# dataGenHMLM2(S=3, n=5, J=3, mu=rep(0,3), Sigma=diag(3), Tau=diag(3))