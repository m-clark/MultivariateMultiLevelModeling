# Data generating function for a multivariate 2-level HLM with factor structure at level 2 (hlm2f). 
# For example, multiple scores per students within classrooms, with classroom-level latent traits. 
# Known limitations of this function include: 
#   Continuous outcomes only 
#   Limited to 2 student-level and to 2 classroom-level covariates 
hlm2f_datGen = function(
  S, # number of potential scores for each unit
  L, # number of latent traits
  n, # number of level 1 units (e.g., students)
  J, # number of level 2 units (e.g., classrooms)
  mu, # vector with S means
  feL2, # vector with student-level fixed effects
  feL3, # vector with classroom-level fixed effects
  Lambda, # S x L factor loadings matrix
  Tau, # L x L dispersion matrix (across clusters)
  Sigma # S x S dispersion matrix (across units)
) {  
  stopifnot(all(S == c(length(mu), nrow(Lambda), dim(Sigma))))
  stopifnot(all(L == c(ncol(Lambda), nrow(Tau), ncol(Tau))))
  # Create data grid:
  dat <- expand.grid(S = 1:S, n = 1:n, J = 1:J)
  # Assign unique IDs:
  dat$idL3 <- dat$J # level-3 (classroom) ID
  dat$idL2 <- c(plyr::id(dat[c("J", "n")], drop = FALSE)) # unique level-2 (student) ID
  # Fixed effects:
  # Intercept:
  dat$mu <- (rep(1,J*n) %x% diag(S)) %*% mu
  # Student-level covariates:
  DM_a <- diag(n*J) %x% rep(1,S) # design matrix
  dat$a1 <- DM_a %*% rbinom(n*J, 1, 0.5) # indicator, e.g., male, female
  dat$a2 <- DM_a %*% rnorm(n*J, 50, 10) # continuous measure
  # Classroom-level covariates:
  DM_w <- diag(J) %x% rep(1,n*S) # design matrix
  dat$w1 <- DM_w %*% rbinom(J, 1, 0.2) # indicator, e.g., urban, rural
  dat$w2 <- DM_w %*% rnorm(J, 10, 5) # continuous measure
  # Fixed portion:
  dat$fix <- dat$mu + (feL2[1]*dat$a1) + (feL2[2]*dat$a2) + (feL3[1]*dat$w1) + (feL3[2]*dat$w2)
  # Random effects:
  require(mvtnorm)
  # Classroom (level-2):
  eta_wide <- mvtnorm::rmvnorm(J, mean = rep(0, nrow(Tau)), sigma = Tau) 
  # Each row in eta_wide represents a classroom (i.e., wide format)
  term_eta <- c() # Initialization (required in the loop below)
  DM_re_stu <- rep(1,n) %x% diag(S) # Student-level design matrix for the random effects
  for (j in 1:J) {
    eta_j <- as.vector(t(eta_wide[j,])) 
    # Each row of eta_wide is put into a column vector called eta_j (one row per loop)
    term_eta <- rbind(term_eta, DM_re_stu %*% Lambda %*% eta_j)
    # Terms of eta_j are weighted and allocated to j
    # Results across all J are stacked
  } 
  # Residuals
  e.wide <- mvtnorm::rmvnorm(n*J, mean = c(rep(0,S)), sigma = Sigma) 
  # Generating a unique error (row) vector for each student 
  # Each row in e.wide represents a student (i.e., wide format)
  dat$e <- as.vector(t(e.wide)) # Stack and add level-1 random effects
  dat$y <- c(dat$fix + term_eta + dat$e) # Add outcome
  dat$exp.y <- c(dat$fix + term_eta) # Add expected value of outcome
  return(dat) 
}

# (Lambda <- matrix(c(1, 0.9, 0.8, 0, 0, 0, 0, 0, 1, 0.8), nrow = 5, byrow = FALSE))
# debugonce(hlm2f_datGen)
# hlm2f_dat <- hlm2f_datGen(S=5, L=2, n=10, J=40, 
#                           mu = seq(from=10, to=50, by=10),
#                           feL2 = c(0,0), 
#                           feL3 = c(0,0), 
#                           Lambda = Lambda,
#                           Tau = matrix(c(10^2, 0.5*10*15,
#                                          0.5*10*15, 15^2), nrow = 2, byrow = TRUE),
#                           Sigma = matrix(c(10^2, 0.8*10*20, 0.7*10*30, 0, 0,
#                                            0.8*10*20, 20^2, 0.6*20*30, 0, 0,
#                                            0.7*10*30, 0.6*20*30, 30^2, 0, 0,
#                                            0, 0, 0, 40^2, 0.8*40*50,
#                                            0, 0, 0, 0.8*40*50, 50^2), nrow = 5, byrow = TRUE)
#                           )
