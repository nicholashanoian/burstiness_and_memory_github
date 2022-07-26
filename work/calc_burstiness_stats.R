library(fitdistrplus)
library(latex2exp)
library(Hmisc)
library(progress)
library(tidyverse)
library(doSNOW)
library(stringr)



read_repos_to_taus <- function(path, n=150039) {
    #' read in file with one line for each repo of the form
    #' repo_id:tau0 tau1 tau2 tau3

    update_every <- 100
    pb <- progress_bar$new(format="[:bar] :percent at :tick_rate it/s eta: :eta total :elapsed", total=n)
    names <- vector(mode="list", length=n)
    data <- vector(mode="list", length=n)
    
    i <- 0
    ## t0 <- as.numeric(Sys.time())
    con <- file(path, "r")
    while (TRUE) {
        # read line and update progress
        ## i <- i + 1

        i <- i + 1
        if (i %% update_every == 0) {
            pb$tick(update_every)            
        }


        line = readLines(con, n=1)

        # exit at end
        if (length(line) == 0) {
            break
        }

        # split on : to extract repo_id
        parts <- str_split_fixed(line, ":", n=2)

        repo_id <- parts[,1]
        
        taus <- as.numeric(strsplit(parts[,2], " ", fixed=TRUE)[[1]])
        
        # update list
        names[[i]] <- repo_id
        ## data[[i]] <- as.numeric(taus)
        data[[i]] <- taus
    }
    # return the data
    close(con)
    names(data) <- names
    data
}





fit_to_burstiness <- function(fit) {
    # Given a fit from fitdist from fitdistrplus, calculate burstiness
    # parameter B. Currently implemented for weibll and lognorm
    # distributions
    dist_mean <- -1
    dist_sd <- -1
    if (fit$distname == "weibull") {
        a <- fit$estimate["shape"]
        b <- fit$estimate["scale"]
        dist_mean <- b * gamma(1 + 1/a)
        dist_sd <- sqrt(b^2 * (gamma(1 + 2/a) - (gamma(1 + 1/a))^2))
    }

    else if (fit$distname == "lnorm") {
        mu <- fit$estimate["meanlog"]
        sigma <- fit$estimate["sdlog"]

        dist_mean <- exp(mu + 1/2 * sigma^2)
        dist_sd <- sqrt(exp(2*mu + sigma^2)*(exp(sigma^2) - 1))
    }

    B <- (dist_sd - dist_mean) / (dist_sd + dist_mean)
    unname(B)
}


memory_coef <- function(data) {
    
    x1 <- data[1:length(data) - 1]
    x2 <- data[2:length(data)]

    cor(x1, x2)
    
}

chunkify <- function(x, n) {
    per_chunk <- ceiling(length(x)/n)
    split(x, ceiling(seq_along(x) / per_chunk))
}

calculate_B_M <- function(data) {
    
    # log start time
    t0 <- Sys.time()
    cat(paste("Started at (", format(t0, "%X"), ")\n"))


    # setup cluster
    cores <- parallel::detectCores()
    cl <- makeSOCKcluster(cores)
    registerDoSNOW(cl)

    # split data into n_cores number of chunks
    chunks <- chunkify(names(data), n=cores)

    # run chunks in parallel
    results <- foreach(chunk=chunks, .packages=c("fitdistrplus"), .combine="rbind",.export=c("fit_to_burstiness", "memory_coef")) %dopar% {

        
        # for results of this chunk
        i <- 1
        best <- vector(mode="list", length=length(chunk))

        # iterate over repos in this chunk
        for (repo_id in chunk) {

            tryCatch({
                # extract list of taus and remove entries with values of 0
                dat <- data[[repo_id]]
                dat <- dat[dat > 0]

                # fit models
                fit_w  <- fitdist(dat, distr="weibull")
                fit_ln  <- fitdist(dat, "lnorm")

                B_w <- fit_to_burstiness(fit_w)
                B_ln <- fit_to_burstiness(fit_ln)

                M <- memory_coef(dat)
                
                best[[i]] <- data.frame(repo_id=repo_id, aic_w=fit_w$aic, aic_ln=fit_ln$aic, B_w=B_w, B_ln=B_ln, M=M, total_activity=length(dat))
                
                # save result, iterate
                ## best[[i]] <- result
                i <- i + 1
            }, error=function(e) {print(repo_id)})
        }
        # return result of this chunk
        do.call("rbind", best)
    }

    # stop cluster and log end time, return results
    stopCluster(cl)
    cat(paste("Finished at (", format(Sys.time(), "%X"), "), duration:", round(Sys.time() - t0, 2), "m\n"))
    results
}



directory <- "/home/nick/apps/data/github-burstiness/04-22/"

cat("\nReading in repos taus\n")
data_raw <- read_repos_to_taus(paste(directory, "repos_to_taus.txt", sep=""))
data <- data_raw[-which(names(data_raw) %in% c("5266168", "11651090", "12521560"))]

cat("\nCalculating repos statistics\n")
repo_B_M_stats <- calculate_B_M(data)

cat("\nMerging repos statistics\n")
repo_stats <- read.csv(paste(directory, "repository_stats_expanded.csv", sep=""))
repos_merged <- merge(repo_stats, repo_B_M_stats, by="repo_id")

cat("\nWriting repos statistics\n")
write.csv(repos_merged, paste(directory, "repos_burstiness_merged.csv", sep=""))


cat("\nReading in users taus\n")
users_taus <- read_repos_to_taus(paste(directory, "users_to_taus.txt", sep=""), n=106659)

cat("\nCalculating users taus\n")
users_B_M_stats <- calculate_B_M(users_taus)
old_names <- names(users_B_M_stats)
old_names[1] <- "user"
names(users_B_M_stats) <- old_names

cat("\nMerging users statistics\n")
user_success <- read.csv(paste(directory, "top_users_stats.csv", sep=""))
users_merged <- merge(users_B_M_stats, user_success, by="user")

cat("\nWriting users statistics\n")
write.csv(users_merged, paste(directory, "users_burstiness_merged.csv", sep=""))
