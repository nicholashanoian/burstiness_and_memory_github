library(fitdistrplus)
library(viridis)
library(latex2exp)
library(Hmisc)
library(progress)
library(scales)
library(ggplot2)
library(dplyr)
library(tidyverse)
library(ggExtra)
library(cowplot)
library(ggpubr)
library(corrplot) # correlation heatmap


read_repos_to_taus <- function(path, n=150039) {
    #' read in file with one line for each repo of the form
    #' repo_id:tau0 tau1 tau2 tau3
    
    ## pb = txtProgressBar(min=0, max=n, initial=0, style=3)
    ## pb <- progress_bar$new(format="[:bar] :percent at :tick_rate it/s", total=n)
    ## data <- vector(mode="list", length=n)

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

theme_nhh <- function(base_size=14, base_family="helvetica") {
    library(grid)
    library(ggthemes)
    (theme_foundation(base_size=base_size, base_family=base_family) +
     theme_pubclean() +
     theme(panel.grid.major.x = element_line(color="#DDDDDD", linetype="dotted"),
           panel.grid.major.y = element_line(color="#DDDDDD", linetype="dotted"),
           text=element_text(size=base_size),
           legend.position="right"))
}


directory <- "/home/nick/apps/data/github-burstiness/04-22/"

repos_merged <- read.csv(paste(directory, "repos_burstiness_merged.csv", sep=""))
users_merged <- read.csv(paste(directory, "users_burstiness_merged.csv", sep=""))

repos_to_taus <- read_repos_to_taus(paste(directory, "repos_to_taus.txt", sep=""))

figures_dir <- "figures/"


stargazers_vs <- function(data, var, xlab, xlab_inset, inset_x, inset_y, inset_width, inset_height) {
    # scatterplot of stargazers vs var with trend line, inset of same
    # plot but log scale y, marginal density plots

    # log transformed inset
    logged <- data %>%
            ggplot(aes_string(x=var, y="max_stargazers")) +
            geom_point(alpha=0.01, size=0.5) +
            geom_smooth(method="gam", color="red") +
            xlim(-1,1) +
            labs(x=TeX(xlab_inset),
                 y=TeX("\\textit{S}")) +
        scale_y_continuous(trans=log10_trans(),
                           breaks=trans_breaks("log10", function(x) 10^x),
                           labels=trans_format("log10", math_format(10^.x))) +
        theme_nhh()


    # main plot
    filtered <- data %>%
        filter(max_stargazers < 1031.65) %>%
        ggplot(aes_string(x=var, y="max_stargazers")) +
        geom_point(alpha=0.01) +
        geom_smooth(method="gam", color="red") +
        xlim(-1,1) +
        ylim(-10,NA) +
        theme_nhh() +
        labs(x=TeX(xlab),
             y=TeX("success \\textit{S}"))
    # add marginal densities to main plot
    filtered_with_marginals <- ggMarginal(filtered, type="density", color="red", fill="red", alpha=0.4)

    # put the two plots together
    ggdraw() +
        draw_plot(filtered_with_marginals) +
        draw_plot(logged, x=inset_x, y=inset_y, width=inset_width, height=inset_height)
}

p1 <- stargazers_vs(repos_merged, "B_w", "burstiness \\textit{B}", "\\textit{B}",
                    inset_x=0.15, inset_y=0.35, inset_width=0.4, inset_height=0.4)

p2 <- stargazers_vs(repos_merged, "M", "memory coefficient \\textit{M}", "\\textit{M}",
                    inset_x=0.13, inset_y=0.35, inset_width=0.3, inset_height=0.3)

p <- ggarrange(p1, p2, nrow=1, ncol=2, labels=c(" a", " b"), font.label=list(size=24))
ggsave(paste(figures_dir, "stargazers-vs.png", sep=""), p, width=12, height=6)




joint_burstiness_memory <- function(data, cuts) {
    single <- data %>% 
        ggplot(aes(x=round(M, 3),
                   y=round(B_w, 3))) +
        geom_bin2d(aes(fill=..density..), bins=100)+ 
        scale_fill_viridis_c() +
        xlim(-1,1) +
        ylim(-1,1) +
        coord_fixed() +
        theme_nhh() +
        labs(x=TeX("memory coefficient \\textit{M}"),
             y=TeX("burstiness \\textit{B}"),
             fill="Density")

    split_up <- data %>% mutate(stargazer_cat=cut2(max_stargazers, cuts=cuts)) %>%
        ggplot(aes(x=round(M, 3),
                   y=round(B_w, 3))) +
            geom_bin2d(aes(fill=..density..), bins=75)+ 
            scale_fill_viridis_c() +
            facet_wrap(.~stargazer_cat) +
            xlim(-1,1) +
            ylim(-1,1) +
            coord_fixed() +
            theme_nhh() +
            labs(x=TeX("memory coefficient \\textit{M}"),
                 y=TeX("burstiness \\textit{B}"),
                 fill="Density")

    ggarrange(single, split_up, nrow=1, ncol=2, labels=c(" a", " b"), font.label=list(size=24))
}
stargazer_cuts <- c(1,5,10,50)
p <- joint_burstiness_memory(repos_merged, cuts=stargazer_cuts)
ggsave(paste(figures_dir, "joint-burstiness-memory.png", sep=""), p, width=12, height=6)

most_probable <- function(x) {
    dens <- density(x)
    dens$x[which.max(dens$y)]
}

# most probable value of joint dist
c(B=most_probable(repos_merged$B_w), M=most_probable(repos_merged$M))

repos_merged %>% mutate(stargazer_cat=cut2(max_stargazers, cuts=stargazer_cuts)) %>%
    group_by(stargazer_cat) %>%
    summarize(most_prob_B=most_probable(B_w),
              most_prob_M=most_probable(M))


M_density <- density(repos_merged$M)
M_density$x[which.max(M_density$y)]

# percentage of teams in each chunk
ecdf(repos_merged$max_stargazers)(stargazer_cuts)








barcode_plot <- function(ts) {
    # plot time distribution like a barcode
    
    ggplot(data.frame(x=ts), aes(x=x, y=0)) +
        geom_point(alpha=0.6, shape="|", size=4) +
        theme_nhh(base_size=10) +
        scale_y_continuous(breaks=0) +
        labs(x=TeX("\\textit{t} (hours)")) +
        # black border, nothing else inside except ticks
        theme(panel.border = element_rect(colour = "black", fill=NA, size=1),
              aspect.ratio=0.1,
              axis.title.y=element_blank(),
              axis.ticks.y=element_blank(),
              axis.text.y=element_blank(),
              panel.grid.major.x = element_blank(),
              panel.grid.minor.x = element_blank(),
              panel.grid.major.y = element_blank(),
              panel.grid.minor.y = element_blank(),
              panel.background = element_blank())
}


ccdf_plot <- function(taus, spaces=25) {
    # plot the ccdf for empirical, weibull, and log-normal distributions of taus
    
    # fit dists
    fit_w  <- fitdist(taus, distr="weibull")
    fit_ln <- fitdist(taus, distr="lnorm")
    fit_exp <- fitdist(taus, distr="exp")

    # range of xs to plot dists
    plotting_xs <- seq(min(taus),max(taus), by=0.1)

    # to add right padding to legend labels
    str_pad_custom <- function(labels){
        new_labels <- stringr::str_pad(labels, spaces, "right")
        # remove padding of last label
        new_labels[length(labels)] <- labels[length(labels)]
        return(new_labels)
    }

    # gray, orange, blue
    cols <- c("Empirical"="#555555", "Weibull"="#dea000", "Log-normal"="#76add7", "Exponential"="#df3123")
    linetypes <- c("Empirical"="solid", "Weibull"="dotdash", "Log-normal"="dashed", "Exponential"="dotted")

    if (length(taus) > 10000000) {
        taus_sorted <- sort(taus)[seq(1, length(taus), by=50)]
    } else {
        taus_sorted <- sort(taus)
    }
    
    ggplot() +
        # empirical
        geom_step(aes(x=taus_sorted,
                      y=1-(ecdf(taus))(taus_sorted),
                      color="Empirical", linetype="Empirical"), size=1.5) +
        # weibull
        geom_line(aes(x=plotting_xs,
                      y=1 - pweibull(plotting_xs, shape=fit_w$estimate["shape"], scale=fit_w$estimate["scale"]),
                      color="Weibull", linetype="Weibull"), size=1.5) +
        # exp
        geom_line(aes(x=plotting_xs,
                      y=1 - pexp(plotting_xs, rate=fit_exp$estimate["rate"]),
                      color="Exponential", linetype="Exponential"), size=1.5) +
        # log-normal
        geom_line(aes(x=plotting_xs,
                      y=1 - plnorm(plotting_xs, meanlog=fit_ln$estimate["meanlog"], sdlog=fit_ln$estimate["sdlog"]),
                      color="Log-normal", linetype="Log-normal"), size=1.5) +
        # log-log scale
        scale_y_continuous(trans=log10_trans(),
                           breaks=trans_breaks("log10", function(x) 10^x),
                           labels=trans_format("log10", math_format(10^.x)),
                           limits=c(10^-4, NA)) +
        scale_x_continuous(trans=log10_trans(),
                           breaks=trans_breaks("log10", function(x) 10^x),
                           labels=trans_format("log10", math_format(10^.x))) +
        theme_nhh() +
        # add legends
        scale_color_manual(name="Distribution", values=cols, labels=str_pad_custom) +
        scale_linetype_manual(name="Distribution", values=linetypes, labels=str_pad_custom) +
        labs(x=TeX("$\\tau$"),
             y=TeX("\\textit{P}($>\\tau$)")) +
        # move legend to bottom and make boxes for each item longer
        theme(aspect.ratio=1,
              legend.position="bottom",
              legend.key.width=unit(2, "cm")) +
        # move legend title to top and center it
        guides(color=guide_legend(title.position="top", title.hjust=0.5),
               linetype=guide_legend(title.position="top", title.hjust=0.5))
}

stats_plot <- function(stats) {
    # plot "B=0.123, M=0.123"
    
    ggplot() +
        geom_label(aes(label=TeX(paste("$\\textit{B} = ",round(stats$B_w, 3), "$,  ", 
                                       "$\\textit{M} = ",round(stats$M, 3), "$", sep=""),
                                 output="character"),
                       x=0, y=0), color="black", size=4, parse=TRUE, label.size=0) +
        theme_void()
}



one_repo_plot <- function(repo_id, repos_merged, repos_to_taus) {
    # combine the three plots
    
    stats <- repos_merged[repos_merged$repo_id ==repo_id,]
    taus <- repos_to_taus[[as.character(repo_id)]]
    taus <- taus[taus > 0]

    ggdraw() +
        draw_plot(ccdf_plot(taus) + theme(legend.position="none")) +
        draw_plot(barcode_plot(cumsum(taus)),
                  x=0.22, y=0.25, width=0.7, height=0.19) +
        draw_plot(stats_plot(stats),
                  x=0.17, y=0.4, width=0.6, height=0.1)
}





three_repos_plot <- function(repo_ids, repos_merged, repos_to_taus) {
    # legend to add to combined plot
    sample_legend <- get_legend(ccdf_plot(c(1,2,3)))

    # plot the three repos
    repo_plots <- ggarrange(one_repo_plot(repo_ids[1], repos_merged, repos_to_taus),
                            one_repo_plot(repo_ids[2], repos_merged, repos_to_taus),
                            one_repo_plot(repo_ids[3], repos_merged, repos_to_taus),
                            one_repo_plot(repo_ids[4], repos_merged, repos_to_taus),
                            one_repo_plot(repo_ids[5], repos_merged, repos_to_taus),
                            one_repo_plot(repo_ids[6], repos_merged, repos_to_taus),
                            nrow=2, ncol=3, labels=c(" a", " b", " c", " d", " e", " f"), font.label=list(size=24))
    # add legend back
    ggarrange(repo_plots + theme(plot.margin=unit(c(0,0,-1,0), "cm")),
              as_ggplot(sample_legend), ncol=1, nrow=2, heights=c(1, 0.15))

}



# around mean of M and B_w: 11338900
repos_merged %>%
    filter(M > 0.084 & M < 0.087 & B_w > 0.54 & B_w < 0.55) %>%
    select(repo_id, total_work, B_w, M) %>%
    filter(total_work > 120 & total_work < 130)

# high B_w: 8179952
repos_merged %>%
    filter(B_w > 0.9) %>%
    select(repo_id, total_work, B_w, M) %>%
    filter(total_work > 120 & total_work < 130)

# low B_w: 6203143
repos_merged %>%
    filter(M < 0.01 & B_w < 0) %>%
    select(repo_id, total_work, B_w, M) %>%
    filter(total_work > 120 & total_work < 130)

# 0 B_w: 8501375
repos_merged %>%
    filter(B_w < 0.05 & B_w > -0.05) %>%
    select(repo_id, total_work, B_w, M) %>%
    filter(total_work > 120 & total_work < 130)
    

# high M: 3193259
repos_merged %>%
    filter(M > 0.9) %>%
    select(repo_id, total_work, B_w, M) %>%
    filter(total_work > 120 & total_work < 130)

# low M: 1841110
repos_merged %>%
    filter(M < -0.3) %>%
    select(repo_id, total_work, B_w, M) %>%
    filter(total_work > 120 & total_work < 130)

## chosen_repos <- c(15823141, 4996903, 15205769)

# close to mean(total_work) 125.8
summary(repos_merged$total_work)

chosen_repos <- c(11338900, 8179952, 6203143, 8501375, 3193259, 1841110)


p <- three_repos_plot(chosen_repos, repos_merged, repos_to_taus)

# save
ggsave(paste(figures_dir, "sample-repos.png", sep=""), p, width=12, height=10)

# verify that all fall between 120 and 130
repos_merged[repos_merged$repo_id %in% chosen_repos,]$total_work



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










merged %>%
    mutate(stargazer_cat=cut2(max_stargazers, cuts=c(1,10,100,1000,10000))) %>%
    group_by(stargazer_cat) %>%
    dplyr::summarize(mean=mean(burstiness))





# linear regression model
regression_data <- filter(repos_merged, max_stargazers < 1031.65) %>% dplyr::select(c(B_w, M, team_size, max_stargazers, total_work, effective_size, experience, diversity, n_leads, age)) %>% scale() %>% data.frame() 

model1 <- lm(max_stargazers ~ team_size + effective_size + total_work +
                 experience + diversity + n_leads + age, data=regression_data)
summary(model1)

model2 <- lm(max_stargazers ~ team_size + effective_size + total_work +
                 experience + diversity + n_leads + age + B_w + M, data=regression_data)
summary(model2)

anova(model2, model1)

# model coefs
round(data.frame(est=coef(model2),
                 moe=sqrt(diag(vcov(model2))) * 1.96,
                 pval=summary(model2)$coefficients[,"Pr(>|t|)"]), 4)




all_taus <- vector(mode="list", length=sum(repos_merged$total_activity))
i <- 1
for (repo in repos_to_taus) {
    for (tau in repo) {
        if (tau > 0) {
            all_taus[[i]] <- tau
            i <- i + 1
        }
    }
}
all_taus <- unlist(all_taus)

all_taus_ccdf_plot <- ccdf_plot(all_taus, space=5)

B_w_hist <- ggplot(repos_merged, aes(x=B_w)) +
    geom_histogram(aes(y=..density..), bins=40,
                   color="#009f73", fill="#009f73", alpha=0.4) +
    xlim(-1, 1) +
    labs(x=TeX("\\textit{B}$_{W}$")) +
    theme_nhh()

B_ln_hist <- ggplot(repos_merged, aes(x=B_ln)) +
    geom_histogram(mapping=aes(y=..density..), bins=40, 
                   color="#009f73", fill="#009f73", alpha=0.4) +
    xlim(-1, 1) +
    labs(x=TeX("\\textit{B}$_{LN}$")) +
    theme_nhh()



p <- ggarrange(ggarrange(all_taus_ccdf_plot + theme(legend.key.width=unit(0.75, "cm")),
                         labels=c("a"), font.label=list(size=24)),
                      ggarrange(B_w_hist, B_ln_hist, nrow=2, ncol=1, labels=c("b", "c"), font.label=list(size=24)),
                      nrow=1, ncol=2, widths=c(1,0.5))
ggsave(paste(figures_dir, "picking-dist.png", sep=""), p, width=9, height=6)


fit_w_all_taus <- fitdist(all_taus, distr="weibull")
fit_ln_all_taus <- fitdist(all_taus, distr="lnorm")

relative_likelihood <- function(aic1, aic2) {
    exp(-1/2 * abs(aic1-aic2))
}

fit_to_burstiness(fit_w_all_taus)


# ln better
sum(repos_merged$aic_ln < repos_merged$aic_w)
# w better
sum(repos_merged$aic_ln >repos_merged$aic_w)




x <- rexp(10000)

fit_x_w <- fitdist(x, distr="weibull")
fit_x_ln <- fitdist(x, distr="lnorm")

fit_to_burstiness(fit_x_ln)
fit_to_burstiness(fit_x_w)
