#!/usr/bin/env Rscript
# staggered_did_balanced.R - R parity anchor for staggered_did() balanced panel.
#
# CS2021 DR-DiD group-time ATTs + simple aggregation.
# Entities 0-9: never-treated, 10-19: treated at t=3, 20-29: excluded (gvar=5).
#
# Args: argv[1] = input csv, argv[2] = output json

library(did)
library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
df <- read.csv(args[1])

# Balanced: keep entities 0-19 (gvar=5 entities 20-29 excluded, never turn on)
df <- df[df$entity < 20, ]
df$gvar <- 0L
df$gvar[df$entity >= 10 & df$entity < 20] <- 3L

gt <- att_gt(
  yname = "y", tname = "time", idname = "entity", gname = "gvar",
  xformla = ~ x + z, data = df, est_method = "dr",
  control_group = "nevertreated", base_period = "varying",
  bstrap = FALSE, cband = FALSE, compute_inffunc = TRUE, print_details = FALSE
)
agg <- aggte(gt, type = "simple")

post <- data.frame(group = gt$group, time = gt$t, att = gt$att, se = gt$se)
post <- post[post$group == 3 & post$time >= 3, ]

out <- c(
  as.list(setNames(post$att, paste0("b_g", post$group, "_t", post$time - 1, "_", post$time))),
  as.list(setNames(post$se,  paste0("se_g", post$group, "_t", post$time - 1, "_", post$time))),
  list(agg_att_simple = agg$overall.att, agg_se_simple = agg$overall.se, s_N = nrow(df))
)
write_json(out, args[2], auto_unbox = TRUE, digits = 17)
