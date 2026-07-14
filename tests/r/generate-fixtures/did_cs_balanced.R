#!/usr/bin/env Rscript
# did_cs_balanced.R - R parity anchor for did_cs() balanced panel (Callaway & Sant'Anna 2021).
#
# CS2021 DR-DiD group-time ATTs + simple/dynamic/group/calendar aggregation.
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

# Simple aggregation (existing)
agg_simple <- aggte(gt, type = "simple")

# Dynamic aggregation (event-time)
agg_dynamic <- aggte(gt, type = "dynamic")

# Group aggregation
agg_group <- aggte(gt, type = "group")

# Calendar aggregation
agg_calendar <- aggte(gt, type = "calendar")

# Group-time ATTs (post-treatment only)
post <- data.frame(group = gt$group, time = gt$t, att = gt$att, se = gt$se)
post <- post[post$group == 3 & post$time >= 3, ]

# Dynamic: ATT by event time
dyn_df <- data.frame(
  lead = agg_dynamic$egt,
  att  = agg_dynamic$att.egt,
  se   = agg_dynamic$se.egt
)

# Group: ATT by group
grp_df <- data.frame(
  group = agg_group$egt,
  att   = agg_group$att.egt,
  se    = agg_group$se.egt
)

# Calendar: ATT by time
cal_df <- data.frame(
  time = agg_calendar$egt,
  att  = agg_calendar$att.egt,
  se   = agg_calendar$se.egt
)

out <- c(
  # Group-time ATTs
  as.list(setNames(post$att, paste0("b_g", post$group, "_t", post$time - 1, "_", post$time))),
  as.list(setNames(post$se,  paste0("se_g", post$group, "_t", post$time - 1, "_", post$time))),
  # Simple aggregation
  list(agg_att_simple = agg_simple$overall.att, agg_se_simple = agg_simple$overall.se),
  # Dynamic aggregation
  list(agg_dynamic_overall_att = agg_dynamic$overall.att,
       agg_dynamic_overall_se  = agg_dynamic$overall.se),
  as.list(setNames(dyn_df$att, paste0("agg_dynamic_att_e", dyn_df$lead))),
  as.list(setNames(dyn_df$se,  paste0("agg_dynamic_se_e", dyn_df$lead))),
  # Group aggregation
  list(agg_group_overall_att = agg_group$overall.att,
       agg_group_overall_se  = agg_group$overall.se),
  as.list(setNames(grp_df$att, paste0("agg_group_att_g", grp_df$group))),
  as.list(setNames(grp_df$se,  paste0("agg_group_se_g", grp_df$group))),
  # Calendar aggregation
  list(agg_calendar_overall_att = agg_calendar$overall.att,
       agg_calendar_overall_se  = agg_calendar$overall.se),
  as.list(setNames(cal_df$att, paste0("agg_calendar_att_t", cal_df$time))),
  as.list(setNames(cal_df$se,  paste0("agg_calendar_se_t", cal_df$time))),
  # Metadata
  list(s_N = nrow(df))
)
write_json(out, args[2], auto_unbox = TRUE, digits = 17)
