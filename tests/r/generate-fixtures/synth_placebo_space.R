# Synthetic control placebo-in-space parity vs R Synth.
# Reads committed input CSV (argv[1]); writes expected-output JSON (argv[2]).
# Mirrors _R_PLACEBO_SPACE in tests/test_synth_placebo.py.
library(Synth)
library(jsonlite)
args <- commandArgs(trailingOnly = TRUE)
csv <- args[1]
out_json <- args[2]
pre_last <- 1994L
post_first <- 1995L
df <- read.csv(csv)
df$unit_num <- as.numeric(factor(df$unit))
lut <- unique(df[, c("unit_num", "unit")])
lut_names <- as.character(lut$unit)
names(lut_names) <- as.character(lut$unit_num)
ids <- sort(unique(df$unit_num))
times_all <- sort(unique(df$time))
pre_times <- times_all[times_all <= pre_last]
post_times <- times_all[times_all >= post_first]
analysis <- sort(unique(c(pre_times, post_times)))
treated_num <- df$unit_num[df$unit == "t"][1]
preds <- sort(names(df)[grep("^x[0-9]+$", names(df))])
ratios <- c()
pre_mspe <- c()
post_mspe <- c()
units <- c()
treated_ratio <- NA
for (u in ids) {
  donors_num <- setdiff(ids, u)
  dp <- dataprep(foo = df, dependent = "y", predictors = preds,
    predictors.op = "mean", special.predictors = NULL,
    unit.variable = "unit_num", unit.names.variable = "unit", time.variable = "time",
    treatment.identifier = u, controls.identifier = donors_num,
    time.predictors.prior = pre_times, time.optimize.ssr = pre_times,
    time.plot = analysis)
  so <- synth(dp)
  w <- as.numeric(so$solution.w)
  pre <- as.numeric(so$loss.v)
  treated_path <- as.numeric(dp$Y1plot[, 1])
  synthetic_path <- as.numeric(dp$Y0plot %*% so$solution.w)
  post_idx <- which(times_all >= post_first)
  post_err <- treated_path[post_idx] - synthetic_path[post_idx]
  post <- mean(post_err^2)
  if (u == treated_num) {
    treated_ratio <- as.numeric(post / pre)
  } else {
    ratios <- c(ratios, as.numeric(post / pre))
    pre_mspe <- c(pre_mspe, as.numeric(pre))
    post_mspe <- c(post_mspe, as.numeric(post))
    units <- c(units, as.character(lut_names[as.character(u)]))
  }
}
p_value <- as.numeric(mean(ratios >= treated_ratio))
out <- list(treated_ratio = treated_ratio, ratios = ratios,
  pre_mspe = pre_mspe, post_mspe = post_mspe, units = units, p_value = p_value)
write_json(out, out_json, auto_unbox = TRUE, digits = 15)
