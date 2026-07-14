# Synthetic control parity vs R Synth -- DEFAULT predictor mode.
# Reads committed input CSV (argv[1]); writes expected-output JSON (argv[2]).
# Mirrors the inline _R_SCRIPT in tests/test_synth.py (mode="default").
library(Synth)
library(jsonlite)
args <- commandArgs(trailingOnly = TRUE)
csv <- args[1]
out_json <- args[2]
mode <- "default"
pre_last <- 1994L
post_first <- 1995L
df <- read.csv(csv)
# Synth's dataprep requires a NUMERIC unit variable; keep the original string
# names for the solution.w rownames via unit.names.variable.
df$unit_num <- as.numeric(factor(df$unit))
treated_num <- df$unit_num[df$unit == "t"][1]
donors_num <- sort(unique(df$unit_num[df$unit != "t"]))
times_all <- sort(unique(df$time))
pre_times <- times_all[times_all <= pre_last]
post_times <- times_all[times_all >= post_first]
analysis <- sort(unique(c(pre_times, post_times)))
# Explicit predictors are every "x<digits>" column present in the panel.  This
# makes the script work for both the well-determined (x1..x12) and the
# rank-deficient (x1, x2) explicit fixtures without hardcoding the count.
preds <- sort(names(df)[grep("^x[0-9]+$", names(df))])
if (mode == "default") {
   sp <- lapply(pre_times, function(t) list("y", t, "mean"))
   dp <- dataprep(foo = df, dependent = "y", predictors = NULL,
     special.predictors = sp, predictors.op = "mean",
     unit.variable = "unit_num", unit.names.variable = "unit", time.variable = "time",
     treatment.identifier = treated_num, controls.identifier = donors_num,
     time.predictors.prior = pre_times, time.optimize.ssr = pre_times,
     time.plot = analysis)
 } else {
   dp <- dataprep(foo = df, dependent = "y", predictors = preds,
     predictors.op = "mean", special.predictors = NULL,
     unit.variable = "unit_num", unit.names.variable = "unit", time.variable = "time",
     treatment.identifier = treated_num, controls.identifier = donors_num,
     time.predictors.prior = pre_times, time.optimize.ssr = pre_times,
     time.plot = analysis)
 }
so <- synth(dp)
w <- as.numeric(so$solution.w)
# solution.w rownames are the numeric unit codes; map them back to the
# original donor string names via the unit_num <-> unit lookup.
lut <- unique(df[, c("unit_num", "unit")])
lut_names <- as.character(lut$unit)
names(lut_names) <- as.character(lut$unit_num)
w_names <- as.character(lut_names[as.character(rownames(so$solution.w))])
v <- as.numeric(so$solution.v)
v_names <- names(so$solution.v)
tw <- as.numeric(rownames(dp$Y1plot))
treated_path <- as.numeric(dp$Y1plot[, 1])
synthetic_path <- as.numeric(dp$Y0plot %*% so$solution.w)
gap_path <- treated_path - synthetic_path
out <- list(
  w = w, w_names = w_names,
  v = v, v_names = if (is.null(v_names)) rep("y", length(v)) else v_names,
  loss_v = as.numeric(so$loss.v),
  times = tw, treated = treated_path, synthetic = synthetic_path, gap = gap_path
)
write_json(out, out_json, auto_unbox = TRUE, digits = 15)
