args <- commandArgs(trailingOnly=TRUE)
in_csv <- args[1]; out_json <- args[2]
d <- read.csv(in_csv); y <- d$y
library(urca)
w <- ur.kpss(y, type="tau", lag="short")
s <- as.numeric(w@teststat)
cat(sprintf("{\"stat\":%s}", s), file=out_json)
