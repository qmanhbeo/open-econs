args <- commandArgs(trailingOnly=TRUE)
in_csv <- args[1]; out_json <- args[2]
d <- read.csv(in_csv); y <- d$y
library(urca)
z <- ur.za(y, model="both")
s <- as.numeric(z@teststat)
cat(sprintf("{\"stat\":%s}", s), file=out_json)
