args <- commandArgs(trailingOnly=TRUE)
in_csv <- args[1]; out_json <- args[2]
d <- read.csv(in_csv); y <- d$y
library(urca)
v <- ur.pp(y, type="Z-tau", model="constant", lag="short")
s <- as.numeric(v@teststat)
cat(sprintf("{\"stat\":%s}", s), file=out_json)
