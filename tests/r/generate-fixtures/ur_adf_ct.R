args <- commandArgs(trailingOnly=TRUE)
in_csv <- args[1]; out_json <- args[2]
d <- read.csv(in_csv); y <- d$y
library(urca)
u <- ur.df(y, type="trend", lags=0)
s <- as.numeric(u@teststat[1,"tau3"])
cv <- u@cval["tau3",]
cat(sprintf("{\"stat\":%s,\"cv1\":%s,\"cv5\":%s,\"cv10\":%s}", s, cv[1], cv[2], cv[3]), file=out_json)
