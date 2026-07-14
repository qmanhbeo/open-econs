args <- commandArgs(trailingOnly=TRUE)
in_csv <- args[1]; out_json <- args[2]
d <- read.csv(in_csv); y <- d$y
fit <- arima(y, order=c(1,0,1), include.mean=TRUE)
out <- list(cons=as.numeric(fit$coef["intercept"]), ar1=as.numeric(fit$coef["ar1"]),
            ma1=as.numeric(fit$coef["ma1"]), ll=as.numeric(fit$loglik))
cat(sprintf("{\"cons\":%s,\"ar1\":%s,\"ma1\":%s,\"ll\":%s}", out$cons, out$ar1, out$ma1, out$ll), file=out_json)
