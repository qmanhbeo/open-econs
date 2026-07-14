args <- commandArgs(trailingOnly=TRUE)
in_csv <- args[1]; out_json <- args[2]
d <- read.csv(in_csv); y <- d$y
library(rugarch)
spec <- ugarchspec(variance.model=list(model="sGARCH", garchOrder=c(1,1)),
                   mean.model=list(armaOrder=c(0,0), include.mean=TRUE),
                   distribution.model="norm")
fit <- ugarchfit(spec, data=y, solver="hybrid")
cf <- coef(fit)
out <- list(mu=as.numeric(cf["mu"]), omega=as.numeric(cf["omega"]),
            alpha=as.numeric(cf["alpha1"]), beta=as.numeric(cf["beta1"]),
            ll=as.numeric(likelihood(fit)))
cat(sprintf("{\"mu\":%s,\"omega\":%s,\"alpha\":%s,\"beta\":%s,\"ll\":%s}",
    out$mu, out$omega, out$alpha, out$beta, out$ll), file=out_json)
