library(plm)
data("EmplUK")
options(error = function() { tb <- sys.calls(); print(tb[length(tb)]); traceback(3) })
m <- pgmm(log(emp) ~ lag(log(emp), -1) + lag(log(wage), 0) + lag(log(capital), 0) | lag(log(emp), -2),
          data = EmplUK, effect = "twoways", model = "onestep")
print(coef(m))
