import pandas as pd
import numpy as np
from scipy.special import gammaln
from scipy.optimize import minimize_scalar

df = pd.read_csv('tests/r/fixtures/inputs/nbreg_input.csv')
y = df['y'].values.astype(float)
X = np.column_stack([np.ones(len(df)), df['x1'].values, df['x2'].values])

# Stata NB2 beta
b_stata2 = np.array([0.01797124, 0.4145353, -0.14962152])
# Stata NB1 beta
b_stata1 = np.array([-0.0244143, 0.49289614, -0.20775441])

def ll_nb2(beta, alpha):
    mu = np.exp(X@beta)
    a = 1.0/alpha
    t = a*np.log(a/(a+mu)) + y*np.log(mu/(a+mu)) + gammaln(y+a) - gammaln(a) - gammaln(y+1.0)
    return t.sum()

def ll_nb1(beta, alpha):
    mu = np.exp(X@beta)
    a = 1.0/alpha
    t = y*np.log(mu) - (y+a)*np.log(mu+alpha) + a*np.log(alpha) + gammaln(y+a) - gammaln(a) - gammaln(y+1.0)
    return t.sum()

# evaluate my NB2 form at Stata NB2 coefs, best alpha
def best_a2(beta):
    r = minimize_scalar(lambda a: -ll_nb2(beta, a), bounds=(1e-3,1e4), method='bounded')
    return r.x, -r.fun
a2, ll2 = best_a2(b_stata2)
print('my NB2-form at StataNB2 coef: alpha', a2, 'll', ll2, '(Stata ll -842.20281)')
a1, ll1 = best_a2(b_stata1)
print('my NB2-form at StataNB1 coef: alpha', a1, 'll', ll1, '(StataNB1 ll -836.53808)')

# Evaluate my NB1 form at Stata NB1 coefs
def best_a1(beta):
    r = minimize_scalar(lambda a: -ll_nb1(beta, a), bounds=(1e-3,1e4), method='bounded')
    return r.x, -r.fun
aa1, ll1b = best_a1(b_stata1)
print('my NB1-form at StataNB1 coef: alpha', aa1, 'll', ll1b)

# Try NB2 alternative: Var = mu + alpha*mu^2 but mean model mu=exp(eta), theta=1/alpha
# Stata NB2 delta = 1.263565. Is delta = alpha? Test: does my NB2 with this data have 2nd optimum?
# Try maximizing NB2 over beta AND alpha with Stata starting? 
res = minimize(lambda p: -ll_nb2(p[:3], np.exp(p[3])), [0.017,0.414,-0.15,np.log(1.2636)], method='Nelder-Mead')
print('NB2 start Stata: beta', res.x[:3], 'alpha', np.exp(res.x[3]), 'll', -res.fun)
