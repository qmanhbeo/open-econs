import pandas as pd, numpy as np
from scipy.special import gammaln
from scipy.optimize import minimize

df = pd.read_csv('tests/r/fixtures/inputs/nbreg_input.csv')
y = df['y'].values.astype(float)
x1 = df['x1'].values.astype(float)
x2 = df['x2'].values.astype(float)
X = np.column_stack([np.ones(len(df)), x1, x2])

def ll_nb1(beta, alpha):
    eta = X @ beta
    mu = np.exp(eta)
    a = 1.0/alpha
    t = y*np.log(mu) - (y+a)*np.log(mu+alpha) + a*np.log(alpha) + gammaln(y+a) - gammaln(a) - gammaln(y+1.0)
    return -t.sum()

def ll_nb2(beta, alpha):
    eta = X @ beta
    mu = np.exp(eta)
    a = 1.0/alpha
    t = a*np.log(a/(a+mu)) + y*np.log(mu/(a+mu)) + gammaln(y+a) - gammaln(a) - gammaln(y+1.0)
    return -t.sum()

# NB1 optimize
res1 = minimize(lambda p: ll_nb1(p[:3], np.exp(p[3])), [0,0,0,0.0], method='Nelder-Mead')
print('NB1 direct: beta', res1.x[:3], 'alpha', np.exp(res1.x[3]), 'll', -res1.fun)
# NB2 optimize
res2 = minimize(lambda p: ll_nb2(p[:3], np.exp(p[3])), [0,0,0,0.0], method='Nelder-Mead')
print('NB2 direct: beta', res2.x[:3], 'alpha', np.exp(res2.x[3]), 'll', -res2.fun)
print('Stata NB1: beta', [0.49289614, -0.20775441, -0.0244143], 'alpha 1.0563 ll -836.53808')
print('Stata NB2: beta', [0.4145353, -0.14962152, 0.01797124], 'delta 1.263565 ll -842.20281')
