import numpy as np
import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt

data = pd.read_csv('trafic.csv')
trafic = data['trafic'].values

with pm.Model() as model:
    lambda = pm.Exponential('lambda', lam=1)
    trafic_observed = pm.Poisson('traficobserved', mu=lambda, observed=trafic)
    trace = pm.sample(10000, tune=2000, step=pm.Metropolis())

pm.traceplot(trace)
plt.show()

intervale = [(4, 7), (7, 8), (8, 16), (16, 19), (19, 24)]
for interval in intervale:
    start, end = interval
    values = trace['lambda'][(trace['lambda'] >= start) & (trace['lambda'] < end)]
    print(f'Intervalul {start}-{end}:')
    print(f'Valoare medie estimată pentru λ: {values.mean():.2f}')
    print(f'Interval de confidență: ({np.percentile(values, 2.5):.2f}, {np.percentile(values, 97.5):.2f})')