import pymc3 as pm
import numpy as np
import arviz as az

valuesY = np.array([0, 5, 10])
valuesTheta = [0.2, 0.5]

with pm.Model() as model:
    n = pm.Poisson('n', mu=10)
    Y = pm.Binomial('Y', n=n, p=valuesTheta, observed=valuesY)
    trace = pm.sample(1000, tune=1000)

az.plot_posterior(trace)
