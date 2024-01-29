import pandas as pd
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

# Subiectul 1a
df = pd.read_csv("Titanic.csv")
df["Age"].fillna(df["Age"].mean(), inplace=True) # punem media varstelor acolo unde varsta lipseste

# Subiectul 1b
with pm.Model() as model:
    betaAge = pm.Normal("beta_age", mu=0, sd=10) # Pentru varsta
    betaPclass = pm.Normal("beta_Pclass", mu=0, sd=10) # Pentru clasele de pasageri
    alpha = pm.Normal("alpha", mu=0, sd=10)
    mu = alpha + betaAge * df["Age"] + betaPclass * df["Pclass"] # Combinam clasele pentru a folosi rezultatul la functia viitoare bernoulli 
    likelihood = pm.Bernoulli("likelihood", p=pm.math.sigmoid(mu), observed=df["Survived"].values) # Folosim bernoulli pentru a estima supravietuitorii

with model:
    trace = pm.sample(2000, tune=1000, cores=2)

pm.traceplot(trace)
plt.show()

# Subiectul 1c: Variabila care va influenta cel mai mult supravietuirea pasagerilor va fi varsta ("Age").

# Subiectul 1d
ageSurv = 30
pclassSurv = [0, 1, 0]

with model:
    postPred = pm.sample_posterior_predictive(trace, samples=2000)
    survivalProb = postPred["survived"]
    survivalHdi = az.hdi(survivalProb, hdi_prob=0.9)
    print("Probabilitatea de supraviețuire: ", survivalHdi[ageSurv])
