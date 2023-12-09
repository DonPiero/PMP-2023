import pymc3 as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Admission.csv')
gre_scores = df['GRE'].values
gpa_scores = df['GPA'].values
admission = df['Admission'].values

#Primul subpunct

with pm.Model() as logistic_model:
    beta0 = pm.Normal('beta0', mu=0, sd=10)
    beta1 = pm.Normal('beta1', mu=0, sd=10)
    beta2 = pm.Normal('beta2', mu=0, sd=10)

    logit_p = beta0 + beta1 * gre_scores + beta2 * gpa_scores
    admission_prob = pm.Deterministic('admission_prob', pm.math.sigmoid(logit_p))
    admission_obs = pm.Bernoulli('admission_obs', p=admission_prob, observed=admission)

    trace = pm.sample(2000, tune=1000, cores=1)

pm.summary(trace)

#Subpunctul al doilea

beta0_samples = trace['beta0']
beta1_samples = trace['beta1']
beta2_samples = trace['beta2']

decision_boundary = -beta0_samples / beta2_samples
hdi = pm.hdi(decision_boundary, hdi_prob=0.94)

plt.scatter(gre_scores, gpa_scores, c=admission, cmap='viridis', alpha=0.7)
plt.axline((0, hdi[0]), slope=-np.median(beta1_samples) / np.median(beta2_samples), color='red', linestyle='--', label='Decision Boundary')
plt.fill_betweenx(y=[-5, 5], x1=hdi[0], x2=hdi[1], color='red', alpha=0.2, label='94% HDI')
plt.xlabel('GRE')
plt.ylabel('GPA')
plt.legend()
plt.show()
