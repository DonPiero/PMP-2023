import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, expon

params = [(4, 3), (4, 2), (5, 2), (5, 3)]
lambda_exp = 4
probabilities = [0.25, 0.25, 0.30, 0.20]
t = np.linspace(0, 10, 1000)

processing_times = [gamma.rvs(k, scale=theta, size=10000) for k, theta in params]
latency = expon.rvs(scale=1/lambda_exp, size=10000)

total_service_times = [processing + latency for processing in processing_times]
prob_gt_3ms = sum((total_service > 3).mean() * p for total_service, p in zip(total_service_times, probabilities))

print(f"Probabilitatea ca timpul total de servire să fie mai mare de 3 milisecunde: {prob_gt_3ms:.4f}")

plt.hist(np.concatenate(total_service_times), bins=50, density=True, alpha=0.6, color='b', label='Densitatea observată')
plt.show()
