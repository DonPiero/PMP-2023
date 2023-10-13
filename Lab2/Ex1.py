import numpy as np
from scipy import stats

lambda1 = 4
lambda2 = 6

nr_clienti = 10000

timp_servire_total = []

for _ in range(nr_clienti):
    mecanic = np.random.choice([1, 2], p=[0.4, 0.6])
    if mecanic == 1:
        timp_servire = stats.expon(scale=1/lambda1).rvs()
    else:
        timp_servire = stats.expon(scale=1/lambda2).rvs()

    timp_servire_total.append(timp_servire)


media_timp_servire = np.mean(timp_servire_total)
deviatia_standard_timp_servire = np.std(timp_servire_total)


print("Media timpului de servire:", media_timp_servire)
print("Deviația standard a timpului de servire:", deviatia_standard_timp_servire)

az.plot_kde(np.array(timp_servire_total))
plt.show()
