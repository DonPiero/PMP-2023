import numpy as np
import arviz as az


numar_aruncari = 10
numar_experimente = 100
probabilitate_stema = 0.3
rezultate_posibile = ['ss', 'sb', 'bs', 'bb']
rezultate_count = {rezultat: 0 for rezultat in rezultate_posibile}
rezultate_total = {rezultat: 0 for rezultat in rezultate_posibile}

for i in range(numar_experimente):
    for j in range(numar_aruncari):
        moneda_1 = np.random.choice(['s', 'b'])
        moneda_2 = np.random.choice(['s', 'b'], p=[probabilitate_stema, 1 - probabilitate_stema])
        rezultat_aruncare = moneda_1 + moneda_2
        rezultate_total[rezultat_aruncare] += 1
        rezultate_count[rezultat_aruncare] += 1
    print(rezultate_count)
    for key in rezultate_posibile:
        rezultate_count[key] = 0

print()
print(rezultate_total)
print()

values = list(rezultate_total.values())
az.plot_kde(np.array(values))
plt.show()
