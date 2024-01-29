# Subiectul 2.b.

import numpy as np
from scipy.stats import geom

thetaX = 0.3
thetaY = 0.5
numarIteratii = 10000
k = 30
rezultateAproximari = []

for _ in range(k):
    numarEvenimenteDorite = 0
    for _ in range(numarIteratii):
        x = geom.rvs(thetaX)
        y = geom.rvs(thetaY)
        if x > y**2:
            numarEvenimenteDorite += 1
    piAproximat = numarEvenimenteDorite*4 / numarIteratii
    rezultateAproximari.append(piAproximat)

mediaAprobarilor = np.mean(rezultateAproximari)
deviatiaStandard = np.std(rezultateAproximari)

print("Media aproximărilor:", mediaAprobarilor)
print("Deviatia standard a aproximărilor:", deviatiaStandard)

# Explicatii: Am facut un "for" extern pentru a efectua 30 de seturi a cate 10000 aproximari ale lui "pi" si le-am contorizat in "rezultateAproximari". Media și deviația standard a acestor aproximari sunt calculate si afisate la sfarsit, folosind datele din "rezultateAproximari".
