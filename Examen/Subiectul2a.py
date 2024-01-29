# Subiectul 2.a.

import numpy as np
from scipy.stats import geom

thetaX = 0.3
thetaY = 0.5
numarIteratii = 10000
numarEvenimenteFavorabile = 0

for _ in range(numarIteratii):
    x = geom.rvs(thetaX)
    y = geom.rvs(thetaY)
    if x > y**2:
        numarEvenimenteFavorabile += 1

piAproximat = numarEvenimenteFavorabile*4 / numarIteratii 
print("Aproximarea lui pi folosind Monte Carlo:", piAproximat)

# Explicatii: În acest cod, am simulat variabilele aleatoare repartizate geometric folosind geom.rvs(theta). Am verificat apoi condiția dată și am numărat câte evenimente îndeplinesc această condiție. In curs, pi exte raportul dintre numarul punctelor cazute in interior( cazuri favorabile) inmultit cu 4, supra numarul de puncte aruncate( cazuri posibile). Acest cod folosește distribuțiile geometrice cu parametrii specificați și efectuează 10000 de iterații pentru a estima probabilitatea că X este mai mare decât pătratul lui Y.

