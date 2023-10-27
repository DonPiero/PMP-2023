import numpy as np
from scipy.stats import norm, expon, poisson

lambdaPoisson = 20
medieDistributieNormala = 2
deviatiaStandard = 0.5

# Subpunctul A

medieDistributieExponentiala = 0.1 # valoare aleatoare alocata pentru subpunctul a)
numarClientiSimulatiA = poisson.rvs(lambdaPoisson, size=1)

def simulareServireRestaurant(numarClientiSimulatiA):
    timpPlasareComanda = norm.rvs(medieDistributieNormala, deviatiaStandard, size=numarClientiSimulatiA)
    timpPregatireComanda = expon.rvs(scale=medieDistributieExponentiala, size=numarClientiSimulatiA)
    timpTotalComanda = timpPregatireComanda + timpPlasareComanda
    return np.sum(timpTotalComanda)

# Subpunctul B

valoriAlpha = np.linspace(0.1, 10, 1000) # numar de simulari pentru subpunctul b)
alphaMaxim = None
contor = 0
numarExperiente = 0

def simulareServire(alpha):
    numarClientiB = poisson.rvs(lambdaPoisson, size=1)[0]
    timpPlasareComanda = norm.rvs(medieDistributieNormala, deviatiaStandard, size=numarClientiB)
    timpPregatireComanda = expon.rvs(scale=alpha, size=numarClientiB)
    timpTotalComanda = timpPregatireComanda + timpPlasareComanda
    return np.all(timpTotalComanda <= 15)

for alpha in valoriAlpha:
    numarExperiente += 1
    if simulareServire(alpha):
        contor += 1
        alphaMaxim = alpha
    else:
        break

probabilitate = contor / numarExperiente # testam la afisare daca >= 95%

# Subpunctul C

numarClientiSimulatiC = 1000 # numar de simulari pentru subpunctul c)

def simulareTimpAsteptare():
    timpPlasareComanda = norm.rvs(medieDistributieNormala, deviatiaStandard, size=1)[0]
    timpPregatireComanda = expon.rvs(scale=alphaMaxim, size=1)[0] # folosim alpha maxim calculat la b)
    return timpPlasareComanda + timpPregatireComanda

timpTotalAsteptare = sum(simulareTimpAsteptare() for _ in range(numarClientiSimulatiC))
timpMediuAsteptare = timpTotalAsteptare / numarClientiSimulatiC

# Afisari exercitiu

print(f"A) Va dura {simulareServireRestaurant(numarClientiSimulatiA):.2f} minute pentru a fi serviti {numarClientiSimulatiA} de clienti.")
print(f"B) Probabilitatea pentru a servi mancarea tuturor clienților dintr-o oră( în mai puțin de 15 minute) este mai mare de 95% ({probabilitate >= 0.95}), valoarea maxima al lui alpha este de {alphaMaxim:.2f}.")
print(f"C) Timpul mediu de așteptare pentru a fi servit un client este de {timpMediuAsteptare:.2f} minute, valoarea lui alpha fiind de {alphaMaxim:.2f}.")
