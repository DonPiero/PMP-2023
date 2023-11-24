import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
def game():
    start = np.random.choice([0, 1])
    if start == 0:
      n = np.random.choice([0, 1], p=[1 / 2, 1 / 2])
      m = np.random.choice([0, 1], p=[1 / 3, 2 / 3])
      if n == 1:
         m = m + np.random.choice([0, 1], p=[1 / 3, 2 / 3])
    else:
      n = np.random.choice([0, 1], p=[1 / 3, 2 / 3])
      m = np.random.choice([0, 1], p=[1 / 2, 1 / 2])
      if n == 1:
         m = m + np.random.choice([0, 1], p=[1 / 2, 1 / 2])
    if n < m:
        win = 1 - start
    else:
        win = start
    return win

simulations = 1000
first_player = 0
second_player = 0
for _ in range(simulations):
    if game() == 0:
        first_player = first_player + 1
    else:
        second_player = second_player + 1

winning_prob_j0 = first_player / simulations * 100
winning_prob_j1 = second_player / simulations * 100

print(f"Șansele pentru j0: {winning_prob_j0}%")
print(f"Șansele pentru j1: {winning_prob_j1}%")

modelEx1 = BayesianNetwork([('Incepe', 'J0'), ('Incepe', 'J1'), ('J0', 'J1')])
cpd_incepe = TabularCPD(variable='Incepe', variable_card=2, values=[[0.5], [0.5]])
cpd_j0 = TabularCPD(variable='J0', variable_card=2, values=[[0.5, 0.5, 0.5, 0.5],
                                                            [0.33, 0.33, 0.17, 0.17]],
                          evidence=['Incepe', 'J1'], evidence_card=[2])
cpd_j1 = TabularCPD(variable='J1', variable_card=2,
                        values=[[0.66, 0.66, 0.25, 0.25],
                                [0.66, 0.34, 0.66, 0.34]],
                        evidence=['Incepe', 'J0'], evidence_card=[2, 2])
modelEx1.add_cpds(cpd_incepe, cpd_j0, cpd_j1)

assert modelEx1.check_model()
