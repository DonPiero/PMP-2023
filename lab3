from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

modelEx1 = BayesianNetwork([('Cutremur', 'Incendiu'), ('Cutremur', 'Alarmă'), ('Incendiu', 'Alarmă')])

cpd_cutremur = TabularCPD(variable='Cutremur', variable_card=2, values=[[0.9995], [0.0005]])

cpd_incendiu = TabularCPD(variable='Incendiu', variable_card=2, values=[[0.01, 0.03],
                                                                        [0.99, 0.97]],
                          evidence=['Cutremur'], evidence_card=[2])

cpd_alarmă = TabularCPD(variable='Alarmă', variable_card=2,
                        values=[[0.9999, 0.98, 0.05, 0.02],
                                [0.0001, 0.02, 0.95, 0.98]],
                        evidence=['Cutremur', 'Incendiu'], evidence_card=[2, 2])

modelEx1.add_cpds(cpd_cutremur, cpd_incendiu, cpd_alarmă)

assert modelEx1.check_model()

inference = VariableElimination(modelEx1)

resultEx2 = inference.query(variables=['Cutremur'], evidence={'Alarmă': 1}) # calculam probabilitatea sa fi avut loc un cutremur,stiind ca alarma de incendiu a fost declansata
resultEx3 = inference.query(variables=['Incendiu'], evidence={'Alarmă': 0}) # afisati probabilitatea ca un incendiu sa fi avut loc, fara ca alarma de incendiu să se activeze

print(resultEx2)
print(resultEx3)
