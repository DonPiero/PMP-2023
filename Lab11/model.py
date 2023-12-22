import numpy as np
import arviz as az
from sklearn.mixture import GaussianMixture

np.random.seed(42)
num_points = 500
mean1, mean2, mean3 = 0, 5, 10
std_dev1, std_dev2, std_dev3 = 1, 1.5, 0.5

data1 = np.random.normal(mean1, std_dev1, int(num_points * 0.4))
data2 = np.random.normal(mean2, std_dev2, int(num_points * 0.3))
data3 = np.random.normal(mean3, std_dev3, int(num_points * 0.3))

data = np.concatenate((data1, data2, data3))
np.random.shuffle(data)

data = data.reshape(-1, 1)
n_components_list = [2, 3, 4]

for n_components in n_components_list:
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(data)

    samples = gmm.sample(500)

    # Calcul log likelihood manual
    log_likelihood = gmm.score_samples(samples[0].reshape(-1, 1))

    # Reshape esantioanele pentru a fi compatibile cu arvizz
    samples_reshaped = samples[0].reshape((500, -1, 1))

    # Creare dictionar cu nume de variabile
    variables = {"posterior_predictive": samples_reshaped, "log_likelihood": log_likelihood}

    # Creare obiect arviz InferenceData
    az_data = az.from_dict(variables)

    waic = az.waic(az_data)
    loo = az.loo(az_data)

    print(f"Model cu {n_components} componente:")
    print(f"WAIC: {waic.waic:.2f}")
    print(f"LOO: {loo.loo:.2f}")
    print("\n")

# Comparatie si concluzie: cu cat valorile WAIC si LOO sunt mai mici, cu atat modelul este considerat mai bun.
