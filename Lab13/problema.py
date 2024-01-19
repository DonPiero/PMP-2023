import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    # Primul exercitiu
    centered_eight = az.load_arviz_data('centered_eight')
    nr_lanturi_centrate = centered_eight.posterior.chain.size
    nr_mostre_centrate = centered_eight.posterior.draw.size
    print(f"Lanturi modelul centrat: {nr_lanturi_centrate}")
    print(f"Esantioane modelul centrat: {nr_mostre_centrate}")

    non_centered_eight = az.load_arviz_data('non_centered_eight')
    nr_lanturi_necentrate = non_centered_eight.posterior.chain.size
    nr_mostre_necentrate = non_centered_eight.posterior.draw.size
    print(f"Lanturi modelul necentrat: {nr_lanturi_necentrate}")
    print(f"Esantioane modelul necentrat: {nr_mostre_necentrate}")

    az.plot_posterior(centered_eight)
    az.plot_posterior(non_centered_eight)
    plt.show()

    # Al doilea exercitiu
    rhat_centrat = az.rhat(centered_eight, var_names=["mu", "tau"])
    rhat_necentrat = az.rhat(non_centered_eight, var_names=["mu", "tau"])
    print("Rhat model centrat( mu, tau):", rhat_centrat)
    print("Rhat model necentrat( mu, tau):", rhat_necentrat)

    print(f"Rhat pentru ambele modele, in functie de mu")
    summaries = pd.concat(
        [az.summary(centered_eight, var_names=['mu']), az.summary(non_centered_eight, var_names=['mu'])])
    summaries.index = ['centered', 'non_centered']
    print(summaries)

    print(f"Rhat ambele modele, in functie de tau")
    summaries = pd.concat(
        [az.summary(centered_eight, var_names=['tau']), az.summary(non_centered_eight, var_names=['tau'])])
    summaries.index = ['centered', 'non_centered']
    print(summaries)

    autocor_centrat_mu = az.autocorr(centered_eight.posterior["mu"].values)
    autocor_necentrat_mu = az.autocorr(non_centered_eight.posterior["mu"].values)
    autocor_centrat_tau = az.autocorr(centered_eight.posterior["tau"].values)
    autocor_necentrat_tau = az.autocorr(non_centered_eight.posterior["tau"].values)

    print(f"Autocorelatia model centrat, in functie de parametrul mu : {np.mean(autocor_centrat_mu)}")
    print(f"Autocorelatia model necentrat, in functie de parametrul mu : {np.mean(autocor_necentrat_mu)}")
    print(f"Autocorelatia model centrat, in functie de parametrul tau : {np.mean(autocor_centrat_tau)}")
    print(f"Autocorelatia model necentrat, in functie de parametrul tau : {np.mean(autocor_necentrat_tau)}")

    az.plot_autocorr(centered_eight, var_names=["mu", "tau"], combined=True, figsize=(10, 10))
    az.plot_autocorr(non_centered_eight, var_names=["mu", "tau"], combined=True, figsize=(10, 10))
    plt.show()

    # Al treilea exercitiu
    divergente_centrate = centered_eight.sample_stats["diverging"].sum()
    divergente_necentrate = non_centered_eight.sample_stats["diverging"].sum()
    print(f"Divergente model centrat: {divergente_centrate.values}")
    print(f"Divergente model non-centrat: {divergente_necentrate.values}")

    az.plot_pair(centered_eight, var_names=["mu", "tau"], divergences=True)
    plt.suptitle("Model centrat")
    plt.show()

    az.plot_pair(non_centered_eight, var_names=["mu", "tau"], divergences=True)
    plt.suptitle("Model necentrat")
    plt.show()

    az.plot_parallel(centered_eight, var_names=["mu", "tau"])
    plt.suptitle("Paralel model centrat")
    plt.show()

    az.plot_parallel(non_centered_eight, var_names=["mu", "tau"])
    plt.suptitle("Paralel model necentrat")
    plt.show()


if __name__ == '__main__':
    main()
    
