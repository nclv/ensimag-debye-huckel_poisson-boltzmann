#!/usr/bin/python3
# coding: utf-8

"""
affichages.py : Debye-Huckel (finite differences), Poisson-Boltzmann (finite differences and multidimensional Newton)
"""


import matplotlib.pyplot as plt
import numpy as np

from solveurs import solve_debye_huckel, solve_poisson_boltzmann_differences_finies

plt.style.use("ggplot")


def plot_solution(x, u, mu, system, method):
    """
        Affichage des couples (xi, ui)
    """
    plt.plot(x, u, label=f"{system} mu = {mu}")
    plt.xlabel("Points de discrétisations $x_i$")
    plt.ylabel("Solutions ponctuelles $u_i$")
    plt.title(f"{method} pour l'équation de {system}")
    plt.legend()


def plot_solution_poisson_boltzmann(n, mu, solveur_poisson_boltzmann, method):
    x, u, k = solveur_poisson_boltzmann(n, mu)
    print(f"k vaut {k} pour n = {n} et mu = {mu}")
    plot_solution(x, u, mu, system="Poisson-Boltzmann", method=method)


def plot_solution_debye_huckel(n, mu, solveur_debye_huckel=solve_debye_huckel, method="Schéma aux différences finies"):
    x, u, _ = solveur_debye_huckel(n=n, mu=mu)
    plot_solution(x, u, mu, system="Debye-Huckel", method=method)


def plot_discretisation_debye_huckel(mu, solveur_debye_huckel=solve_debye_huckel):
    """
        Etude de l'influence du pas de discrétisation
    """
    u0, h = [], []
    for n in [10, 15, 30, 70, 100, 1000, 10000]:
        _, u, _ = solveur_debye_huckel(n, mu=mu)
        # plot_solution(x, u, mu, "", "")
        # plt.show()
        u0.append(u[0])
        h.append(10 / n)
    # print(h, u0)
    plt.plot(h, u0, "o", label="$u_0$")
    plt.xlabel("Pas de discrétisations $h$")
    plt.ylabel("Solutions ponctuelles $u_0$")
    plt.title("Influence du pas de la discrétisation")
    plt.legend()

def plot_variations_mu_debye_huckel():
    fig = plt.figure()
    for mu in np.linspace(0.1, 6.4, 10):
        mu = round(mu, 3)
        plot_solution_debye_huckel(1000, mu)
    title = f"debye_mu_variations"
    plt.title(title)
    fig.savefig(title + ".png")


def plot_variations_mu_poisson_boltzmann(solveur_poisson_boltzmann, method):
    """
        Au dela de mu = 6.4 :
            RuntimeWarning: overflow encountered in sinh
            RuntimeWarning: invalid value encountered in double_scalars
            g = lambda x: np.sinh(x) - x
        RESOLVED avec math.sinh
    """
    fig = plt.figure()
    for mu in np.linspace(0.1, 6.4, 10):
        mu = round(mu, 3)
        plot_solution_poisson_boltzmann(1000, mu, solveur_poisson_boltzmann, method)
    title = f"poisson_mu_variations_{solveur_poisson_boltzmann.__name__}"
    plt.title(title)
    fig.savefig(title + ".png")
    # pour mu = 5.7 et mu = 6.4 on retombe ie. diverge

plot_variations_mu_debye_huckel()

def plot_q9():
    plot_variations_mu_poisson_boltzmann(solve_poisson_boltzmann_differences_finies, "Schéma aux différences finies")

def plot_variations_mu_superposed():
    for mu in np.linspace(0.1, 6.4, 10):
        mu = round(mu, 3)
        fig = plt.figure()
        plot_solution_debye_huckel(1000, mu)
        plot_solution_poisson_boltzmann(1000, mu, solve_poisson_boltzmann_differences_finies, "Schéma aux différences finies")
        title = f"superposed_mu{mu}_differences_finies"
        plt.title(title)
        fig.savefig(title + ".png")
