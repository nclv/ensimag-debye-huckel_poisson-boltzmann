#!/usr/bin/python3
# coding: utf-8

"""
main.py : //
"""


import matplotlib.pyplot as plt

from affichages import (
    plot_comparison_finite_differences,
    plot_comparison_newton,
    plot_discretisation_debye_huckel,
    plot_solution_debye_huckel,
    plot_solution_poisson_boltzmann,
    plot_variations_mu_debye_huckel,
    plot_variations_mu_poisson_boltzmann,
    plot_variations_mu_superposed_differences_finies,
    plot_variations_mu_superposed_newton,
)
from convergence import find_mu_limit
from lu import descente, lutri, remontee
from solveurs import (
    solve_poisson_boltzmann_differences_finies,
    solve_poisson_boltzmann_newton,
)


def q1_2_3():
    b, a, c = [2, 6], [1, 10, 10], [4, 5]
    l, v = lutri(a, b, c)
    print(l, v)
    y = descente(l, [3, 3, 3])
    print(y)
    x = remontee(v, c, [3, 3, 3])
    print(x)

    b, a, c = [-4, -3, -2, 2], [-2, 5, -1, 4, -2], [1, 2, -1, 1]
    l, v = lutri(a, b, c)
    assert l == [2.0, -1.0, -2.0, 1.0] and v == [-2, 3.0, 1.0, 2.0, -3.0]
    print(l, v)


def plot_q5():
    plot_solution_debye_huckel(1000, 1)
    plt.show()


def plot_q6():
    plot_discretisation_debye_huckel(1)
    plt.show()


def plot_q8():
    plot_comparison_finite_differences(1000, 1)
    plt.show()

    plot_comparison_finite_differences(1000, 4)
    plt.show()


def plot_q9():
    plot_variations_mu_poisson_boltzmann(
        solve_poisson_boltzmann_differences_finies, "Schéma aux différences finies"
    )


def plot_q9_extended():
    plot_variations_mu_debye_huckel()
    plot_variations_mu_superposed_differences_finies()

    mu_limit = find_mu_limit(0, 7, 200, solve_poisson_boltzmann_differences_finies)
    print(mu_limit)


def plot_q11():
    """
        Returns : Newton puis Finite Differences
            k vaut 3 pour n = 1000 et mu = 1
            k vaut 6 pour n = 1000 et mu = 1

            k vaut 5 pour n = 1000 et mu = 4
            k vaut 43 pour n = 1000 et mu = 4
    """
    plot_comparison_newton(1000, 1)
    plt.show()

    plot_comparison_newton(1000, 4)
    plt.show()


def plot_q12():
    plot_variations_mu_poisson_boltzmann(
        solve_poisson_boltzmann_newton, "Méthode de Newton"
    )


def plot_q12_extended():
    plot_variations_mu_superposed_newton()

    mu_limit = find_mu_limit(0, 7, 50, solve_poisson_boltzmann_newton)
    print(mu_limit)


def main():

    # q1_2_3()
    # plot_q5()
    # plot_q6()
    # plot_q8()
    # plot_q9()

    # plot_q9_extended()

    # plot_q11()
    # plot_q12()

    # plot_q12_extended()

    mu_limit = find_mu_limit(0, 1000, 50, solve_poisson_boltzmann_newton)
    print(mu_limit)


main()
