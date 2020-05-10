#!/usr/bin/python3
# coding: utf-8

"""
main.py : //
"""


import matplotlib.pyplot as plt

from affichages import (plot_discretisation_debye_huckel,
                        plot_solution_debye_huckel,
                        plot_solution_poisson_boltzmann)
from lu import descente, lutri, remontee
from solveurs import solve_poisson_boltzmann_differences_finies


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
    # mu = 1
    plot_solution_poisson_boltzmann(
        1000,
        1,
        solve_poisson_boltzmann_differences_finies,
        "Schéma aux différences finies",
    )
    plot_solution_debye_huckel(1000, 1)
    plt.show()
    # mu = 4
    plot_solution_poisson_boltzmann(
        1000,
        4,
        solve_poisson_boltzmann_differences_finies,
        "Schéma aux différences finies",
    )
    plot_solution_debye_huckel(1000, 4)
    plt.show()


def main():

    # q1_2_3()
    plot_q5()
    plot_q6()
    plot_q8()

    # plot_variations_mu()

    # mu = find_convergence_mu(0, 7, 200, calcul_uk0_differences_finies)
    # print(mu) # 4.828635215759277

    calcul_uk0_newton(mu=7)


main()
