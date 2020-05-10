#!/usr/bin/python3
# coding: utf-8

"""
solveurs.py : Debye-Huckel (finite differences), Poisson-Boltzmann (finite differences and multidimensional Newton)
"""

import math

import numpy as np

from lu import descente, lutri, remontee
from utils import tridiag


def solve_debye_huckel(n, mu):
    """
        Résolution de l'équation de Debye-Huckel.

        Sortie : x vecteur réel de dimension n, u vecteur solution de dimension n
    """
    h = 10 / n
    # Initialisation
    x = [1] * n
    b, a, c, z = [0] * (n - 1), [-(2 + h ** 2)] * n, [0] * (n - 1), [0] * n
    c[0], z[0] = 2, mu * h * (h - 2)
    # Attribution
    for i in range(1, n - 1):
        xi = 1 + i * h
        b[i - 1] = 1 - h / (2 * xi)
        c[i] = 1 + h / (2 * xi)
        x[i] = xi
    # Dernier élément
    xn = 1 + (n - 1) * h
    b[n - 2] = 1 - h / (2 * xn)
    x[n - 1] = xn

    l, v = lutri(a, b, c)
    y = descente(l, z)
    u = remontee(v, c, y)

    return x, np.array(u), [b, a, c]


def iterate_poisson_boltzmann_differences_finies(u, n, mu):
    """
        Résolution de l'équation de Poisson-Boltzmann avec la méthode des approximations successives

        On calcule u(k+1) à partir de u(k).

        Sortie : x vecteur réel de dimension n, u vecteur solution de dimension n
    """
    h = 10 / n
    # overflow avec np.sinh
    g = lambda x: math.sinh(x) - x

    # Initialisation
    x = [1] * n
    b, a, c, z = [0] * (n - 1), [-(2 + h ** 2)] * n, [0] * (n - 1), [0] * n
    c[0], z[0] = 2, (h ** 2) * g(u[0]) + mu * h * (h - 2)

    # Attribution
    for i in range(1, n - 1):
        xi = 1 + i * h
        b[i - 1] = 1 - h / (2 * xi)
        c[i] = 1 + h / (2 * xi)
        z[i] = (h ** 2) * g(u[i])
        x[i] = xi

    # Dernier élément
    xn = 1 + (n - 1) * h
    b[n - 2] = 1 - h / (2 * xn)
    z[n - 1] = (h ** 2) * g(u[n - 1])
    x[n - 1] = xn

    # Calcule de uk+1
    l, v = lutri(a, b, c)
    y = descente(l, z)
    u_suivant = remontee(v, c, y)

    return x, np.array(u_suivant), np.array(z)


def iterate_poisson_boltzmann_newton(u, n, mu):
    """
        Résolution de l'équation de Poisson-Boltzmann avec la méthode de Newton

        On calcule u(k+1) à partir de u(k).

        Sortie : x vecteur réel de dimension n, u vecteur solution de dimension n, F vecteur de la fonction
    """
    h = 10 / n
    # Initialisation
    x = [1] * n
    b, a, c, F = [0] * (n - 1), -2 - (h ** 2) * np.cosh(u), [0] * (n - 1), [0] * n
    c[0], F[0] = 2, -2 * u[0] - (h ** 2) * np.sinh(u[0]) + 2 * u[1] + mu * h * (2 - h)
    # Attribution
    for i in range(1, n - 1):
        xi = 1 + i * h
        bi = 1 - h / (2 * xi)
        ci = 1 + h / (2 * xi)
        b[i - 1] = bi
        c[i] = ci
        F[i] = bi * u[i - 1] - 2 * u[i] - (h ** 2) * np.sinh(u[i]) + ci * u[i + 1]
        x[i] = xi
    # Dernier élément
    xn = 1 + (n - 1) * h
    bn = 1 - h / (2 * xn)
    b[n - 2] = bn
    F[n - 1] = bn * u[n - 2] - 2 * u[n - 1] - (h ** 2) * np.sinh(u[n - 1])
    x[n - 1] = xn

    # Calcule de uk+1
    F = np.array(F)

    inv_Jk = np.linalg.inv(tridiag([b, a, c]))
    u_suivant = u - inv_Jk.dot(F)

    return x, u_suivant, F


def solve_poisson_boltzmann(n, mu, klimit, solveur, function_ecart1):
    # Paramètres de la simualtion
    mu1, mu2 = 10e-12, 10e-9

    # Initialisation de u et calcul de A
    _, u, diags = solve_debye_huckel(n, mu)
    A = tridiag(diags)
    # Initialisation des écarts
    ecart1 = A.dot(u)
    ecart2 = u

    k = 0
    while not (
        k > klimit
        or (
            np.linalg.norm(ecart1, np.inf) < mu1
            and np.linalg.norm(ecart2, np.inf) < mu2
        )
    ):
        x, u_suivant, F = solveur(u, n, mu)
        ecart1 = function_ecart1(u_suivant, A, F)
        ecart2 = u_suivant - u
        # print(k, np.linalg.norm(ecart1, np.inf), np.linalg.norm(ecart2, np.inf))
        u = u_suivant
        k += 1

    return x, u, k


def solve_poisson_boltzmann_differences_finies(n, mu):
    return solve_poisson_boltzmann(
        n=n,
        mu=mu,
        klimit=200,
        solveur=iterate_poisson_boltzmann_differences_finies,
        function_ecart1=lambda u, A, F: A.dot(u) - F,
    )


def solve_poisson_boltzmann_newton(n, mu):

    def calcul_F(u, n=n, mu=mu):
        h = 10 / n
        # Initialisation
        F = [0] * n
        F[0] = -2 * u[0] - (h ** 2) * np.sinh(u[0]) + 2 * u[1] + mu * h * (2 - h)
        # Attribution
        for i in range(1, n - 1):
            xi = 1 + i * h
            bi = 1 - h / (2 * xi)
            ci = 1 + h / (2 * xi)
            F[i] = bi * u[i - 1] - 2 * u[i] - (h ** 2) * np.sinh(u[i]) + ci * u[i + 1]
        # Dernier élément
        xn = 1 + (n - 1) * h
        bn = 1 - h / (2 * xn)
        F[n - 1] = bn * u[n - 2] - 2 * u[n - 1] - (h ** 2) * np.sinh(u[n - 1])

        # Calcule de uk+1
        return np.array(F)

    return solve_poisson_boltzmann(
        n=n,
        mu=mu,
        klimit=50,
        solveur=iterate_poisson_boltzmann_newton,
        function_ecart1=lambda u, A, F: calcul_F(u),
    )