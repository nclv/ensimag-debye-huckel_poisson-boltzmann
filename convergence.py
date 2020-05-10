#!/usr/bin/python3
# coding: utf-8

"""
convergence.py : trouver le mu limite
"""


def find_mu_limit(a, b, klimit, solveur, epsilon=10e-6):
    debut = a
    fin = b
    ecart = fin - debut
    while ecart > epsilon:
        m = (debut + fin) / 2
        _, _, k = solveur(mu=m)
        # print(m, k)
        if k > klimit:
            fin = m
        else:
            debut = m
        ecart = fin - debut
    return m
