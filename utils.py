#!/usr/bin/python3
# coding: utf-8

"""
utils.py : //
"""

import numpy as np


def tridiag(diags, k1=-1, k2=0, k3=1):
    """Retourne une matrice tridiagonale."""
    b, a, c = diags
    return np.diag(b, k1) + np.diag(a, k2) + np.diag(c, k3)
