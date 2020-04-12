"""
    Paquetage pour factorisation LU
    Contient : Trois fonctions (lutri, descente, remontee)
"""

import matplotlib.pyplot as plt
import numpy as np

def lutri(a, b, c):
    """
        Effectue une factorisation LU d'une matrice A tridiagonale
        Entrée : a un vecteur réel de dimension n (diagonale de A)
                 b un vecteur réel de dimension n - 1 (sous-diagonale de A)
                 c un vecteur réel de dimension n - 1 (sur-diagonale de A)
        Sortie : [l, v] : [sous-diagonale de L vecteur réel de dimension n - 1, 
                           diagonale de U vecteur réel de dimension n]
    """
    n = len(a)
    l, v = [], [a[0]]
    for i in range(1, n):
        l.append(b[i - 1] / v[i - 1])
        v.append(a[i] - l[i - 1] * c[i - 1])
    return [l, v]    

def descente(l, z):
    """
        Effectue l'étape de descente lors de la factorisation LU d'une matrice A tridiagonale
        On cherche à résoudre Ly = z
        Entrée : l un vecteur réel de dimension n - 1
                 z un vecteur réel de dimension n
        Sortie : y un vecteur réel de dimension n
    """
    n = len(z)
    y = [z[0]]
    for i in range(1, n):
        y.append(z[i] - l[i - 1] * y[i - 1])
    return y

def remontee(v, c, y):
    """
        Effectue l'étape de remontée lors de la factorisation LU d'une matrice A tridiagonale
        On cherche à résoudre Ux = y
        Entrée : v un vecteur réel de dimension n
                 c un vecteur réel de dimension n - 1
                 y un vecteur réel de dimension n
        Sortie : x un vecteur réel de dimension n
    """
    n = len(y)
    x = [y[n - 1] / v[n - 1]]
    for i in range(n - 2, -1, -1):
        x.insert(0, (y[i] - c[i] * x[0]) / v[i])
    return x

def resoudre_systeme(n, mu, z):
    h = 10 / n
    #mu = 1
    a = [-(2 + h ** 2) for _ in range(n)]
    b = [1 - h / (2 * (1 + i * h)) for i in range(1, n)]
    c = [2] + [1 + h / (2 * (1 + i * h)) for i in range(1, n - 1)]
    #z = [mu * h * (h - 2)] + [0 for _ in range(n - 1)]
    [l, v] = lutri(a, b, c)
    y = descente(l, z)
    u = remontee(v, c, y)
    return u

def plot_echantillon_q5(n):
    """
        Réponse à la question 5
    """
    try:
        mu = 1
        # ISSUE : variable h appelée avant d'avoir été assignée
        z = [mu * h * (h - 2)] + [0 for _ in range(n -1)]
        u = resoudre_systeme(n, mu, z)
        h = 10 / n
        xi = [1 + i * h for i in range(n)]
        y = np.array(u)
        x = np.array(xi)
        plt.plot(x, y)
        plt.show()
    except KeyboardInterrupt:
        print("FIN DU PROGRAMME")

def plot_echantillon_q6(n):
    """
        Réponse à la question 6
    """
    try:
        suite_u0 = []
        mu = 1
        # ISSUE : variable h appelée avant d'avoir été assignée
        z = [mu * h * (h - 2)] + [0 for _ in range(n -1)]
        for i in range(1, n + 1):
            u = resoudre_systeme(i, mu, z)
            suite_u0.append(u[0])
        yh = [10 / i for i in range(1, n + 1)]
        yh.reverse()
        suite_u0.reverse()
        y = np.array(yh)
        x = np.array(suite_u0)
        plt.plot(x, y)
        plt.show()
    except KeyboardInterrupt:
        print("FIN DU PROGRAMME")

def calculer_u(u, mu, n):
    """
        Calcule u(k+1) en fonction de u(k)
    """
    h = 10 / n
    z = [(h ** 2) * (np.sinh(u[0]) - u[0]) + mu * h * (h - 2)] + [(h ** 2) * (np.sinh(i) - i) for i in u[1:]]
    u_suivant = resoudre_systeme(n, mu, z)
    return u_suivant

def plot_echantillon_q8(mu, n):
    eta1 = 10 ** (-12)
    eta2 = 10 ** (-9)
    k = 1
    h = 10 / n
    a = [-(2 + h ** 2) for _ in range(n)]
    b = [1 - h / (2 * (1 + i * h)) for i in range(1, n)]
    c = [2] + [1 + h / (2 * (1 + i * h)) for i in range(1, n - 1)]
    z = [mu * h * (h - 2)] + [0 for _ in range(n - 1)] 
    [l, v] = lutri(a, b, c)
    y = descente(l, z)
    u = remontee(v, c, y) #Calcul de u0 JUSQUA LA C'EST OK
    tmp_u = u
    while True:
        u_suivant = calculer_u(tmp_u, mu, n)
        a_modifie = [elt1 * elt2 for elt1, elt2 in zip(a, u_suivant)]
        b_modifie = [elt1 * elt2 for elt1, elt2 in zip(b, u_suivant[:len(u_suivant) - 1])]
        c_modifie = [elt1 * elt2 for elt1, elt2 in zip(c, u_suivant[1:])] #JUSQUE LA CA SEMBLE OK
        #print(a_modifie)
        #print(c_modifie)
        au = [a_modifie[0] + c_modifie[0]] + [elt1 + elt2 + elt3 for elt1, elt2, elt3 in zip(b_modifie[:len(b_modifie) - 1], a_modifie[1:len(a_modifie) - 1], c_modifie[1:])] + [b_modifie[-1] + a_modifie[-1]]
        g = [(h ** 2) * (np.sinh(u_suivant[0]) - u_suivant[0]) + mu * h * (h - 2)] + [(h ** 2) * (np.sinh(i) - i) for i in u_suivant[1:]]
        norme1 = [elt1 - elt2 for elt1, elt2 in zip(au, g)]
        norme2 = [elt1 - elt2 for elt1, elt2 in zip(u_suivant, tmp_u)]
        if (k > 200 or np.linalg.norm(np.array(norme1)) < eta1 and np.linalg.norm(np.array(norme2)) < eta2):
            print("K = {}".format(str(k)))
            print("Norme 1 = {}".format(str(np.linalg.norm(np.array(norme1)))))
            print("Norme 2 = {}".format(str(np.linalg.norm(np.array(norme2)))))
            break
        k += 1
        tmp_u = u_suivant
    return k

def main():
    #b, a, c = [2, 6], [1, 10, 10], [4, 5]
    #l, v = lutri(a, b, c)
    #print(l, v)
    #y = descente(l, [3, 3, 3])
    #print(y)
    #x = remontee(v, c, [3, 3, 3])
    #print(x)

    #b, a, c = [-4, -3, -2, 2], [-2, 5, -1, 4, -2], [1, 2, -1, 1]
    #l, v = lutri(a, b, c)
    #assert(l == [2.0, -1.0, -2.0, 1.0] and v == [-2, 3.0, 1.0, 2.0, -3.0])
    #print(l, v)
    k = plot_echantillon_q8(7, 1000)
    print(k)


main()
