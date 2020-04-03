"""
    Paquetage pour factorisation LU
    Contient : Trois fonctions (lutri, descente, remontee)
"""

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

def main():
    b, a, c = [2, 6], [1, 10, 10], [4, 5]
    l, v = lutri(a, b, c)
    print(l, v)
    y = descente(l, [3, 3, 3])
    print(y)
    x = remontee(v, c, [3, 3, 3])
    print(x)

    b, a, c = [-4, -3, -2, 2], [-2, 5, -1, 4, -2], [1, 2, -1, 1]
    l, v = lutri(a, b, c)
    assert(l == [2.0, -1.0, -2.0, 1.0] and v == [-2, 3.0, 1.0, 2.0, -3.0])
    print(l, v)

main()
