import gaussxw
import numpy as np
import matplotlib.pyplot as plt
import math

def basis(x, n, L):
#Basis function |bn>
    return math.sqrt(2/L) * np.sin((n+1)*x*np.pi/L)
def d2basis(x, n, L):
#Second derivative of basis function
    return -(n+1)*(n+1)*np.pi*np.pi/(L*L)*math.sqrt(2/L) \
                * np.sin((n+1)*x*np.pi/L)

def Vho(x, L, w):
#Harmonic Oscillator potential.
    return 0.5 * w * w * (x-0.5*L) * (x-0.5*L)

def Vfsw(x, L, width, V0):
#Finite square well potential. x can be array or scalar
    a = 0.5*(L-width)
    b = 0.5*(L+width)
    try:
        v = np.zeros(x.shape)
        v[(x>a) * (x<b)] = V0
    except AttributeError:
        if x > a and x < b:
            v = V0
        else:
            v = 0.0
        
    return v

def Vcol(x, L, width, V0):
#Coloumb barrier potential. x can be array or scalar

    try:
        v = np.zeros(x.shape)
        v = 1.0/x
        v[x<width] = V0
    except AttributeError:
        if x < width:
            v = N0
        else:
            v = 1.0/x
    return v

def matrix_element(N, Nx, L, V, *args):
#Calculates matrix representation of the Hamiltonian:
# H[i,j] = <bi|H|bj>

    H = np.empty((N,N), dtype=np.complex)

    x, weight = gaussxw.gaussxwab(Nx, 0.0, L)

    b = np.empty((N,len(x)), dtype=np.complex)
    d2b = np.empty((N,len(x)), dtype=np.complex)
    for i in range(N):
        b[i,:] = basis(x, i, L)
        d2b[i,:] = d2basis(x, i, L)

    Hb = -0.5*d2b[:,:] + V(x,L,*args)[None,:]*b[:,:]
    for i in range(N):
        H[i,:] = (weight[None,:] * np.conj(b[i])[None,:] * Hb[:,:]
                        ).sum(axis=1)

    return H

def ortho_test(N, Nx, L, *args):
# Calculate <bi|bj>. should be Identity if |bi> orthonormal

    bbmn = np.empty((N,N), np.complex)
    
    x, weight = gaussxw.gaussxwab(Nx, 0.0, L)

    b = np.empty((N,Nx),np.complex)
    for i in range(N):
        b[i,:] = basis(x,i,L)

    for i in range(N):
        bbmn[i,:] = (weight[None,:] * np.conj(b[i])[None,:]
                        * b[:,:]).sum(axis=1)

    return bbmn

def evolve_bad(N, Nx, NT, t, L, psi0, V, *args):
# Evolves the wave function psi0 in the potential V using a truncated
# Taylor series.  Bad Idea!
#
# psi0: a function that takes a single array argument psi0(x)
# V: a function that takes at least two arguments, the first an array 
#        V(x,L,*args)


    x, weight = gaussxw.gaussxwab(Nx, 0.0, L)

    c = np.empty(N, dtype=np.complex)
    for i in range(N):
        c[i] = (weight * np.conj(basis(x,i,L)) * psi0(x)).sum()

    Hmn = matrix_element(N, Nx, L, V, *args)

    Hc = c.copy()
    for i in range(1,NT):
        Hc = -1.0j * np.dot(Hmn, Hc) * t / i
        c += Hc

    psi = np.zeros(Nx, dtype=np.complex)
    for i in range(N):
        psi[:] += c[i] * basis(x, i, L)

    return x, psi, c

def evolve_good(N, Nx, t, L, psi0, V, *args):
# Evolves the wave function psi0 in the potential V using exact time-evolution
# operator calculated by diagonalizing H.  Good Idea!
#
# psi0: a function that takes a single array argument psi0(x)
# V: a function that takes at least two arguments, the first an array 
#        V(x,L,*args)

    print("Calculating c0")
    x, weight = gaussxw.gaussxwab(Nx, 0.0, L)
    
    b = np.empty((N,Nx), dtype=np.complex)
    for i in range(N):
        b[i,:] = basis(x,i,L)
    psi0x = psi0(x)

    c0 = (weight[None,:] * np.conj(b)[:,:] * psi0x[None,:]).sum(axis=1)

    print("Calculating H")
    Hmn = matrix_element(N, Nx, L, V, *args)
    print("Calculating E, B")
    En, B = np.linalg.eigh(Hmn)

    Hp = np.dot(B, np.dot(np.diag(En), B.T))

    B = B.astype(np.complex)
    En = En.astype(np.complex)
    psiL = []
    cL = []

    print(B.dtype)

    T = np.atleast_1d(t)

    for tt in T:
        print("Calculating ct: {0:f}".format(tt))

        D = np.exp(-1.0j*En*tt)

        ct = np.dot(B, D * np.dot(B.T, c0))

        psi = (ct[:,None] * b[:,:]).sum(axis=0)

        psiL.append(psi)
        cL.append(ct)

    return x, np.array(psiL), np.array(cL)

def gaussian_func(x0, sig, p):
# Returns a gaussian wavefunction that can be called with only a position 
# argument x.

    def f(x):
        return math.pow(2*np.pi*sig*sig, -0.25)\
                * np.exp(1.0j*p*x - (x-x0)*(x-x0)/(4*sig*sig))
    return f

def test(N, Nx, L, V, *args):
# Runs tests on a potential V. Plots the matrix of basis state inner products
# (should be identity), the Hamiltonian matrix elements (should be hermitian),
# the eigenvalues of H, and the first few stationary states.

    print("calculating orthogonality")
    bb = ortho_test(N, Nx, L)

    print("calculating Hmn")
    Hmn = matrix_element(N, Nx, L, V, *args)
    print("calculating Eigenvalues")
    En, B = np.linalg.eigh(Hmn)
    ind = np.argsort(En)

    En = En[ind]
    B = B[:,ind]

    X = np.linspace(0, L, Nx)
    b = np.empty((N,Nx))
    for i in range(N):
        b[i,:] = basis(X, i, L)

    psi0 = (B[:,0,None] * b[:,:]).sum(axis=0)
    psi1 = (B[:,1,None] * b[:,:]).sum(axis=0)
    psi2 = (B[:,2,None] * b[:,:]).sum(axis=0)
    psi3 = (B[:,3,None] * b[:,:]).sum(axis=0)

    NN = np.arange(N)
    E_inf = (NN+1)*(NN+1) * np.pi*np.pi / (2.0*L*L)
    E_ho = (NN+0.5)*w

    if len(args) > 1:
        Efsw_exact = fswEn(args[0], args[1])

        print(Efsw_exact)
        print(En[:Efsw_exact.shape[0]])

    print("plotting")
    fig, ax = plt.subplots(2,2)
    mat = ax[0,0].matshow(np.real(bb))
    fig.colorbar(mat,ax=ax[0,0])

    mat = ax[0,1].matshow(np.real(Hmn))
    fig.colorbar(mat,ax=ax[0,1])

    ax[1,0].plot(NN, E_inf, 'b-')
    ax[1,0].plot(NN, E_ho, 'r-')
    ax[1,0].plot(NN, En, 'k+')

    ax[1,1].plot(X, np.real(psi0))
    ax[1,1].plot(X, np.real(psi1))
    ax[1,1].plot(X, np.real(psi2))
    ax[1,1].plot(X, np.real(psi3))

def testEvolveBad(N, Nx, NT, t, L, x0, p, sig, V, *args):

    psi0_func = gaussian_func(x0, sig, p)

    x, psi_arr, c_arr = evolve_bad(N, Nx, NT, t, L, psi0_func, V, *args)

    psi0_arr = psi0_func(x)

    psi0_2 = np.abs(psi0_arr) * np.abs(psi0_arr)
    psi_2 = np.abs(psi_arr) * np.abs(psi_arr)

    fig, ax = plt.subplots()
    ax.plot(x, psi0_2, 'k+')
    ax.plot(x, psi_2, 'b+')

def testEvolveGood(N, Nx, t, L, x0, p, sig, V, *args):

    psi0_func = gaussian_func(x0, sig, p)

    x, psi_arr, c_arr = evolve_good(N, Nx, t, L, psi0_func, V, *args)

    psi0_arr = psi0_func(x)

    psi0_2 = np.abs(psi0_arr) * np.abs(psi0_arr)

    fig, ax = plt.subplots()
    ax.plot(x, psi0_2, 'k+')
    T = np.atleast_1d(t)
    for i in range(len(T)):
        psi_2 = np.abs(psi_arr[i]) * np.abs(psi_arr[i])
        ax.plot(x, psi_2)

def testTunnel(N, Nx, t, L, x0, p, sig, width, depth):

    psi0_func = gaussian_func(x0, sig, p)

    x, psi_arr, c_arr = evolve_good(N, Nx, t, L, psi0_func, Vcol, width, depth)

    psi0_arr = psi0_func(x)

    psi0_2 = np.abs(psi0_arr) ** 2

    T = np.atleast_1d(t)

    x,weight = gaussxw.gaussxwab(Nx,0.0,width)
    b = np.empty((N,Nx),np.complex)
    for i in range(N):
        b[i,:] = basis(x,i,L)

    P = np.empty(T.shape,np.float)
    for i,tt in enumerate(t):
        psi = (c_arr[i,:,None]*b[:,:]).sum(axis=0)
        psiR = np.real(psi)
        psiI = np.imag(psi)
        psi2 = psiR*psiR + psiI*psiI
        P[i] = (weight * psi2).sum()

    fig, ax = plt.subplots()
    ax.plot(T, P, 'k+')

def fswEn(width, V0):

    z0 = width*math.sqrt(-0.5*V0)
    N = int(2*z0/np.pi) + 1
    print(N, z0)

    En = np.empty(N)
    tol = 1.0e-10

    def solve_even_z(i, z0):
        a = 0.5*i*np.pi
        b = 0.5*(i+1)*np.pi
        z = 0.5*(a+b)
        if z > z0:
            z = 0.5*(a + z0)
        dz = np.inf
        while abs(dz) > tol:
            f = math.tan(z) - math.sqrt(z0*z0/(z*z)-1.0)
            sz = 1.0/math.cos(z)
            df = sz*sz + z0*z0/(z*z*z * math.sqrt(z0*z0/(z*z)-1.0))
            dz = -f/df
            if f > 0:
                b = z
            if f < 0:
                a = z
            z += dz
            if z < a or z > b:
                z = 0.5*(a+b)
        return z

    def solve_odd_z(i, z0):
        a = 0.5*i*np.pi
        b = 0.5*(i+1)*np.pi
        z = 0.5*(a+b)
        if z > z0:
            z = 0.5*(z0 + 0.5*i*np.pi)
        dz = np.inf
        while abs(dz) > tol:
            f = -1.0/math.tan(z) - math.sqrt(z0*z0/(z*z)-1.0)
            cz = 1.0/math.sin(z)
            df = cz*cz + z0*z0/(z*z*z * math.sqrt(z0*z0/(z*z)-1.0))
            dz = -f/df
            if f > 0:
                b = z
            if f < 0:
                a = z
            z += dz
            if z < a or z > b:
                z = 0.5*(a+b)
        return z

    for i in range(N):
        if i%2 == 0:
            z = solve_even_z(i, z0)
            En[i] = 2*z*z/(width*width) + V0
        else:
            z = solve_odd_z(i, z0)
            En[i] = 2*z*z/(width*width) + V0

    return En

if __name__ == "__main__":

    N = 400
    L = 50.0
    Nx = 2000
    
    w = 1.0
    #test(N, Nx, L, Vho, w)

    width = 5.0
    V0 = -10.0
    test(N, Nx, L, Vfsw, width, V0)

    NT = 800
    t = np.array([0.0,0.25,0.5,0.75,1.0]) * 2*np.pi/w
    x0 = L/3.0
    p = 0.0
    sig = 1.0
    #testEvolveBad(N, Nx, NT, t, L, x0, p, sig, Vho, w)
    testEvolveGood(N, Nx, t, L, x0, p, sig, Vho, w)
    
    x0 = 15
    p = -0.5
    sig = 1.0
    T = np.linspace(0.0, 100.0, 101)
    width = 0.5
    depth = -1.0
    #print(T)
    #testEvolveGood(N, Nx, T, L, x0, p, sig, Vcol, width, depth)

    testTunnel(N, Nx, T, L, x0, p, sig, width, depth)

    plt.show()
