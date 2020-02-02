from py_ecc import optimized_bls12_381 as bls12_381
from py_ecc.optimized_bls12_381 import FQ, FQ2, FQ12, G1, G2, G12, add, neg, multiply, eq

from finitefield.finitefield import FiniteField
from finitefield.polynomial import polynomialsOver

from polynomial_evalrep import get_omega, polynomialsEvalRep, RowDictSparseMatrix

from babysnark_opt import *

# BLS12_381 group order
Fp = FiniteField(52435875175126190479447740508185965837690552500527637822603658699938581184513,1)  # (# noqa: E501)
Poly = polynomialsOver(Fp)
Fp.__repr__ = lambda self: hex(self.n)[:15] + "..." if len(hex(self.n))>=15 else hex(self.n)


omega_base = get_omega(Fp, 2**32, seed=0)
mpow2 = 128  # nearest power of 2 greather than m, the number of constraints
omega = omega_base ** (2**32 // mpow2)


def generate_solved_instance(m, n):
    """
    Generates a random circuit and satisfying witness
    U, V, W, (stmt, wit)
    """
    # Generate a, U
    a = np.array([random_fp() for _ in range(n)])
    U = random_sparse_matrix(m, n)
    V = random_sparse_matrix(m, n)
    W = random_sparse_matrix(m, n)

    # Normalize U to satisfy constraints
    UaVa = U.dot(a) * V.dot(a)
    print(UaVa)
    Wa = W.dot(a)
    print(Wa)
    for (i,j), val in W.items():
        if(UaVa[i]==0):
            W[i,j] = 0
        else:
            W[i,j] *= Wa[i].inverse()
            W[i,j] *= UaVa[i]

    assert((U.dot(a) * V.dot(a) == W.dot(a)).all())
    Ud = U.to_dense()
    Vd = V.to_dense()
    Wd = W.to_dense()
    assert((Ud.dot(a) * Vd.dot(a) == Wd.dot(a)).all())
    return U, V, W, a
#-
# Example
m, n = 10, 12 
U, V, W, a = generate_solved_instance(m, n)

#| # Baby SNARK with quasilinear overhead

# Setup
def groth16_setup(U, V, W, n_stmt):
    (m, n) = U.shape
    assert n_stmt < n

    # Generate roots for each gate
    # TODO: Handle padding?
    global mpow2, omega
    mpow2 = m
    assert mpow2 & mpow2 - 1 == 0, "mpow2 must be a power of 2"
    omega = omega_base ** (2**32 // mpow2)
    PolyEvalRep = polynomialsEvalRep(Fp, omega, mpow2)

    global ROOTS
    if len(ROOTS) != m:
        ROOTS.clear()
        ROOTS += [omega**i for i in range(m)]

    # Generate polynomials u from columns of U
    Us = [PolyEvalRep((), ()) for _ in range(n)]
    Vs = [PolyEvalRep((), ()) for _ in range(n)]
    Ws = [PolyEvalRep((), ()) for _ in range(n)]

    for (i,k), y in U.items():
        x = ROOTS[i]
        Us[k] += PolyEvalRep([x],[y])
    for (i,k), y in V.items():
        x = ROOTS[i]
        Vs[k] += PolyEvalRep([x],[y])
    for (i,k), y in W.items():
        x = ROOTS[i]
        Ws[k] += PolyEvalRep([x],[y])


    # Trapdoors
    global alpha, beta, gamma, delta, tau
    
    alpha = random_fp()
    beta  = random_fp()
    gamma = random_fp()
    delta = random_fp()
    tau   = random_fp()

    t = vanishing_poly(omega, m)
    # CRS elements
    CRS = [G * alpha, G * beta, G * gamma, G * delta] + \
          [G * (tau ** i) for i in range(m)] + \
          [G * ((beta * Us[i](tau) + alpha * Vs[i](tau) + Ws[i](tau))/gamma) for i in range(n_stmt)] + \
          [G * ((beta * Us[i](tau) + alpha * Vs[i](tau) + Ws[i](tau))/delta) for i in range(n_stmt,n)] + \
          [G * (tau ** i * t(tau)) for i in range(m-1)]

    return CRS


# Prover
def groth16_prover(U, V, W, CRS, n_stmt,  a):
    (m, n) = U.shape
    assert n == len(a)
    assert len(CRS) == 4 + m + n_stmt + (n-n_stmt) + (m-1)

    Alpha, Beta, Gamma, Delta = CRS[: 4]
    Taus = CRS[4: m+4]
    UVWs = CRS[m+4+n_stmt: m+4+n_stmt +(n-n_stmt)]
    tTaus = CRS[-(m-1): ]
    # UVWs = CRS[-((m-1) + (n-n_stmt)): -(m-1)]
    # tTaus = CRS[-(m-1): ]

    # Target is the vanishing polynomial
    mpow2 = m
    assert mpow2 & mpow2 - 1 == 0, "mpow2 must be a power of 2"
    omega  = omega_base ** (2**32 // mpow2)
    omega2 = omega_base ** (2**32 // (2*mpow2))
    PolyEvalRep = polynomialsEvalRep(Fp, omega, mpow2)
    t = vanishing_poly(omega, mpow2)
    
    # 1. Find the polynomial h(X)S

    # First compute A, B
    ux = PolyEvalRep((),())
    for (i,k), y in U.items():
        x = ROOTS[i]
        ux += PolyEvalRep([x], [y]) * a[k]
    
    uxx = U.dot(a)
    
    vx = PolyEvalRep((),())
    for (i,k), y in V.items():
        x = ROOTS[i]
        vx += PolyEvalRep([x], [y]) * a[k]
    
    wx = PolyEvalRep((),())
    for (i,k), y in W.items():
        x = ROOTS[i]
        wx += PolyEvalRep([x], [y]) * a[k]
    
    # Now we need to convert between representations to multiply and divide
    PolyEvalRep2 = polynomialsEvalRep(Fp, omega2, 2*mpow2)
    roots2 = [omega2**i for i in range(2*mpow2)]

    u2 = PolyEvalRep2.from_coeffs(ux.to_coeffs())
    v2 = PolyEvalRep2.from_coeffs(vx.to_coeffs())
    w2 = PolyEvalRep2.from_coeffs(wx.to_coeffs())

    p2 = u2 * v2 - w2
    p = p2.to_coeffs()
    
    # Find the polynomial h by dividing p / t
    h = PolyEvalRep2.divideWithCoset(p, t)    
    assert p == h * t


    # 2. Compute the A, B term
    # r = random_fp()
    # s = random_fp()

    A = Alpha + evaluate_in_exponent(Taus, ux.to_coeffs()) 
    B = Beta + evaluate_in_exponent(Taus, vx.to_coeffs())
    # A = Alpha + evaluate_in_exponent(Taus, ux.to_coeffs()) + (Delta * r)
    # B = Beta + evaluate_in_exponent(Taus, vx.to_coeffs()) + (Delta * s)

    # 3. Compute the C terms
    C = sum([UVWs[k-n_stmt] * a[k] for k in range(n_stmt, n)], G*0) + \
        evaluate_in_exponent(tTaus, h)

    return A, B, C


# Verifier
def groth16_verifier(U, V, W, CRS, a_stmt, Pi):
    (m, n) = U.shape
    (A, B, C) = Pi
    assert len(ROOTS) >= m
    n_stmt = len(a_stmt)

    # Parse the CRS
    Alpha, Beta, Gamma, Delta = CRS[: 4]
    Stmt = CRS[m+4: m+4+n_stmt]

    # Compute D
    D = sum([Stmt[k] * a_stmt[k] for k in range(n_stmt)], G * 0)

    # Check 1
    print('Checking (1)')
    # print(A.pair(B))
    # print(Alpha.pair(Beta) * D.pair(Gamma) * C.pair(Delta))
    assert A.pair(B) == Alpha.pair(Beta) * D.pair(Gamma) * C.pair(Delta) 

    return True

#-
if __name__ == '__main__':
    # Sample a problem instance
    print("Generating a QAP instance")
    n_stmt = 4
    m,n = (16, 6)
    U, V, W, a = generate_solved_instance(m, n)
    a_stmt = a[:n_stmt]
    print('U:', repr(U))
    print('V:', repr(U))
    print('W:', repr(U))
    print('a_stmt:', a_stmt)
    print('m x n:', m * n)

    # Setup
    print("Computing Setup (optimized)...")
    CRS = groth16_setup(U, V, W, n_stmt)
    print("CRS length:", len(CRS))

    # Prover
    print("Proving (optimized)...")
    A, B, C = groth16_prover(U, V, W, CRS, n_stmt, a)

    # Verifier
    print("[opt] Verifying (optimized)...")
    groth16_verifier(U, V, W, CRS, a[:n_stmt], (A, B, C))
