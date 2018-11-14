# load some libraries
import numpy as np
from scipy import integrate
from scipy.special import sph_harm
from scipy.special import erf

class sto:
    def __init__(self,N,alpha,d,R):
        self.N = N
        self.alpha = alpha
        self.d = d
        self.R = R

class atom:
    def __init__(self,pos,charge):
        self.pos = pos
        self.charge = charge

# compute S, the overlap matrix, and S^-1
def overlap(basis):
    M = len(basis)
    S = np.empty((M,M),dtype=float)
    for i in range(M):
        for j in range(M):
            S[i,j] = basis_overlap(basis[i].alpha,basis[i].d,basis[i].R,basis[j].alpha,basis[j].d,basis[j].R)
    Sinv = np.linalg.inv(S)
    return S, Sinv

def basis_overlap(alpha,dA,RA,beta,dB,RB):
    overlap = 0.0
    for i in range(len(alpha)):
        for j in range(len(beta)):
            overlap += alpha[i]**0.75*beta[j]**0.75*dA[i]*dB[j]*gaussian_overlap(alpha[i],RA,beta[j],RB)
    return overlap*(2.0)**1.5

def gaussian_overlap(alpha, RA, beta, RB):
    prefactor = ((alpha+beta))**(-1.5)
    diff = RA - RB
    dist2 = np.dot(diff,diff)
    return prefactor*np.exp(-alpha*beta/(alpha+beta)*dist2)    

# compute T, the kinetic energy matrix
def kinetic(basis):
    M = len(basis)
    T = np.empty((M,M),dtype=float)
    for i in range(M):
        for j in range(M):
            T[i,j] = basis_kinetic(basis[i].alpha,basis[i].d,basis[i].R,basis[j].alpha,basis[j].d,basis[j].R)
    return T

def basis_kinetic(alpha,dA,RA,beta,dB,RB):
    kineticE = 0.0
    for i in range(len(alpha)):
        for j in range(len(beta)):
            kineticE += (alpha[i]*beta[j])**0.75*dA[i]*dB[j]*gaussian_kinetic(alpha[i],RA,beta[j],RB)
    return kineticE*(2.0/np.pi)**1.5

def gaussian_kinetic(alpha, RA, beta, RB):
    AplusB = alpha + beta
    diff = RA - RB
    dist2 = np.dot(diff,diff)
    prefactor = alpha*beta/AplusB*(3-2*alpha*beta/AplusB*dist2)*(np.pi/AplusB)**1.5
    return prefactor*np.exp(-alpha*beta/AplusB*dist2)

# compute V, the potential energy matrix
def core_potential(basis,atoms):
    M = len(basis)
    V = np.empty((M,M),dtype=float)
    for i in range(M):
        for j in range(M):
            V[i,j] = basis_potential(basis[i].alpha,basis[i].d,basis[i].R,basis[j].alpha,basis[j].d,basis[j].R,atoms)
    return V

def basis_potential(alpha,dA,RA,beta,dB,RB,atoms):
    potential = 0.0
    nAtoms = len(atoms)
    for atom in range(nAtoms):
        for i in range(len(alpha)):
            for j in range(len(beta)):
                prefactor = -atoms[atom].charge*(2.0/np.pi)**1.5*(alpha[i]*beta[j])**0.75*dA[i]*dB[j]
                potential += prefactor*gaussian_potential(alpha[i],RA,beta[j],RB,atoms[atom].pos)
    return potential

def gaussian_potential(alpha, RA, beta, RB, RC):
    AplusB = alpha + beta
    RP = (alpha*RA+beta*RB)/AplusB
    RPminusRC = RP - RC
    RPRC2 = np.dot(RPminusRC,RPminusRC)
    RAminusRB = RA - RB
    RARB2 = np.dot(RAminusRB,RAminusRB)
    if (RPRC2 == 0 and RARB2 ==0):
        return 2.0*np.pi/AplusB
    else:
        t = np.sqrt(AplusB*RPRC2)
        prefactor = 2.0*np.pi*0.5/AplusB*np.sqrt(np.pi)/t*erf(t)
        return prefactor*np.exp(-alpha*beta/AplusB*RARB2)

# Compute all two-electron integrals
def compute_twoE(basis):
    M = len(basis)
    twoE = np.empty((M,M,M,M),dtype=float)
    for i in range(M):
        for j in range(M):
            for k in range(M):
                for l in range(M):
                    twoE[i,j,k,l] = basis_2e(basis[i].alpha,basis[i].d,basis[i].R,basis[j].alpha,basis[j].d,basis[j].R,basis[k].alpha,basis[k].d,basis[k].R,basis[l].alpha,basis[l].d,basis[l].R)
    return twoE

# two electron integrals
def basis_2e(alpha,dA,RA,beta,dB,RB,gamma,dC,RC,delta,dD,RD):
    twoE = 0.0
    for i in range(len(alpha)):
        for j in range(len(beta)):
            for k in range(len(gamma)):
                for l in range(len(delta)):
                    prefactor = (alpha[i]*beta[j]*gamma[k]*delta[l])**0.75*dA[i]*dB[j]*dC[k]*dD[l]
                    twoE += prefactor*gaussian_2e(alpha[i],RA,beta[j],RB,gamma[k],RC,delta[l],RD)
    return twoE*(2.0/np.pi)**3  

def gaussian_2e(alpha,RA,beta,RB,gamma,RC,delta,RD):
    AplusB = alpha + beta
    # weighted average of RA and RB
    RP = (alpha*RA+beta*RB)/AplusB
    GplusD = gamma + delta
    # weighted average of RC and RD
    RQ = (gamma*RC+delta*RD)/GplusD
    RAminusRB = RA - RB
    RARB2 = np.dot(RAminusRB,RAminusRB)
    RCminusRD = RC - RD
    RCRD2 = np.dot(RCminusRD,RCminusRD)
    RPminusRQ = RP - RQ
    RPRQ2 = np.dot(RPminusRQ,RPminusRQ)
    denom = AplusB*GplusD*np.sqrt(AplusB+GplusD)
    prefactor = 2.0*np.pi**2.5/denom
    t = np.sqrt(AplusB*GplusD/(AplusB+GplusD)*RPRQ2)
    if (RPRQ2 !=0):
        prefactor *= 0.5*np.sqrt(np.pi)/t*erf(t)
    return prefactor*np.exp( -alpha*beta/AplusB*RARB2 - gamma*delta/GplusD*RCRD2)

# populate G matrix using two-electron integrals and density matrix
def compute_G(PT,Pab,twoE):
    M = Pab.shape[1]
    G = np.zeros((M,M),dtype=float)

    for i in range(M):
        for j in range(M):
            G[i,j] = 0.0
            for k in range(M):
                for l in range(M):
                    G[i,j] += PT[k,l]*twoE[i,j,l,k]-Pab[k,l]*twoE[i,k,l,j]
    return G

# create and populate density matrix
def constructDensityMat(C):
    M = C.shape[0]
    P = np.zeros((M,M),dtype=float)
    for i in range(M):
        for j in range(i,M):
            for a in range(M/2):
                P[i,j] += C[i,a]*C[j,a]
            P[j,i] = P[i,j]
    return P

# for case of h2 we know Cs:
def optimal_C(S):
    M = S.shape[0]
    C = np.empty((M,M),dtype=float)
    # in this case we know the answer so can set it to be
    C[0,0] = 1.0/np.sqrt(2*(1+S[0,1]))
    C[0,1] = 1.0/np.sqrt(2*(1-S[0,1]))
    C[1,0] = C[0,0]
    C[1,1] = -C[0,1]
    return C

# compute total energy for HF
def total_energy(Fa,Fb,Hcore,PT,Pa,Pb,atoms):
#    e,C = np.linalg.eig(np.dot(Sinv,F))
#    P = constructDensityMat(C)
    M = Fa.shape[1]
    Etotal = 0.0
    for i in range(M):
        for j in range(M):
            Etotal += PT[j,i]*Hcore[i,j]+Pa[j,i]*Fa[i,j]+Pb[j,i]*Fb[i,j]
    Etotal*=0.5
    nAtoms = len(atoms)
    for atom1 in range(nAtoms-1):
        for atom2 in range(atom1+1,nAtoms):
            diff = atoms[atom1].pos - atoms[atom2].pos
            dist = np.linalg.norm(diff)
            Etotal += atoms[atom1].charge*atoms[atom2].charge/dist
    return Etotal

def C_u(S,Hcore,twoE):
    M = S.shape[0]
    Ca = np.empty((M,M),dtype=float)
    Cb = np.empty((M,M),dtype=float)
    C = np.empty((M,M),dtype=float)
    C[0,0] = 1.0/np.sqrt(2*(1+S[0,1]))
    C[0,1] = 1.0/np.sqrt(2*(1-S[0,1]))
    C[1,0] = C[0,0]
    C[1,1] = -C[0,1]
    h11 = 0.0
    h22 = 0.0
    for i in range(M):
        for j in range(M):
            h11 += C[i,0]*C[j,0]*Hcore[i,j]
            h22 += C[i,1]*C[j,1]*Hcore[i,j]
    J11 = 0.0
    J12 = 0.0
    J22 = 0.0
    K12 = 0.0
    for i in range(M):
        for j in range(M):
            for k in range(M):
                for l in range(M):
                    J11 += C[i,0]*C[j,0]*C[k,0]*C[l,0]*twoE[i,j,k,l]
                    J12 += C[i,0]*C[j,0]*C[k,1]*C[l,1]*twoE[i,j,k,l]
                    J22 += C[i,1]*C[j,1]*C[k,1]*C[l,1]*twoE[i,j,k,l]
                    K12 += C[i,0]*C[j,1]*C[k,1]*C[l,0]*twoE[i,j,k,l]
    eta = (h22-h11+J22-J12+2*K12)/(J11+J22-2*J12+4*K12)
    if ( eta >= 0.0 and eta <= 1.0):
        sin2theta = 1.0 - eta
        cosTheta = np.sqrt(eta)
        sinTheta = np.sqrt(sin2theta)
        Ca[0,0] = (2*(1.0+S[0,1]))**-0.5*cosTheta + (2*(1.0-S[0,1]))**-0.5*sinTheta
        Ca[1,0] = (2*(1.0+S[0,1]))**-0.5*cosTheta - (2*(1.0-S[0,1]))**-0.5*sinTheta
        norm = np.linalg.norm(Ca[:,0])
#        Ca[0,0] /= norm
#        Ca[1,0] /= norm
        Ca[0,1] = (2*(1.0-S[0,1]))**-0.5*cosTheta - (2*(1.0+S[0,1]))**-0.5*sinTheta
        Ca[1,1] = -(2*(1.0-S[0,1]))**-0.5*cosTheta - (2*(1.0+S[0,1]))**-0.5*sinTheta
        norm = np.linalg.norm(Ca[:,1])
#        Ca[0,1] /= norm
#        Ca[1,1] /= norm
        Cb[0,0] = Ca[1,0]
        Cb[1,0] = Ca[0,0]
        Cb[0,1] = (2*(1.0-S[0,1]))**-0.5*cosTheta + (2*(1.0+S[0,1]))**-0.5*sinTheta
        Cb[1,1] = -(2*(1.0-S[0,1]))**-0.5*cosTheta + (2*(1.0+S[0,1]))**-0.5*sinTheta
        norm = np.linalg.norm(Cb[:,1])
#        Cb[0,1] /= norm
#        Cb[1,1] /= norm
    else:
        Ca[0,0] = 1.0/np.sqrt(2*(1+S[0,1]))
        Ca[0,1] = 1.0/np.sqrt(2*(1-S[0,1]))
        Ca[1,0] = Ca[0,0]
        Ca[1,1] = -Ca[0,1]
        Cb = Ca
    return Ca, Cb, eta




