#! /usr/bin/python3
# imports
from scipy import integrate
import math
import numpy as np
cimport numpy as np
cimport cython



#+------------------------------------------------------------------+
#PURPOSE  : Hilbert transform of DO for the Bethe disperion
#+------------------------------------------------------------------+

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef HilbertBethe(np.complex128_t z,double d):
  cdef np.complex128_t ret = 0.0
  ret = ( 2.0*z/d**2*(1.0-np.sqrt(1.0-d**2/z**2)))
  return ret

#+------------------------------------------------------------------+
#PURPOSE  : Hilbert transform of DO for the Dirac dispersion
#+------------------------------------------------------------------+

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef HilbertDirac(np.complex128_t z,double cutoff):
  cdef np.complex128_t ret = 0.0
  ret =  3 * (-z ** 2 * np.log((z - cutoff) / (z + cutoff)) - 2 * z*cutoff) / 2
  return ret


#+------------------------------------------------------------------+
#PURPOSE  : Hilbert transform of DO for the DiracPlusSquare dispersion
#+------------------------------------------------------------------+

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef HilbertDiracPlusSquare(np.complex128_t z,double d,double l):
  cdef np.complex128_t ret = 0.0
  znorm=(2.*((d-l)*l**2)+(2.*l**3)/3.)
  ret = (l**2*(np.log((z-l)/(z+l))+np.log((z+d)/(z-d)))-2.*z*l-z**2*np.log((z-l)/(z+l)))/znorm 
  return ret


#+------------------------------------------------------------------+
#PURPOSE  : Hilbert transform of DO for the Square dispersion
#+------------------------------------------------------------------+

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef HilbertSquare(np.complex128_t z,double d):
  cdef np.complex128_t ret = 0.0
  ret = (np.log((z+d)/(z-d))/(2*d))
  return ret

#+------------------------------------------------------------------+
#PURPOSE  : Kramers-Kronign transformation (by component)
#+------------------------------------------------------------------+

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef KK(np.ndarray[double, ndim=1] omega,np.ndarray[double, ndim=1] function):
  cdef int omegasize = len(omega)
  cdef np.ndarray[double, ndim=1] sum=np.zeros(omegasize)
  cdef int w=0
  cdef int x=1
  for  w in range(0, omegasize):
    for  x in range(1, omegasize-1):
      if omega[x] != omega[w]:
        sum[w] += function[x]/(omega[x]-omega[w])*(-omega[x-1]+omega[x+1])/2.0   
  return sum

@cython.boundscheck(False) 
@cython.wraparound(False) 
cpdef KK_single_value(np.ndarray[double, ndim=1] function,np.ndarray[double, ndim=1] omega, double w):
    cdef int omegasize = len(omega)
    cdef double sum=0
    cdef int x=1
    for  x in range(1, omegasize-1):
        if omega[x] != w:
            sum += function[x]/(omega[x]-w)*(-omega[x-1]+omega[x+1])/2.0   
    return sum


#+------------------------------------------------------------------+
#PURPOSE  : Calculate local G, from DOS or k-sum
#+------------------------------------------------------------------+

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef Gloc_dos(np.ndarray[np.complex128_t, ndim=1] omega, np.ndarray[np.complex128_t, ndim=3] sigma,para):
  cdef np.ndarray[np.complex128_t, ndim=3] gloc = np.zeros((len(omega),para.nbands,para.nbands),dtype=np.complex)
  for w in range(len(omega)):
    for iii in range(para.nbands):
      if para.model == "simple_dirac":
        gloc[w]= HilbertDirac(omega[w] +para.mu- sigma[w, iii, iii],para.d)
      if para.model == "dirac_square":
        gloc[w]= HilbertDiracPlusSquare(omega[w] +para.mu- sigma[w, iii, iii],para.d,para.l)
      if para.model == "bethe":
        gloc[w]= HilbertBethe(omega[w] +para.mu- sigma[w, iii, iii],para.d)
      if para.model == "square":
        gloc[w]= HilbertSquare(omega[w] +para.mu- sigma[w, iii, iii],para.d)
  return gloc

@cython.boundscheck(False)
@cython.wraparound(False) 
cpdef Gloc_ksum(np.ndarray[np.complex128_t, ndim=1] omega, np.ndarray[np.complex128_t, ndim=3] sigma, np.ndarray[np.complex128_t, ndim=3] hamiltonianList, para):
  cdef int k=0
  cdef int w=0
  cdef int numberOfKpoints= hamiltonianList.shape[0]
  cdef np.ndarray[np.complex128_t, ndim=3] gloc =  np.zeros((len(omega),para.nbands,para.nbands),dtype=np.complex)
  cdef np.ndarray[np.complex128_t,ndim=2]   id  =  np.identity(para.nbands,dtype=np.complex)
  for  w in range(0, len(omega)):
    for  k in range(0, numberOfKpoints):
      gloc[w] += np.linalg.inv((omega[w] + para.mu)*id- hamiltonianList[k] - sigma[w]) / numberOfKpoints
  return gloc


@cython.boundscheck(False)
@cython.wraparound(False) 
cpdef Gloc_change_mu(np.ndarray[np.complex128_t, ndim=3] gloc, para, double mu_old):
  cdef int k=0
  cdef int w=0
  cdef np.ndarray[np.complex128_t, ndim=3] new_gloc =  np.zeros((len(gloc[:]),para.nbands,para.nbands),dtype=np.complex)
  cdef np.ndarray[np.complex128_t,ndim=2]   id  =  np.identity(para.nbands,dtype=np.complex)
  for  w in range(0, len(gloc[:])):
      new_gloc[w] = np.linalg.inv(np.linalg.inv(gloc[w])+(para.mu-mu_old)*id)
  return new_gloc


#+------------------------------------------------------------------+
#PURPOSE  : Spectral function and response functions
#+------------------------------------------------------------------+



def CalcA(np.ndarray[np.complex128_t, ndim=1] omega, np.ndarray[np.complex128_t, ndim=3] sigma,np.ndarray[np.complex128_t, ndim=3] hamiltonianList, para, n):
  cdef int k=0
  cdef int w=0
  cdef double v=1.0
  cdef int numberOfKpoints= hamiltonianList.shape[0]
  cdef np.ndarray[double,ndim=1]   dEdK  =  np.full(numberOfKpoints, v)
  cdef np.ndarray[np.complex128_t,ndim=2]   id  =  np.identity(para.nbands,dtype=np.complex)
  cdef np.ndarray[double,ndim=2]   amatrix_1  =  np.identity(para.nbands,dtype=float)
  cdef np.ndarray[double,ndim=2]   amatrix_2  =  np.identity(para.nbands,dtype=float)
  cdef double sum=0.0
  #
  for w in range(0,len(omega)):
    for  k in range(0, numberOfKpoints):
      amatrix_1=-(1.0/np.pi)*np.imag(np.linalg.inv((omega[w] + para.mu)*id- hamiltonianList[k] - sigma[w]))
      amatrix_2=amatrix_1 #for the DC case, otherwise needs another loop
      sum += np.trace(dEdK[k]**2*np.matmul(amatrix_1,amatrix_2))*((omega[w].real*para.beta)**n)/(4*np.cosh(omega[w].real*para.beta/2)**2)
  return sum*(omega[1]-omega[0])




#+------------------------------------------------------------------+
#PURPOSE  : Calculate Self-Energy from P1,P2 (by component)
#+------------------------------------------------------------------+

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef sigmaIntegral(double w, np.ndarray[double, ndim=1] omega, np.ndarray[double, ndim=1] g0, np.ndarray[double, ndim=1] fermi, np.ndarray[double, ndim=1] p1, np.ndarray[double, ndim=1] p2):
  cdef int omegasize = len(omega)
  cdef int ind = 0
  cdef int x = 0
  cdef double integral = 0.0
  cdef double f = 0.0
  cdef double fi = 0.0
  cdef double g = 0.0
  cdef double omdiff = omega[x]-w
  cdef double omegastep = omega[1]-omega[0]
  for x in range(0, omegasize):
    omdiff =w- omega[x]
    ind = int(omdiff / omegastep + omegasize / 2.0)
    if ind < 0:
      f = 1.0
      g = 0.0
      fi = 0.0
    elif ind >= omegasize:
      f = 0.0
      g = 0.0
      fi = 1.0
    else:
      f = fermi[ind]
      g = g0[ind]
      fi = 1.0 - f
    integral += (f * p2[x] + p1[x] * fi) * g * omegastep
  return integral



#+------------------------------------------------------------------+
#PURPOSE  : Calculate P1, P2 (by component)
#+------------------------------------------------------------------+

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef CalcP(np.ndarray[double, ndim=1] omega, np.ndarray[double, ndim=1] g0,np.ndarray[double, ndim=1] fermi):
  cdef int omegasize = len(omega)
  cdef int w = 0
  cdef int x = 0
  cdef int ind = 0
  cdef double f = 0.0
  cdef double fi = 0.0
  cdef double g = 0.0
  cdef double omdiff = 0.0
  cdef double omegastep = omega[1]-omega[0]
  cdef np.ndarray[double, ndim=1] p1 = np.zeros(omegasize)
  cdef np.ndarray[double, ndim=1] p2 = np.zeros(omegasize)
  for  w in range(0, omegasize):
    for  x in range(0, omegasize):
      f = 0.0
      fi= 0.0
      g = 0.0
      omdiff= omega[x]-omega[w]
      ind = int(omdiff / omegastep+omegasize/2.0)
      if ind < 0:
        f=1.0
        fi=0.0
        g=0.0
      elif ind >= omegasize:
        f=0.0
        g=0.0
        fi=1.0-f
      else:
        f=fermi[ind]
        g=g0[ind]
        fi=1.0-f
      p1[w]+=(1.0-fermi[x])*f*g0[x]*g*omegastep
      p2[w]+= fermi[x]*fi*g0[x]*g *omegastep
  return p1,p2

cpdef CalcP_singleValue(np.ndarray[double, ndim=1] g0,np.ndarray[double, ndim=1] omega,np.ndarray[double, ndim=1] fermi,double w):
    cdef int omegasize = len(omega)
    cdef double p1=0
    cdef double p2=0
    cdef int x=0
    cdef double f=0.0
    cdef double fi=0.0
    cdef double g=0.0
    cdef double omdiff= 0.0
    cdef double omegastep = omega[1]-omega[0]
    cdef int ind = 0
    for  x in range(0, omegasize):
        f=0.0
        fi=0.0
        g=0.0
        omdiff= omega[x]-w
        ind = int(omdiff / omegastep+omegasize/2.0)
        # get f(x-w) and make sure we are not running out of bounds
        if ind < 0:
            f=1.0
            fi=0.0
            g=0.0
        else:
            if ind >= omegasize:
                f=0.0
                g=0.0
                fi=1.0-f
            else:
                f=fermi[ind]
                g=g0[ind]
                fi=1.0-f
        p1+=(1.0-fermi[x])*f*g0[x]*g*omegastep
        p2+= fermi[x]*fi*g0[x]*g *omegastep   
    return p1,p2


#+------------------------------------------------------------------+
#PURPOSE  : Determine chemical potential (scalar)
#+------------------------------------------------------------------+

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef muSearch_dos(np.ndarray[np.complex128_t, ndim=1] omega, np.ndarray[double, ndim=1] fermi, np.ndarray[np.complex128_t, ndim=3] sigma, para):
  cdef int omegasize = len(omega)
  cdef np.ndarray[np.complex128_t, ndim=3] gloc = np.zeros((len(omega),para.nbands,para.nbands),dtype=np.complex)
  cdef double n_test=0
  cdef double mu_old=0
  cdef searching = True
  cdef int counter = 0
  cdef double pi =np.pi
  cdef double incr=0.2
  cdef double errToZero=0.00000001
  cdef double divisor =2
   

  gloc=Gloc_dos(omega,sigma,para)
  n_test=0
  for i in range(0,para.nbands):
    n_test+= integrate.trapz(np.imag(gloc[:, i, i] * fermi), np.real(omega)) / (-pi)

  while searching == True:
    if abs(n_test - para.n_aim) < errToZero:
      searching = False
    else:
      mu_old=para.mu
      if n_test > para.n_aim:
        para.mu -= incr
        gloc = Gloc_dos(omega,sigma,para)
        n_test=0
        for i in range(0,para.nbands):
          n_test+= integrate.trapz(np.imag(gloc[:, i, i] * fermi), np.real(omega)) / (-pi)
        if n_test < para.n_aim:
          incr = abs(para.mu-mu_old)/2
      else:
        para.mu += incr
        gloc = Gloc_dos(omega,sigma,para)
        n_test=0
        for i in range(0,para.nbands):
          n_test+= integrate.trapz(np.imag(gloc[:, i, i] * fermi), np.real(omega)) / (-pi)
        if n_test  > para.n_aim:
          incr = abs(para.mu-mu_old)/2
    counter += 1
    if counter > 150:
      searching = False
      print("DID NOT FIND mu !!!", n_test, para.n_aim, para.mu, incr)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef muSearch_ksum(np.ndarray[np.complex128_t, ndim=1] omega, np.ndarray[double, ndim=1] fermi, np.ndarray[np.complex128_t, ndim=3] sigma, np.ndarray[np.complex128_t, ndim=3] hamiltonianList, para):
  cdef int omegasize = len(omega)
  cdef np.ndarray[np.complex128_t, ndim=3] gloc = np.zeros((len(omega),para.nbands,para.nbands),dtype=np.complex)
  cdef double n_test=0
  cdef double mu_old=0
  cdef searching = True
  cdef int counter = 0
  cdef double pi =np.pi
  cdef double incr=0.2
  cdef double errToZero=0.00000001
  cdef double divisor =2

  gloc=Gloc_ksum(omega,sigma,hamiltonianList,para)
  n_test=0
  for i in range(0,para.nbands):
    n_test+= integrate.trapz(np.imag(gloc[:, i, i] * fermi), np.real(omega)) / (-pi)

  while searching == True:
    if abs(n_test - para.n_aim) < errToZero:
      searching = False
    else:
      mu_old=para.mu
      if n_test > para.n_aim:
        #print(n_test, para.n_aim,"Decreasing mu")
        para.mu -= incr
        #gloc=Gloc_ksum(omega,sigma,hamiltonianList,para) #OLD
        gloc=Gloc_change_mu(gloc, para, mu_old)           #NEW FASTER
        n_test=0
        for i in range(0,para.nbands):
          n_test+= integrate.trapz(np.imag(gloc[:, i, i] * fermi), np.real(omega)) / (-pi)
        if n_test < para.n_aim:
          incr = abs(para.mu-mu_old)/2
      else:
        #print(n_test, para.n_aim,"Increasing mu")
        para.mu += incr
        #gloc=Gloc_ksum(omega,sigma,hamiltonianList,para) #OLD
        gloc=Gloc_change_mu(gloc, para, mu_old)           #NEW FASTER
        n_test=0
        for i in range(0,para.nbands):
          n_test+= integrate.trapz(np.imag(gloc[:, i, i] * fermi), np.real(omega)) / (-pi)
        if n_test  > para.n_aim:
          incr = abs(para.mu-mu_old)/2
    #print("Step",counter,para.mu,n_test,para.n_aim)
    counter += 1
    if counter > 150:
      searching = False
      print("DID NOT FIND mu !!!", n_test, para.n_aim, para.mu, incr)


