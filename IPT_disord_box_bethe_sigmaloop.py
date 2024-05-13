#! /usr/bin/python3
# imports

from scipy import integrate
import math
import cmath
import numpy as np
import os
import sys
import time
import datetime
from multiprocessing import Process, Value, Array, Queue, Manager
import multiprocessing
from numpy.linalg import inv
from numpy import linalg as LA
from scipy.fftpack import fft, ifft
#import matplotlib.pyplot as plt
import pyximport; pyximport.install(language_level=3,setup_args={"include_dirs":np.get_include()})
import auxiliaries

sigma0 = np.matrix('1 0; 0 1')
sigma1 = np.matrix('0 1; 1 0')
sigma2 = np.matrix('0 -1j; 1j 0')
sigma3 = np.matrix('1 0; 0 -1')

#+------------------------------------------------------------------+
#PURPOSE  : Parameters class
#+------------------------------------------------------------------+



class Parameters:
    #stores all parameters
    def __init__(self, **params):
        self.d = params['d']
        self.l = params['l']
        self.nbands = params['nbands']
        self.ULOC= params['ULOC']
        self.UST= params['UST']
        self.mu= params['mu']
        self.omega_max = params['omega_max']
        self.omegasteps= params['omegasteps']
        self.ksum= params['ksum']
        self.delta=params['delta']
        self.omegastep=params['omegastep']
        self.beta = params['beta']
        self.mu_tilde = params['mu_tilde'] #needed for non-half-filling
        self.n_aim = params['n_aim']
        self.zeroT = params['zeroT']
        self.CounterTerm = params['CounterTerm']
        self.dopedMode = params['dopedMode']
        self.dopedMode_type = params['dopedMode_type']
        self.mix = params['mix']
        self.mix0= params['mix']
        self.loops = params['loops']
        self.fix_n = params['fix_n']
        self.debug = params['debug']
        self.threshold = params['threshold']
        self.min_loops = params['min_loops']
        self.adaptiveMixing = params['adaptiveMixing']
        self.outputN = params['outputN']
        self.omega_grid = params['omega_grid']
        self.model = params['model']
        self.mu_0 = params['mu_0']
        self.csi = params['csi']  # needed for disorder
    
    def Check(self):
        #check if parameters reasonable
        passed = True

        if (not (self.model in ["simple_dirac", "bethe", "dirac_square", "square"])) and not self.ksum:
            print("!!! Wrong model for DOS-based calculation :", self.model)
            passed = False

        if (not (self.model in ["dirac", "simple_dirac","simple_weyl","C4_transformed","V130","V130_1","V130_2","Weyl"])) and self.ksum:
            print("!!! Did not understand 'model' :", self.model)
            passed = False
        
        if not self.omega_grid == "linear":
            print("!!! Did not understand 'omega_grid' :", self.omega_grid)
            print("!!! Currently only 'linear' is implemented :", self.omega_grid)
            passed = False

        if self.omegasteps % 2 != 1: 
            print(" !!! Use odd number of frequencies, not", self.omegasteps)
            passed = False
        
        if self.dopedMode and self.dopedMode_type > 2:
            print("!!! 'dopedMode_type' must be '1 or '2'; got", self.dopedMode_type)
            passed = False

        if len(self.ULOC) != self.nbands:
            print("!!! dim ULOC different from nbands")
            passed = False

        if self.ksum and not self.nbands == len(hamiltonianOfK(0,0,0,self.model)):
            self.nbands=  len(hamiltonianOfK(0,0,0,self.model))
            print("!!! Wrong 'nbands', corrected to", self.nbands, "in order to match Hamltonian !!!")

        #check positivity
        var_list = [self.nbands, self.omega_max, self.omegasteps,self.delta,self.beta,self.n_aim,self.mix,self.threshold,self.loops,self.d, self.fix_n]
        name_list =["nbands", "omega_max","omegasteps","delta","beta","n_aim","mix","threshold","loops","d","fix_n"]
        for i in range(0,len(var_list)):         
            if var_list[i] < 0:          
                print("!!! expected positive vale for '", name_list[i], "'; got:",var_list[i] )
                passed = False
        
        if not passed:
            print("!!! stopping code execution because some parameters are problematic !!!")
            sys.exit(1)
        else:
            print("... parameters seem reasonable ...")

#+------------------------------------------------------------------+
#PURPOSE  : Read  and set parameters from file
#+------------------------------------------------------------------+

def readParameters(filename):
    print("--- reading parameters from", filename, "---")
    file = open(filename)
    parameterList = []
    #default values
    loops=10
    min_loops = 1
    threshold=0.1
    zeroT=False
    CounterTerm=False
    mix=0.4 
    adaptiveMixing=True
    readold=False
    number_of_threads = 8
    debug =False
    omega_max=8
    omegasteps =6001
    ksteps=5
    nbands=1
    cutoff =10000
    ksum = False
    d = 1 
    l = 0.5
    dopedMode=False
    dopedMode_type=1 
    outputN=False 
    mu_tilde =-0.0
    fix_n=False 
    n_aim=0.47
    model="simple_dirac"
    for line in file:
        parameterList.append(line)
    for i in range(0,len(parameterList)):
        parameter = parameterList[i]
        end_index = parameter.find("=")
        parameterToSet=""
        if end_index != -1:
            parameterToSet= parameter[0:end_index].strip()
        if( parameterToSet=="loops"):
            value = setParameter(parameter)
            if value != 0:
                loops= int(value.strip())
        if( parameterToSet=="min_loops"):
            value = setParameter(parameter)
            if value != 0:
                min_loops= int(value.strip())
        if( parameterToSet=="threshold"):
            value = setParameter(parameter)
            if value != 0:
                threshold= float(value.strip())
        if( parameterToSet=="zeroT"):
            value = setParameter(parameter)
            if value != 0:
                zeroT=False
                if value.strip() in ["True", "true", "1"]:
                    zeroT= True
        if( parameterToSet=="CounterTerm"):
            value = setParameter(parameter)
            if value != 0:
                CounterTerm=False
                if value.strip() in ["True", "true", "1"]:
                    CounterTerm= True
        if( parameterToSet=="mix"):
            value = setParameter(parameter)
            if value != 0:
                mix= float(value.strip())
        if( parameterToSet=="adaptiveMixing"):
            value = setParameter(parameter)
            if value != 0:
                adaptiveMixing=False
                if value.strip() in ["True", "true", "1"]:
                    adaptiveMixing= True
        if( parameterToSet=="readold"):
            value = setParameter(parameter)
            if value != 0:
                readold=False
                if value.strip() in ["True", "true", "1"]:
                    readold= True
        if( parameterToSet=="number_of_threads"):
            value = setParameter(parameter)
            if value != 0:
                number_of_threads= int(value.strip())
        if( parameterToSet=="debug"):
            value = setParameter(parameter)
            if value != 0:
                debug=False
                if value.strip() in ["True", "true", "1"]:
                    debug= True
        if( parameterToSet=="omega_max"):
            value = setParameter(parameter)
            if value != 0:
                omega_max= int(value.strip())
        if( parameterToSet=="omegasteps"):
            value = setParameter(parameter)
            if value != 0:
                omegasteps= int(value.strip())
        if( parameterToSet=="ksteps"):
            value = setParameter(parameter)
            if value != 0:
                ksteps= int(value.strip())
        if( parameterToSet=="nbands"):
            value = setParameter(parameter)
            if value != 0:
                nbands= int(value.strip())  
        if( parameterToSet=="cutoff"):
            value = setParameter(parameter)
            if value != 0:
                cutoff= float(value.strip())  
        if( parameterToSet=="ksum"):
            value = setParameter(parameter)
            if value != 0:
                ksum=False
                if value.strip() in ["True", "true", "1"]:
                    ksum= True   
        if( parameterToSet=="d"):
            value = setParameter(parameter)
            if value != 0:
                d= float(value.strip())
        if( parameterToSet=="l"):
            value = setParameter(parameter)
            if value != 0:
                l= float(value.strip())  
        if( parameterToSet=="dopedMode"):
            value = setParameter(parameter)
            if value != 0:
                dopedMode=False
                if value.strip() in ["True", "true", "1"]:
                    dopedMode= True   
        if( parameterToSet=="dopedMode_type"):
            value = setParameter(parameter)
            if value != 0:
                dopedMode_type= int(value.strip())     
        if( parameterToSet=="outputN"):
            value = setParameter(parameter)
            if value != 0:
                outputN=False
                if value.strip() in ["True", "true", "1"]:
                    outputN= True   
        if( parameterToSet=="mu_tilde"):
            value = setParameter(parameter)
            if value != 0:
                mu_tilde= float(value.strip())
        if( parameterToSet=="fix_n"):
            value = setParameter(parameter)
            if value != 0:
                fix_n=False
                if value.strip() in ["True", "true", "1"]:
                    fix_n= True     
        if( parameterToSet=="n_aim"):
            value = setParameter(parameter)
            if value != 0:
                n_aim= float(value.strip())  
        if( parameterToSet=="model"):
            value = setParameter(parameter)
            if value != 0:
                model= str(value.strip())                        
                                                                                                                 
    return loops,\
           min_loops,\
           threshold,\
           zeroT,\
           CounterTerm,\
           mix,\
           adaptiveMixing,\
           readold,\
           number_of_threads,\
           debug,\
           omega_max,\
           omegasteps,\
           ksteps,\
           nbands,\
           cutoff,\
           ksum,\
           d,\
           l,\
           dopedMode,\
           dopedMode_type,\
           outputN,\
           mu_tilde,\
           fix_n,\
           n_aim,\
           model


def setParameter(parameter):
    index = parameter.find("=")
    end = parameter.find("#")
    if index != -1:
        if end == -1:
            end = len(parameter)-1
        return parameter[index + 1:end]
    else:
        return 0
    
#+------------------------------------------------------------------+
#PURPOSE  : Time string format
#+------------------------------------------------------------------+

def time_string():
    #give nice format for time
    now = datetime.datetime.now()
    hour_str=   str(now.hour)
    if len(hour_str) == 1: hour_str = "0"+hour_str

    minute_str=   str(now.minute)
    if len(minute_str) == 1: minute_str = "0"+minute_str

    second_str=   str(now.second)
    if len(second_str) == 1: second_str = "0"+second_str

    return hour_str+":"+minute_str+":"+second_str


#+------------------------------------------------------------------+
#PURPOSE  : Define some Hamiltonians, returns H(k)
#+------------------------------------------------------------------+

def hamiltonianOfK(kx,ky,kz,model):
    hamiltonian=0
    if model == "dirac":
        dirac_1 = np.kron(sigma0, sigma3) * (math.cos(kx) + math.cos(ky) + math.cos(kz) - 2 - math.cos(math.pi/2))
        dirac_2 = np.kron(sigma1, sigma1) * math.sin(kx)
        dirac_3 = np.kron(sigma2, sigma1) * math.sin(ky)
        hamiltonian = dirac_1 + dirac_2 + dirac_3
    if model == "simple_dirac":
        hamiltonian = np.kron(sigma3,sigma1)*kx+np.kron(sigma3,sigma2)*ky+np.kron(sigma3,sigma3)*kz
    if model == "simple_weyl":
        hamiltonian =sigma1*kx+sigma2*ky+sigma3*kz
    if model == "C4_transformed":
        hamiltonian = np.zeros((4,4),dtype=np.complex)
        hamiltonian += np.kron(sigma0,sigma3)*(math.cos(kx) + math.cos(ky) + math.cos(kz) - 2 - math.cos(math.pi/2))
        lm = math.sin(kx) -1j*math.sin(ky)
        lp = math.sin(kx) +1j*math.sin(ky)
        hamiltonian[0,1]=lp
        hamiltonian[1,0]=lm
        hamiltonian[2,3]=-lm
        hamiltonian[3,2]=-lp
    if model == "V130":
        t_xy = 1
        t_z = 0.5
        l1 = 0.3
        l2 = 0.3
        l3 = 0.3
        base1 = np.kron(sigma0, np.kron(sigma1, sigma0)) * t_xy * math.cos(kx / 2) * math.cos(ky / 2)
        base2 = np.kron(sigma0, np.kron(sigma0, sigma1)) * t_z * math.cos(kz / 2)
        v130_1 = np.kron(sigma3, np.kron(sigma3, sigma2))*l1*math.cos(kz/2)
        v130_2 = l2 * np.kron(sigma1 * math.sin(ky) - sigma2 * math.sin(kx), np.kron(sigma3, sigma0))
        v130_3 = l3 * np.kron(sigma1 * math.sin(kx / 2) * math.cos(ky / 2) + sigma2 * math.cos(kx / 2) * math.sin(ky / 2),np.kron(sigma1, sigma3))
        hamiltonian = base1 + base2 +v130_1+v130_2+v130_3
    if model == "V130_1": #weyl
        t_xy = 1
        t_z = 0.5
        l1 = 0.3
        l2 = 0.3
        l3 = 0.3
        base1 = np.kron(sigma0, np.kron(sigma1, sigma0)) * t_xy * math.cos(kx / 2) * math.cos(ky / 2)
        base2 = np.kron(sigma0, np.kron(sigma0, sigma1)) * t_z * math.cos(kz / 2)
        v130_1 = np.kron(sigma3, np.kron(sigma3, sigma2)) * l1 * math.cos(kz / 2)
        v130_2 = l2 * np.kron(sigma1 * math.sin(ky) - sigma2 * math.sin(kx), np.kron(sigma3, sigma0))
        v130_3 = l3 * np.kron(
            sigma1 * math.sin(kx / 2) * math.cos(ky / 2) + sigma2 * math.cos(kx / 2) * math.sin(ky / 2),
            np.kron(sigma1, sigma3))
        v1 = np.kron(np.kron(sigma1, sigma3), sigma1) * 0.8 + 0 * np.kron(np.kron(sigma0, sigma3), sigma0)
        hamiltonian = base1 + base2 +v1+(v130_1+v130_2+v130_3)*0
    if model == "V130_2":
        t_xy = 1
        t_z = 0.5
        l1 = 0.3
        l2 = 0.3
        l3 = 0.3
        base1 = np.kron(sigma0, np.kron(sigma1, sigma0)) * t_xy * math.cos(kx / 2) * math.cos(ky / 2)
        base2 = np.kron(sigma0, np.kron(sigma0, sigma1)) * t_z * math.cos(kz / 2)
        v1 = np.kron(np.kron(sigma0, sigma3), sigma3) * 0.8 + 1 * np.kron(np.kron(sigma0, sigma0), sigma3)
        hamiltonian = base1 + base2 + v1
    if model == "Weyl":
        t_xy = 1
        t_z = 0.5
        l1 = 0.3
        l2 = 0.3
        l3 = 0.3
        base1 = np.kron(sigma0, np.kron(sigma1, sigma0)) * t_xy * math.cos(kx / 2) * math.cos(ky / 2)
        base2 = np.kron(sigma0, np.kron(sigma0, sigma1)) * t_z * math.cos(kz / 2)
        v1 = np.kron(sigma1, np.kron(sigma3, sigma1)) * math.cos(kz / 2)
        v2 =np.kron(sigma0, np.kron(sigma3, sigma0)) *  math.cos(kz / 2)
        hamiltonian = base1 + base2 + v1 + v2
    return hamiltonian

#+------------------------------------------------------------------+
#PURPOSE  : Generates H(k) on a mesh
#+------------------------------------------------------------------+

def listOfHamiltonians(steps,cutoff,model):
    end = math.pi
    hamiltonianList = []
    step = 2.0 * end / steps
    corr = 0*step / 2
    offset =0
    kx = -end + offset + corr
    for x in range(0, steps):
        ky=-end+corr
        for y in range(0, steps):
            kz=-end+corr
            for z in range(0, steps):
                if (kx**2+ky**2+kz**2)**0.5 <= cutoff:
                    hamiltonianList.append(hamiltonianOfK(kx,ky,kz,model))
                kz+=step
            ky+=step
        kx+=step
    hamiltonianList=np.array(hamiltonianList)
    print("... finished Hamiltonian List ...")
    return hamiltonianList


#+------------------------------------------------------------------+
#PURPOSE  : auxiliaries
#+------------------------------------------------------------------+

def KramersKronig_no_tails(omega,function):
    omega = np.real(omega)
    sum = np.zeros(len(omega), dtype=np.complex)
    sum=auxiliaries.KK(omega,function)
    sum = sum / math.pi
    return sum

def KramersKronig_thread(points,thread_number,retSigma,sigma,omega,parameters):
    sigma_new=np.zeros((len(points), parameters.nbands, parameters.nbands), dtype=np.complex)
    for w in range(0,len(points)):
        for i in range(0,parameters.nbands): # compute the real part of the self-energy using Kramers-Kronig
            sigma_new[w,i,i]= np.imag(points[w,1][i,i])*1j+auxiliaries.KK_single_value(np.imag(sigma[:,i,i]),np.real(omega),np.real(points[w,0]))/np.pi
    retSigma[thread_number]= sigma_new

def HilbertDirac(z,d):
    cutoff=d
    return  3 * (-z ** 2 * np.log((z - cutoff) / (z + cutoff)) - 2 * z*cutoff) / 2

def HilbertDiracPlusSquare(z,d,l):
    d=d
    l=l
    znorm=(2.*((d-l)*l**2)+(2.*l**3)/3.)
    return (l**2*(np.log((z-l)/(z+l))+np.log((z+d)/(z-d)))-2.*z*l-z**2*np.log((z-l)/(z+l)))/znorm 

def HilbertSquare(z,d):
    return (np.log((z+d)/(z-d))/(2*d))

def HilbertBethe(z,d):
    return ( 2.0*z/d**2*(1.0-np.sqrt(1.0-d**2/z**2)))

def getFermi(omega,beta,zeroT,mu):
    if zeroT==True: # zero T
        if not mu == 0: print("mu != 0 NOT IMPLEMENTED for zero T !!!!")
        fermi = np.zeros(len(omega))
        for w in range(0, len(fermi)):
            if w < len(fermi)/2: fermi[w]=1
            if w == len(fermi)/2: fermi[w] = 0.5
            if w > len(fermi) / 2: fermi[w] = 0
        return fermi
    else:
        fermi = np.zeros(len(omega))
        for w in range(0,len(fermi)):
            fermi[w] = 1.0/(np.exp(beta*(omega[w].real-mu))+1.0)
        return fermi

def freq_index(w,omega_max,omegastep):
    index= int(((w+omega_max)/omegastep).real)
    return index


#+---------------------------------------------------------------------+
#PURPOSE  :General function for organizing parallel processing
#+---------------------------------------------------------------------+

def parallel_function(function,number_of_threads,points,res,args):
    # distributes 'points' for 'function' over all threads
    # ret determines how the shape of the result should look
    # args includes all the arguments needed for 'function'; three more arguments needed for the muliprocessing organisation will be added at the beginning of args

    manager = multiprocessing.Manager()
    retDict = manager.dict()
    thread_step = (len(points) - len(points) % number_of_threads) / number_of_threads
    thread_rest = len(points) % number_of_threads
    threadList = []
    stepsdone = 0
   
    for t in range(0, number_of_threads):  # start threads and distribute work
        steps_to_do = int(thread_step)
        if t < thread_rest: # in case the tasks cannot be evenly distributed among the threads, the additional tasks are distributed to the first threads
            steps_to_do += 1
        points_thread = points[stepsdone:stepsdone + steps_to_do]  # part of omega-points this thread will calculate

        #write some additional arguments to the arguments given to the threads
        initial_argsList= list(args)
        argsList=[]
        argsList.append(points_thread)
        argsList.append(t)
        argsList.append(retDict)
        argsList.extend(initial_argsList)
        args_thread=tuple(argsList)
        threadList.append(Process(target=function, args=args_thread))
        threadList[t].start()
        stepsdone += steps_to_do
    for t in range(0, number_of_threads): # join threads
        threadList[t].join()
    for t in range(0,  number_of_threads):  # combine results from different threads
        res = np.vstack((res, retDict.get(t)))
    return res



#+------------------------------------------------------------------+
#PURPOSE  : Parallel processing to obtain G and G0
#+------------------------------------------------------------------+

def G_thread(points,thread_number,retG_G0, hamiltonianList, para):
    omega=points[:,0,0,0]
    sigma=points[:,:,:,1]
    gloc = np.zeros((len(omega), para.nbands, para.nbands), dtype=np.complex)
    g0 = np.zeros((len(omega), para.nbands, para.nbands), dtype=np.complex)
    idmat =  np.identity(para.nbands,dtype=np.complex)
    if not para.ksum:
      for w in range(0, len(omega)):
        for iii in range(0,para.nbands):
          z = omega[w] - sigma[w, iii, iii] + para.mu 
          if para.model =="bethe":
            g0_Loc_diag = HilbertBethe(z,para.d)
          if para.model =="simple_dirac":
            g0_Loc_diag = HilbertDirac(z,para.d)
          if para.model =="dirac_square":
            g0_Loc_diag = HilbertDiracPlusSquare(z,para.d,para.l)
          if para.model =="square":
            g0_Loc_diag = HilbertSquare(z,para.d)
          gloc[w, iii, iii] = g0_Loc_diag
    else:
      gloc = auxiliaries.Gloc_ksum(omega, sigma, hamiltonianList, para)

    for w in range(0, len(omega)):
        g0[w] = inv(inv(gloc[w])+sigma[w])
    retVal= np.zeros((len(points),para.nbands,para.nbands,2), dtype=np.complex)
    for w in range(0,len(retVal)):
        retVal[w,:,:,0]=gloc[w]
        retVal[w,:,:,1]=g0[w]
    retG_G0[thread_number]= retVal

def Calc_g0_and_gLoc(omega,hamiltonianList,sigma,number_of_threads,para):
    points=np.zeros((len(omega),para.nbands,para.nbands,2),dtype=np.complex)
    points[:,0,0,0]=omega[:]
    points[:,:,:,1]=sigma[:]
    ret=parallel_function(G_thread,number_of_threads,points,np.zeros((0,  para.nbands,  para.nbands,2), dtype=np.complex),(hamiltonianList,para))  #AAAAAAAAAAAAA UST
    gloc=ret[:,:,:,0]
    g0=ret[:,:,:,1]
    return g0, gloc

#+---------------------------------------------------------------------+
#PURPOSE  : Call auxiliaries functions that calculate P1, P2, Sigma
#+---------------------------------------------------------------------+

def CalcP(omega,g0,fermi):
    p1 = np.zeros(len(omega))
    p2 = np.zeros(len(omega))
    g0 = np.imag(g0)
    omega = np.real(omega)
    p1,p2=auxiliaries.CalcP(omega,g0,fermi)
    p1=p1/np.pi
    p2= p2/np.pi
    return p1,p2

def sigmaIntegral(w,omega,fermi,g0,p1,p2):
    integral= np.zeros(1, dtype=np.complex)
    g0=np.imag(g0)
    omega=np.real(omega)
    integral=auxiliaries.sigmaIntegral(w,omega,g0,fermi,p1,p2)
    return integral/np.pi

def P_thread(points,thread_number,retP,g0,omega,fermi):
    p1=np.zeros((len(points),1), dtype=np.double)
    p2=np.zeros((len(points),1), dtype=np.double)
    for w in range(0,len(points)):
        p1[w],p2[w]= auxiliaries.CalcP_singleValue(np.imag(g0),np.real(omega), fermi, np.real(points[w]))
    retVal= np.zeros((len(points),2), dtype=np.complex)
    for w in range(0,len(retVal)):
        retVal[w,0]=p1[w]
        retVal[w,1]=p2[w]
    retP[thread_number]= retVal


#+---------------------------------------------------------------------+
#PURPOSE  :Parallel processing to obtain Sigma
#+---------------------------------------------------------------------+

def Sigma_thread(points,thread_number,retSigma,omega, g0, fermi, p1, p2, para): #AAAAAAAAAAAA EXTEND TO CASE WITH UST
  omega_thread=points
  sigma = np.zeros((len(omega_thread), para.nbands, para.nbands), dtype=np.complex)
  for w in range(0, len(omega_thread)):
    for i in range(0, para.nbands):
      sigma[w, i, i] = sigmaIntegral(float(omega_thread[w].real), omega, fermi, g0[:,i,i], p1[:,i], p2[:,i])*1j*para.ULOC[i]**2
  retSigma[thread_number] = sigma


def Calc_Sigma(omega,g0,number_of_threads,para):
    fermi = getFermi(np.real(omega), para.beta, para.zeroT, 0)
    p1 = np.zeros((len(omega),para.nbands))
    p2 = np.zeros((len(omega),para.nbands))
    for i in range(para.nbands):
        points=np.real(omega)
        res=parallel_function(P_thread,number_of_threads,points,np.zeros((0,2)),(g0[:,i,i],omega,fermi)).real/np.pi
        p1[:,i]= res[:,0]
        p2[:,i]= res[:,1]
    manager = multiprocessing.Manager()
    retsigma = manager.dict()

    symm=True # calculate only half of the frequencies by assuming symmetry
    if para.mu_tilde != 0: symm=False # in case of non-half-filling the self-energy is not symmetric !!!
    full_omega=omega # when using symmetry,the full omega is still needed in some formulas;
    if symm:
        omega = omega[int((len(omega)-1)/2):len(omega)]

    points=np.real(omega)
    sigma=parallel_function(Sigma_thread,number_of_threads,points,np.zeros((0,  para.nbands,  para.nbands)),(full_omega,g0,fermi,p1,p2,para))

    if symm:
        new_sigma= np.zeros((int((len(full_omega)-1)/2),  para.nbands,  para.nbands), dtype=np.complex)
        for w in range(len(new_sigma)):
            new_sigma[w] = -np.real(sigma[len(new_sigma)-w]) + np.imag(sigma[len(new_sigma)-w]) * 1j
        sigma= np.vstack(( new_sigma,sigma ))
        omega=full_omega

    points=np.zeros((len(sigma),2), dtype=object)
    for i in range(0,len(points)):
        points[i,0]=omega[i]
        points[i,1]=sigma[i,:,:]
    sigma=parallel_function(KramersKronig_thread,number_of_threads,points,np.zeros((0,  para.nbands,  para.nbands), dtype=np.complex),(sigma,omega,para))  #AAAAAAAAAAAAA UST

    return sigma


def disorderMode_Sigma(omega, g_csi, g0_csi, sigma_csi, para):
    sigma_ret = np.zeros((len(omega), para.nbands, para.nbands), dtype=np.complex)
    fermi= getFermi(np.real(omega),para.beta,para.zeroT,0)
    A=[0,0,0,0]
    B=[0,0,0,0]
    for i in range(0,para.nbands):
      if para.ULOC[i] != 0:
        n=integrate.trapz(np.imag(g0_csi[:, i, i]*fermi), np.real(omega) )/ (-np.pi) #getn_band(omega,g_csi,fermi,para,i) 
        n0=integrate.trapz(np.imag(g0_csi[:, i, i]*fermi), np.real(omega) )/ (-np.pi) # getn0_band(omega,g0_csi,gloc,sigma,para.mu_tilde,fermi,para,i)
        A[i]=n*(1-n)/(n0*(1-n0))
        B[i]=((1-n)*para.ULOC[i]-para.csi-para.mu+para.mu_tilde)/(n0*(1-n0)*para.ULOC[i]**2)

        #print("A, B",A[i],B[i])
        #print("n,n0: ",n,n0)

    for i in range(0,para.nbands):
      if para.ULOC[i] != 0:
        sigma_ret[:,i,i]= para.ULOC[i]*n + (A[i]*sigma_csi[:,i,i])/(1-B[i]*sigma_csi[:,i,i])
      else:
        sigma_ret[:,i,i]= 0
    return sigma_ret


#+---------------------------------------------------------------------+
#PURPOSE  :Get occupation numbers
#+---------------------------------------------------------------------+


def getn0(omega,g0_csi,delta,mu_T,fermi,para):            
    for band in range(0,para.nbands):
        g0_csi[:,band,band] = 1.0/( omega[:] - delta[:,band,band] + (mu_T)*np.ones(para.omegasteps,dtype=np.complex))
    n0=0
    for i in range(0,para.nbands):
        n0 += integrate.trapz(np.imag(g0_csi[:, i, i]*fermi), np.real(omega) )/ (-np.pi)
    return n0

def getn(omega,gloc,fermi,para):
    n_test=0
    for i in range(0,para.nbands):
        n_test+= integrate.trapz(np.imag(gloc[:, i, i] * fermi), np.real(omega)) / (-np.pi)
    return n_test

def getn0_band(omega,g0,gloc,sigma,mu_T,fermi,para,band):
    g0[:] = inv(inv(gloc[:]) + sigma[:] - (para.mu + para.csi - mu_T)*np.identity(para.nbands))
    n0 = integrate.trapz(np.imag(g0[:, band, band]*fermi), np.real(omega) )/ (-np.pi)
    return n0

def getn_band(omega,gloc,fermi,para,band):
    n_test = integrate.trapz(np.imag(gloc[:, band, band] * fermi), np.real(omega)) / (-np.pi)
    return n_test

#+---------------------------------------------------------------------+
#CLASS : Solve Impurity problem
#+---------------------------------------------------------------------+

class Solver:
    def __init__(self, para,omega,fermi,number_of_threads):
        #define member self.G_w, self.G0_w, self.Sigma_w
        #the user of this class will then set the value of delta_w before calling solve
        self.number_of_threads = number_of_threads
        self.fermi = fermi
        self.omega = omega
        self.mu_T = 0.0
        self.delta = np.zeros((para.omegasteps,para.nbands,para.nbands) ,dtype=np.complex)
        self.g0_csi = np.zeros((para.omegasteps,para.nbands,para.nbands) ,dtype=np.complex)
        self.g_csi = np.zeros((para.omegasteps,para.nbands,para.nbands) ,dtype=np.complex)
        self.sigma_csi = np.zeros((para.omegasteps,para.nbands,para.nbands) ,dtype=np.complex)

    def solve(self,para,delta):
        for band in range(0,para.nbands):
            self.g0_csi[:,band,band] = 1.0/( self.omega[:] - delta[:,band,band] + (para.mu_tilde)*np.ones(para.omegasteps,dtype=np.complex))

        self.n0 = getn0(self.omega,self.g0_csi,delta,para.mu_tilde,self.fermi,para)

        # second order sigma diagram
        self.sigma_csi = Calc_Sigma(self.omega, self.g0_csi, self.number_of_threads, para)  

        #Sigma interpolating , ansatz A(n(csi))/(1-B(n(csi)))
        self.sigma_csi = disorderMode_Sigma(self.omega, self.g_csi, self.g0_csi, self.sigma_csi, para)

        # Impurity green function
        self.g_csi[:] = inv(inv(self.g0_csi[:]) - self.sigma_csi[:] + (para.mu+para.csi-para.mu_tilde)*np.identity(para.nbands,dtype=np.complex)  ) 

        self.n = getn(self.omega,self.g_csi,self.fermi,para)

#+---------------------------------------------------------------------+
#PURPOSE  : Interpolating integral for the disorder
#+---------------------------------------------------------------------+

def NumHilbert_disorder(z, weight, xi, sigmaxi):
    HH = 0j
    for ixi in range(len(xi) - 1):
        AW = (weight[ixi]*xi[ixi+1] - weight[ixi+1]*xi[ixi]) / (xi[ixi+1] - xi[ixi])
        BW = (weight[ixi+1] - weight[ixi]) / (xi[ixi+1] - xi[ixi])
        Adis = (sigmaxi[ixi]*xi[ixi+1] - sigmaxi[ixi+1]*xi[ixi]) / (xi[ixi+1] - xi[ixi])
        Bdis = (sigmaxi[ixi+1] - sigmaxi[ixi]) / (xi[ixi+1] - xi[ixi])
        HH += (BW*(xi[ixi+1] - xi[ixi]) + (AW - BW*(z - Adis) / Bdis) * cmath.log((z - Adis + (1.0-Bdis)*xi[ixi+1]) / (z - Adis + (1.0-Bdis)*xi[ixi]))) / (1.0-Bdis)
    return HH

#+---------------------------------------------------------------------+
#PURPOSE  : IPT iteration
#+---------------------------------------------------------------------+

def IPT_loops(omega,hamiltonianList,sigma,fermi,mix0,number_of_threads,para):
    # DMFT self-consistency loop
    print("... starting IPT loop ...")
    hamiltonianList_old=np.zeros((len(hamiltonianList),para.nbands,para.nbands),dtype=np.complex)
    hamiltonianList_old=hamiltonianList #save a copy of H0
    olddiff=100000 # initialize comparision value, just needs to be larger than anything going to appear in the first iteration
    failed_run = False

    #before the loop: first shift of the Hamiltonian if counterterm is requested
    if para.dopedMode and para.CounterTerm:
      print(0.5*np.array(para.ULOC)*np.identity(para.nbands,dtype=np.complex))
      hamiltonianList=hamiltonianList_old-0.5*np.array(para.ULOC)*np.identity(para.nbands,dtype=np.complex)

    #load the vectors for the values of the disorder and respective probabilities
    dis = float(sys.argv[6])

    try:
        if dis==0.0:
            v=np.array([0.0], dtype=np.double)
            prob=1.0
        else:
            v = np.linspace(-dis, dis,31)
            prob = (1.0/(2.0*dis))*np.ones(len(v),dtype=np.double)
    except IOError:
        print("Error: impossible to read the disorder")
        sys.exit(6)

    sigmaxi=np.zeros(len(v), dtype=np.complex)

    print('values for disorder=',v)
    print('probabilities=',prob)

    S = Solver(para,omega,fermi,number_of_threads)
    #initialize the quantities for different csi 
    S.sigma_csi = sigma.copy()
    g0 = np.zeros((para.omegasteps,para.nbands,para.nbands) ,dtype=np.complex)
    gloc = g0.copy()
    nada = g0.copy()
    g_csi_tot = np.zeros((para.omegasteps,para.nbands,para.nbands,len(v)) ,dtype=np.complex)
    s_csi_tot = np.zeros((para.omegasteps,para.nbands,para.nbands,len(v)) ,dtype=np.complex)
    mutilde_csi = np.zeros(len(v),dtype=np.double)

    delta = np.zeros((para.omegasteps,para.nbands,para.nbands) ,dtype=np.complex)

    for w in range(0, len(omega)):
        for iii in range(0,para.nbands):
          z = omega[w] + para.mu - sigma[w, iii, iii]
          if para.model =="bethe":
            gloc[w,iii,iii] = HilbertBethe(z,para.d)
            delta[w,iii,iii] = (para.d*para.d)/4.0*HilbertBethe(z,para.d)
    for i_csi in range(0,len(v)):
        s_csi_tot[:,:,:,i_csi] = S.sigma_csi[:,:,:]
        g_csi_tot[:,:,:,i_csi] = gloc[:,:,:]
        mutilde_csi[i_csi]=v[i_csi]

    for band in range(0,para.nbands):
        g0[:,band,band] = 1.0/( omega[:] - delta[:,band,band] + (para.mu)*np.ones(len(omega),dtype=np.complex))
            

    #LOOP
    for l in range(0,para.loops):

        #Here is changed that we want to fix mu and not n
        if para.fix_n == True:
            para.mu = np.average(para.ULOC)/2.0
            #muSearch(omega,sigma,number_of_threads,hamiltonianList,para)

        for i_csi in range(0,len(v)):

            para.csi = v[i_csi]
            para.mu_tilde = mutilde_csi[i_csi]
            print("csi=", para.csi)
            searching=True
            counter=0
            S.solve(para,delta)
            incr=abs(S.n-S.n0)/2.0
            #print(S.n0, S.n, para.mu_tilde,"first")
            errToZero = 0.0000001
            while searching==True :
                if abs(S.n-S.n0)< errToZero: searching=False
                else:
                    mu_old=para.mu_tilde
                    if S.n0 < S.n:
                        #print(S.n0, S.n,para.mu_tilde,"Increasing mu")
                        para.mu_tilde+=incr
                        S.solve(para,delta)
                        if S.n0 > S.n:
                            incr=abs(para.mu_tilde-mu_old)/2.0
                    else:
                        #print(S.n0, S.n,para.mu_tilde,"Decreasing mu")
                        para.mu_tilde-=incr
                        S.solve(para,delta)
                        if S.n0 < S.n:
                            incr=abs(para.mu_tilde-mu_old)/2.0
                counter+=1
                #data1 = np.column_stack((np.real(omega),np.imag(S.g_csi[:,0,0]),np.real(S.g_csi[:,0,0])))
                #filename1="g_csi"+str(para.csi)+ "it"+ str(l) + str(counter)+".dat"
                #np.savetxt(filename1, data1)

                #data1 = np.column_stack((np.real(omega),np.imag(S.sigma_csi[:,0,0]),np.real(S.sigma_csi[:,0,0])))
                #filename1="s_csi"+str(para.csi)+ "it"+ str(l) + str(counter)+".dat"
                #np.savetxt(filename1, data1)
                
                if counter> 300:
                    searching=False
                    print( "DID NOT FIND MU_TILDE !!!",  S.n, S.n0, para.mu_tilde, para.mu,incr)
                    para.mu_tilde=para.mu
            
            print("counter mu tilde = ", counter, S.n, S.n0, para.mu_tilde)
            
            #save in memory for next iteration g_csi 
            g_csi_tot[:,:,:,i_csi] = S.g_csi[:,:,:]
            s_csi_tot[:,:,:,i_csi] = S.sigma_csi[:,:,:]
            mutilde_csi[i_csi] = para.mu_tilde
            data1 = np.column_stack((np.real(omega),np.imag(S.g_csi[:,0,0]),np.real(S.g_csi[:,0,0])))
            filename1="g_csi"+str(para.csi)+ "it"+ str(l) + ".dat"
            np.savetxt(filename1, data1)

            data1 = np.column_stack((np.real(omega),np.imag(S.sigma_csi[:,0,0]),np.real(S.sigma_csi[:,0,0])))
            filename1="s_csi"+str(para.csi)+ "it"+ str(l) + ".dat"
            np.savetxt(filename1, data1)

        # Calculate the average over the disorder of gloc
        #gloc = np.sum(g_csi_tot*prob, axis=-1)*abs(v[1]-v[0])
        # Average with the linear interpolation
        for band in range(0,para.nbands):
             for w in range (0,len(omega)):
                z = omega[w] + para.mu - delta[w,band,band]
                sigmaxi[:] = s_csi_tot[w,band,band,:]
                gloc[w,band,band] = NumHilbert_disorder(z, prob, v, sigmaxi)
        #

        data1 = np.column_stack((np.real(omega),np.imag(gloc[:,0,0]),np.real(gloc[:,0,0])))
        filename1="gav_" + str(l) + ".dat"
        np.savetxt(filename1, data1)
        data1 = np.column_stack((np.real(omega),np.imag(g0[:,0,0]),np.real(g0[:,0,0])))
        filename1="g0_" + str(l) + ".dat"
        np.savetxt(filename1, data1)
        data1 = np.column_stack((np.real(omega),np.imag(delta[:,0,0]),np.real(delta[:,0,0])))
        filename1="delta_" + str(l) + ".dat"
        np.savetxt(filename1, data1)

        delta[:] = (para.d*para.d)/4.0*gloc[:] 

        for band in range(0,para.nbands):
            g0[:,band,band] = 1.0/( omega[:] - delta[:,band,band] + (para.mu)*np.ones(para.omegasteps,dtype=np.complex))


        #Calculate the self-energy with the Dyson equation
        sigma[:]=inv(g0[:])-inv(gloc[:]) #+ para.mu*np.identity(para.nbands,dtype=np.complex) #+ (para.mu-para.mu_tilde) *np.identity(para.nbands,dtype=np.complex)

        data1 = np.column_stack((np.real(omega),np.imag(sigma[:,0,0]),np.real(sigma[:,0,0])))
        filename1="s_" + str(l) + ".dat"
        np.savetxt(filename1, data1)

        nada, gloc = Calc_g0_and_gLoc(omega,hamiltonianList,sigma,number_of_threads,para)



#---------------------------------------------------------------------------------------------------------------------------
#------------------------CONVERGENCE----------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------
        val = delta.copy()
        print('...using delta conv...')
        #mix old and new value may help achieving convergence
        if l > 0:
          val_new = val * para.mix + (1.0 - para.mix) * val_old
        else:
          val_new = val
          val_old = np.zeros((len(omega),para.nbands,para.nbands) ,dtype=np.complex)

        #if required, fix mu such that Luttinger theorem is fulfilled
        if para.dopedMode and para.dopedMode_type == 2:
             para.mu =para.mu_0+sigma[freq_index(0,para.omega_max,para.omegastep), 0, 0].real 

        # compute difference to last iteration for real and imaginary part
        diffRs=[0,0,0,0]
        diffIs=[0,0,0,0]
        for band in range(0,para.nbands):
          diffRs[band]= integrate.trapz(abs(np.real(val_new - val_old))[:, band, band], np.real(omega))
          diffIs[band] = integrate.trapz(abs(np.imag(val_new - val_old))[:, band, band], np.real(omega))
        diffR=max(diffRs)
        diffI=max(diffIs)


        print(l,  diffI + diffR, para.mix,para.mu,para.mu_tilde,getn(omega,gloc,fermi,para),getn(omega,sigma,1.0,para) )
        #
        # stop loop if below convergence threshold
        if diffR+diffI<para.threshold and l >= para.min_loops: break 

        # reduce mixing value, helps reaching convergence
        if para.adaptiveMixing == True: 
            if diffR+diffI> olddiff: para.mix=para.mix/1.2

        olddiff= diffR+diffI

        val_old[:] = val_new[:]
        delta[:] = val_new[:]

#---------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------

        #g0,gloc=Calc_g0_and_gLoc(omega,hamiltonianList,sigma,number_of_threads,para)

#--------PRINT Iterations --------------------------------------------------------------------------------------------------

        data1 = np.column_stack((np.real(omega),np.imag(gloc[:,0,0]),np.real(gloc[:,0,0])))
        filename1="g_" + str(l) + ".dat"
        np.savetxt(filename1, data1)
        #data1 = np.column_stack((np.real(omega),np.imag(g0[:,0,0]),np.real(g0[:,0,0])))
        #filename1="g0_" + str(l) + ".dat"
        #np.savetxt(filename1, data1)
        
#---------------------------------------------------------------------------------------------------------------------------


    print("Occupations:",getn_band(omega,gloc,fermi,para,0))
    data1 = np.column_stack((para.ULOC[0],para.ULOC[0],getn_band(omega,gloc,fermi,para,0)))
    filename1="n.dat"
    np.savetxt(filename1, data1)

    #write final iteration to  file
    if para.nbands >1:
      for iband in range(0,para.nbands):
        for jband in range(0,para.nbands):
          data1 = np.column_stack((np.real(omega),np.imag(gloc[:,iband,jband]),np.real(gloc[:,iband,jband])))
          filename1="gloc_" + str(iband) + str(jband)+ ".dat"
          np.savetxt(filename1, data1)
          data1 = np.column_stack((np.real(omega),np.imag(g0[:,iband,jband]),np.real(g0[:,iband,jband])))
          filename1="g0_" + str(iband) + str(jband)+ ".dat"
          np.savetxt(filename1, data1)
          data1 = np.column_stack((np.real(omega),np.imag(sigma[:,iband,jband]),np.real(sigma[:,iband,jband])))
          filename1="sigma_" + str(iband) + str(jband)+ ".dat"
          np.savetxt(filename1, data1)
    else:
          data1 = np.column_stack((np.real(omega),np.imag(gloc[:,0,0]),np.real(gloc[:,0,0])))
          filename1="gloc_" + str(0) + str(0)+ ".dat"
          np.savetxt(filename1, data1)
          data1 = np.column_stack((np.real(omega),np.imag(g0[:,0,0]),np.real(g0[:,0,0])))
          filename1="g0_" + str(0) + str(0)+ ".dat"
          np.savetxt(filename1, data1)
          data1 = np.column_stack((np.real(omega),np.imag(sigma[:,0,0]),np.real(sigma[:,0,0])))
          filename1="sigma_" + str(0) + str(0)+ ".dat"
          np.savetxt(filename1, data1)
          filename="test_one.dat"
          data = np.column_stack((np.real(omega),\
                                np.real(gloc[:,0,0]),\
                                np.imag(gloc[:,0,0]),\
                                np.real(g0[:,0,0]),\
                                np.imag(g0[:,0,0]),\
                                np.real(sigma[:,0,0]),\
                                np.imag(sigma[:,0,0])))
          np.savetxt(filename, data)

    return True



#+---------------------------------------------------------------------+
#PURPOSE  : Initialize and run
#+---------------------------------------------------------------------+


def initialize():
    #todo: implement usage of parameter file

    #--- parameters ---
    #read-in:
    #U= float(sys.argv[1]) # interaction strength
    Ulist= sys.argv[1] # interaction strength
    Ulist = Ulist.split(",")
    Ulist = [float(i) for i in Ulist]
    ULOC=Ulist
    UST=float(sys.argv[2])
    beta= 1/float(sys.argv[3]) #inverse temperature
    delta=float(sys.argv[4]) # small quantity
    print(len(sys.argv))
    if len(sys.argv) < 6:
        print("--- using parameters set in code ---")
        #set in code:
        number_of_threads = 8
        debug =False # prints every iteration to file (overwriting last iteration); slow
        loops= 50 #maximum number of loops
        min_loops=1 #minimum number of loops
        threshold=0.01 #convergence threshold
        readold=False #0 or 1, decides if old sigma is read, CHECK AGAIN FOR MU != 0  
        zeroT=False # explicitly needed for T=0 because temperature is given in beta and 1/0 is not valid; overwrites beta setting!  
        CounterTerm=False
        mix=0.4  # mixing in iteration step between new and old self-energy, mix=1.0 means only new self-energy
        adaptiveMixing=True #reduce mixing value when convergence got worse compared to last iteration
        omega_max=8
        omegasteps =6001
        ksteps=5
        nbands=1
        cutoff =10000 # needed for ksum of Dirac Hamiltonian
        ksum = False  # 0 means hilbert transform; 1 means ksum which only works for simple dirac model (or C4?), not for Bethe; Currently some problems with nbands persists 
        d = 1  # half-bandwidth for bethe lattice, cutoff for Dirac. Be careful with normalization in Dirac Case. Nomralized to 1 for d=1 !!!
        l = 0.5 # Cutoff for quadratic DOS in dirac_square 
               #omega grid type,possible values: linear, quadratic, linquad; not sure if non-linear ones are still working,
        omega_grid="linear"
        model = "simple_dirac" # only neede when ksum = 1 is used
        dopedMode=True # if True: dopedMode_type,outputN,mu_tilde,fix_n should be set (which one depends on type)
        dopedMode_type=1 #decides which parameter is fixed during the calulations for finite doping. 1) set mu_tilde so that n=n0 2) fix mu_tilde and change mu
        outputN=True # write information about filling n into output file, needed for automated post-processing for n!=0.5
        mu_tilde =-0.0  #only needed for calculations away from half-filling
        fix_n=True # probably works only in dopedMode_type =1
        n_aim=0.47 # target filling
    else: # read parameter file
        loops,\
        min_loops,\
        threshold,\
        zeroT,\
        CounterTerm,\
        mix,\
        adaptiveMixing,\
        readold,\
        number_of_threads,\
        debug,\
        omega_max,\
        omegasteps,\
        ksteps,\
        nbands,\
        cutoff,\
        ksum,\
        d,\
        l,\
        dopedMode,\
        dopedMode_type,\
        outputN,\
        mu_tilde,\
        fix_n,\
        n_aim,\
        model = readParameters(sys.argv[5])
        omega_grid="linear" 
    if dopedMode and dopedMode_type == 1:
        mu = np.average(ULOC) / 2.0 #float(sys.argv[6]) # set different form U/2 for changing n in dopedMode_type 1, unnecessary if fix_n is used  #AAAAAAAAAAAAAAAA       
        if CounterTerm: 
          mu = 0.0
    else:
        mu = 0
    mu_0 =mu_tilde  # needed for dopedMode_type 2
    if not dopedMode:
        mu_tilde=0
        mu_0=0
        fix_n=False
    csi=0.
    omegastep = omega_max/(omegasteps-1.0)*2.0

    # write parameters into Parameter object
    parameters = Parameters(d=d,\
                            l=l,\
                            nbands= nbands,\
                            ULOC=ULOC,\
                            UST=UST,\
                            mu=mu,\
                            omega_max=omega_max,\
                            omegasteps=omegasteps,\
                            ksum=ksum,\
                            delta=delta,\
                            omegastep=omegastep,\
                            beta=beta,\
                            mu_tilde=mu_tilde,\
                            n_aim=n_aim,\
                            zeroT=zeroT,\
                            CounterTerm=CounterTerm,\
                            dopedMode=dopedMode,\
                            dopedMode_type=dopedMode_type,\
                            mix=mix,\
                            loops=loops,\
                            fix_n=fix_n,\
                            debug=debug,\
                            threshold=threshold,\
                            min_loops=min_loops,\
                            adaptiveMixing=adaptiveMixing,\
                            outputN=outputN,\
                            omega_grid=omega_grid,\
                            model=model,\
                            mu_0=mu_0,\
                            csi=csi)
    parameters.Check() #check if parameters are reasonable

    #---- end of parameter section -----

    #calculate some values and initialize variables
    failed_run = False # used to return information about failed run, probably not needed anymore
    w_mu = (omegasteps-1)/2 # w index corresponding to mu
    mix0=mix
    omegstart = -omegastep * (omegasteps-1) / 2.0
    gloc = np.zeros((omegasteps,nbands,nbands), dtype=np.complex)
    g0 = np.zeros((omegasteps,nbands,nbands), dtype=np.complex)
    sigma = np.zeros((omegasteps,nbands,nbands) ,dtype=np.complex)
    for i in range(0,nbands):
        sigma[:,i,i]=parameters.mu
    if readold:
      for iband in range(0,nbands):
        for jband in range(0,nbands):
          try:
            sigmaLoad= np.loadtxt("sigma_"+ str(iband) + str(jband)+".dat", unpack=True, skiprows=0)
            if len(sigmaLoad[0])== omegasteps:
              sigma[:, iband, jband] = sigmaLoad[2, :] + sigmaLoad[1, :] * 1j
              print("... reading stored sigma...")
            else:
                print("can not use stored sigma, wrong number of frequencies!!! loaded file:", len(sigmaLoad),"new calculation:", omegasteps)
          except:
            print("No sigma file present.")
    omega = np.zeros(omegasteps,dtype=np.complex)

    for w in range(0,omegasteps): # test thoroughly if using anything but linear grid
        if omega_grid=="linear":
            omega[w] = omegstart+w*omegastep +1j*delta
        if omega_grid=="quadratic":
            omega[w] =  (w-(omegasteps - 1) / 2.0)**2 * omegastep *np.sign(w-(omegasteps - 1) / 2.0)+ 1j * delta
        if omega_grid == "linquad":
            border=130
            if np.abs(w - (omegasteps - 1) / 2.0) <= border:
                omega[w] =np.abs((w-(omegasteps - 1) / 2.0)**1) * omegastep *np.sign(w-(omegasteps - 1) / 2.0)+ 1j * delta
            if  np.abs(w-(omegasteps - 1) / 2.0) > border:
                corr = border
                if w-(omegasteps - 1) / 2.0 > 0: corr=-corr
                omega[w] =np.abs((border)**1) * omegastep *np.sign(w-(omegasteps - 1) / 2.0)\
                          +(w+corr-(omegasteps - 1) / 2.0)**2 * omegastep *np.sign(w-(omegasteps - 1) / 2.0) + 1j * delta
        if w > 0 and w < omegasteps:
            if omega[w-1]< mu and omega[w] > mu:w_mu= w

    fermi = getFermi(np.real(omega), parameters.beta, zeroT, 0)
    
    #compute Hamiltonians on k-grid
    if ksum:
        hamiltonianList=listOfHamiltonians(ksteps,cutoff,model)
    else:
        hamiltonianList=np.zeros((0,0,0), dtype=np.complex)
  
    #print some info and warnings
    print("--- running on", number_of_threads, "thread(s) ---")
    if debug == True: print("--- using DEBUG mode ---")
    if dopedMode==False and mu != 0: print("WARNING: mu SHOULD BE 0 !!!") 
    if not d == 1: print("d = ", d) 
    if not ksum:
        print("--- using", parameters.model, "---")
    else:
        print("--- using ksum with",ksteps**3, "points ---")
        print( "cutoff !!!!:", cutoff)
    print("--- Parameters: ---")
    print("    ULOC = ", ULOC, "  UST =", UST, "  delta =", delta, "  beta =", beta,"  omegasteps =", omegasteps,"  omega_max =", omega_max,"  mix =", mix, " threshold = ", threshold, " model =", model)
    if zeroT == True: print("!!! USING ZERO TEMPERATURE !!!")
    if fix_n == True: print("--- Fixing n : ", n_aim,"---")

    #start calculations
    return IPT_loops(omega,hamiltonianList,sigma,fermi,mix0,number_of_threads,parameters)


#+---------------------------------------------------------------------+
#PURPOSE  : MAIN
#+---------------------------------------------------------------------+


if __name__ == "__main__":
    start_time=time.time()
    print("--- starting time: ",time_string(), "---")
    run = True
    counter = 1
    while run == True:
        result = initialize()
        if result: run = False
        else:
            print("run",counter,"failed")
            print("trying again")
            counter+=1
        if counter > 10:
            run = False
            print("failed",counter,"times")
            print("aborting...")
    print("total time: ", time.time()-start_time)



