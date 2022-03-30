# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 11:58:30 2022

@author: Jasar
"""
import numpy as np
class Turbine:
    """ z,C_t,z_0,alpha,r_r,a  """
    def __init__(self,h,C_t,r_r,z0,pf):
        self.z0 = z0
        self.h = h
        self.C_t = C_t
        self.pf =pf
        
        self.a = 0.5*(1-np.sqrt(1-C_t))
        self.r_r = r_r
        self.r1 =  r_r*(np.sqrt((1-self.a)/(1-2*self.a)))
        self.alpha= 0.5/np.log(self.h/z0)
    def p_out(self,colm,u):
        pout = 0
        pf = self.pf
        defl = lambda x : np.square(2*self.a/(1+self.alpha*x/(self.r1)))
        
        D = 10*self.r_r
        n = len(colm)
        uiarr=[]
        for i in range(n):
            
            
            if colm[i]==0:
                continue
            deficit=0
            for j in range(n):
                if j<i and colm[j] == 1:
                    
                    deficit += defl((i-j)*D)
                    
                    
                    
                        
                        
            deficit = np.sqrt(deficit)
            ui = u*(1-deficit)
            
            #print(u)        
            pout += pf*ui**3        
                    
                    
                
                
        return pout
        
    def number_cost(self,N):
        return N*((1/3) + (2*(np.exp(-0.00174*np.square(N)))/3))
        
class Farm:
    def __init__(self,W,E,N,S,Wf,Ef,Nf,Sf):
        self.W,self.E,self.N,self.S,self.Wf,self.Ef,self.Nf,self.Sf = W,E,N,S,Wf,Ef,Nf,Sf
        pass
    
    
class Test:
    def __init__(self,a):
        self.a=a    
    
    def add(self):
        self.a += 1
        return self
        
    def sub(self):
        T=self.add()
        return self.a
        