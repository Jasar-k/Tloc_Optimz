# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 02:53:35 2022

@author: Jasar
"""
from Sys_model import * 
import numpy as np
import time
import matplotlib.pyplot as plt
np.random.seed(0)
def getRandomMAT(N, population_size):
    return np.random.randint(0,2,(N,N,population_size))

def MATtoGenome(p):
    g=[p[i] for i in range(int(np.shape(p)[0]))]
    g=tuple(g)
    n=np.hstack(g)
    return n
def GenometoMAT(genome):
    
    uc=[genome[i*10:i*10+10] for i in range(int(np.shape(genome)[0]/10))]
    uc=tuple(uc)
    uc=np.vstack(uc)
    return uc

def select_parents(pop,cost,ind):
    
    sel=np.random.choice(ind, 2, replace=False, p=cost)
    parents=np.vstack([pop[sel[0]],pop[sel[1]]])
    return parents

def crossover(pop,cost,r_cross,ind):
    
    parents=select_parents(pop, cost,ind)
    offsprings=parents
    if np.random.rand()<r_cross: 
        
        pivot_point_1 = int(np.shape(pop[0])[0]*0.25)
        pivot_point_2 = int(np.shape(pop[0])[0]*0.70)
        
        offsprings[0] = np.hstack((parents[0][0:pivot_point_1],
            parents[1][pivot_point_1:pivot_point_2],
            parents[0][pivot_point_2:]))
        offsprings[1]=np.hstack((parents[1][0:pivot_point_1],parents[0][pivot_point_1:pivot_point_2],parents[1][pivot_point_2:]))

    return offsprings

def mutation(genome,r_mut):
    if np.random.rand()<r_mut:
        #print("mut")
        n=np.random.randint(int(np.shape(genome)[0]))
        genome[n]= not(genome[n])
    return genome

def total_cost(m,T,F):
    m = GenometoMAT(m)
    W,E,N,S,Wf,Ef,Nf,Sf = F.W,F.E,F.N,F.S,F.Wf,F.Ef,F.Nf,F.Sf
    Wf,Ef,Nf,Sf =np.array([Wf,Ef,Nf,Sf])/sum([Wf,Ef,Nf,Sf])
    tc = 0
    od =np.array([Nf,Ef,Sf,Wf])
    U = np.array([N,E,S,W])
    p_cost = 0
    n_cost = 0
    for fr,u in zip(od,U):
        seti=0
        for i in range(10):
            colum = m[:,i]
            
            seti += T.p_out(colum,u)
            
    
        p_cost += fr*seti
        m = np.rot90(m)
    n_cost = T.number_cost(sum(sum(m)))
    
    tc = n_cost/p_cost
    #print(tc,n_cost,p_cost)
    return 1/tc

def costtoprob(cost):
    
    cost=cost/sum(cost)
    return cost
F = Farm(15,15,15,15,100,100,100,100) #W,E,N,S,Wf,Ef,Nf,Sf
T = Turbine(60,0.88,40,0.3,0.3) #h,C_t,r_r,z0,pf



r_mut=0.008
r_cross=0.9
N = 10
generations=200
population_size=102
ind=[i for i in range(population_size)]
uc=getRandomMAT(N, population_size)
ac_cost=[total_cost(uc[:,:,i],T,F) for i in range(population_size)]
ac_cost=np.array(ac_cost)
cost=costtoprob(ac_cost)
pop=[MATtoGenome(uc[:,:,i]) for i in range(np.shape(uc)[2])]
pop=np.array(pop)

plot1=[]
plot2=[]
start_time = time.time()
for gen_no in range(generations):
    print("Gen_No:",gen_no)
    elit=pop[np.argmax(cost)]
    next_gen=[elit,pop[np.random.randint(population_size)]]
    for _ in range(int((population_size-2)/2)):
        offsprings=crossover(pop, cost, r_cross, ind)
        offsprings[0]=mutation(offsprings[0], r_mut)
        offsprings[1]=mutation(offsprings[1], r_mut) 
        next_gen=np.append(next_gen,offsprings,axis=0)
    ac_cost=[total_cost(next_gen[i], T,F) for i in range(population_size)]
    ac_cost=np.array(ac_cost)

    pop=next_gen
    
    m=min(ac_cost)
    
    
    plot1.append(m)
    #maxFIT=GenometoMAT(pop[np.argmax(cost)])
    #plot2.append(GenometoMAT(pop[np.argmax(cost)]))
    cost=costtoprob(ac_cost)
    # plt.figure()
    # plt.imshow(maxFIT,cmap='gray')
    # plt.show()
    print(" ELITE cost:",ac_cost[0],"Population cost:",m)
    print("Execution time: ",time.time()-start_time, " seconds")
    
MaxFit=min(ac_cost)
MaxMAT=GenometoMAT(pop[np.argmax(cost)])

plt.plot(plot1)