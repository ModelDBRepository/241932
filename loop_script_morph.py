# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 10:32:29 2015

@author: rennocosta

This script is part of the publication Renno-Costa & Tort, 2017, JNeurosci
This script relates to the data presented in the Figure 5, 6, 7 and 8
Will run a single experiment with specific parameters determined below
Output data will be saved in file direction defined at support_filename.py script

"""

import sys, argparse
import numpy as np
from numpy import *
import gzip
import pickle
import support_filename as rfn
import copy

# acessory functions

def normalize_weight(www,www_mean):
    www /= np.tile(np.mean(www,axis=0),(www.shape[0],1)) 
    return www
    
def learn_weight(www,activity_pre,activity_pos,lrate):
    #www += lrate*(np.tile(activity_pre,(activity_pos.shape[0],1)).transpose()-www) * (np.tile(activity_pos,(activity_pre.shape[0],1)))
    www += lrate*(np.tile(activity_pre,(activity_pos.shape[0],1)).transpose()) * (np.tile(activity_pos,(activity_pre.shape[0],1)))   
    return www

def lec_whichone(lectype,change,ccc,sss):
    saida = np.zeros(lectype.shape)
    saida[np.logical_and(lectype==1,change<ccc)] = 2
    saida[np.logical_and(lectype==1,change>=ccc)] = 1
    saida[np.logical_and(lectype==0,change<sss)] = 2
    saida[np.logical_and(lectype==0,change>=sss)] = 1
    saida[lectype==2] = 1
    return saida

def main(argv):
    
   
    parser = argparse.ArgumentParser(description='Will run a simulation instance.')
 
	# seeds for the random number generator 
 
	# seed for the input activity pattern 
    parser.add_argument('seed_input', metavar='seed_input', type=int, nargs=1,
                   help='seed_input number') 
	# seed for the initial synaptic weights
    parser.add_argument('seed_www', metavar='seed_www', type=int, nargs=1,
                   help='seed_www') 
	# seed for the trajectories
    parser.add_argument('seed_path', metavar='seed_path', type=int, nargs=1,
                   help='seed_path') 
  

	# variables of the session
    
	# number of theta cycles simulated for each position

    parser.add_argument('theta_cycles', metavar='theta_cycles', type=int, nargs=1,
                   help='theta_cycles')
				   
	# number of times that the agent makes a full exploration of the arena for each session 
	
    parser.add_argument('arena_runs', metavar='arena_runs', type=int, nargs=1,
                   help='arena_runs')                
				   
	# number of sessions with plasticity run before collecting data 
	
    parser.add_argument('pre_runs', metavar='pre_runs', type=int, nargs=1,
                   help='pre_runs')
				   
                   
	# learning rates from place to grid cells, grid to place cells and from lec to place cells...                 

    parser.add_argument('lrate_hpc_mec', metavar='lrate_hpc_mec', type=int, nargs=1,
                   help='lrate_hpc_mec')                         
    parser.add_argument('lrate_mec_hpc', metavar='lrate_mec_hpc', type=int, nargs=1,
                   help='lrate_mec_hpc')                
    parser.add_argument('lrate_lec_hpc', metavar='lrate_lec_hpc', type=int, nargs=1,
                   help='lrate_lec_hpc') 
				   
	# ... relative number of place cells vs recurrent grid cells ...
	# sensibility of pattern completion algorithm                   
   

	# relative number of grid cells vs lec cells as place cell input (0 to 100)
       
    parser.add_argument('mec_ratio', metavar='mec_ratio', type=int, nargs=1,
                   help='MEC ratio (x100)')
				   
	# relative number of place cells vs recurrent grid cells as grid cell input (0 to 100)
	
    parser.add_argument('hpc_ratio', metavar='hpc_ratio', type=int, nargs=1,
                   help='HPC ratio (x100)')
	
	# relative number of place cells vs recurrent grid cells as grid cell input (0 to 100)	
				  
    parser.add_argument('hpc_pcompl_th', metavar='hpc_pcompl_th', type=int, nargs=1,
                   help='HPC pattern completion th (x100)')
				   
	# sensibility of pattern completion algorithm

    parser.add_argument('morph_per', metavar='morph_per', type=int, nargs=1,
                   help='morph_per') 
       
	# flag to save the activity of cells... use with caution
    parser.add_argument('-a', '--activity',dest='actsave',action='store_const',default="no",const="yes")  	


    # flags to set the place to save the files
	
    parser.add_argument('-w', '--windows',dest='envir',action='store_const',default="cluster",const="windows")
    parser.add_argument('-u', '--ufrgs',dest='envir',action='store_const',default="cluster",const="UFRGS")
    parser.add_argument('-s', '--NPAD',dest='envir',action='store_const',default="cluster",const="NPAD")
    parser.add_argument('-z', '--zurich',dest='envir',action='store_const',default="cluster",const="zurich")
    
    # flag to overwrite previous simulation
    parser.add_argument('-k', '--KILL',dest='tokill',action='store_const',default="no",const="yes")
    
	# for conencted environments, not implemented
	#parser.add_argument('-c', '--connected',dest='conntype',action='store_const',default="no",const="yes")        
    
	
	# process the arguments
	
    args = parser.parse_args() 
    envir = args.envir
    conntype = args.conntype
    actsave = args.actsave
    
    tokill = args.tokill;
    
    if(conntype=="yes"):
        conna = True
        ct = 1
    else:
        ct = 0
        conna = False
        
    if(actsave=="yes"):
        acts = True
    else:
        acts = False
        
    seed_input = args.seed_input[0]
    seed_www = args.seed_www[0]
    seed_path = args.seed_path[0]
    
    mec_ratio = float(args.mec_ratio[0])/100
    hpc_ratio = float(args.hpc_ratio[0])/100
    hpc_pcompl_th = float(args.hpc_pcompl_th[0])/100

    morphing_per = float(args.morph_per[0])/100
      
    lrate_hpc_mec = float(args.lrate_hpc_mec[0])/1000
    lrate_mec_hpc = float(args.lrate_mec_hpc[0])/1000
    lrate_lec_hpc = float(args.lrate_lec_hpc[0])/1000
        
    theta_cycles = args.theta_cycles[0]
    arena_runs = args.arena_runs[0]
    pre_runs = args.pre_runs[0]
    

    simulation_num = 69;
  
    listofvalues = [ct,args.seed_input[0],args.seed_www[0],args.seed_path[0],args.theta_cycles[0],args.arena_runs[0],args.pre_runs[0],args.lrate_hpc_mec[0],args.lrate_mec_hpc[0],args.lrate_lec_hpc[0],args.mec_ratio[0],args.hpc_ratio[0],args.hpc_pcompl_th[0],args.morph_per[0]]
      
    filenames = rfn.remappingFileNames(envir)
    
    filenames.prepareSimulation(listofvalues,simulation_num)  
    
    
    if (tokill == "no"):
        try:
            tosee = 0;
            with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,0)+'z', 'rb') as ff:
                tosee = tosee + 1
            with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,1)+'z', 'rb') as ff:
                tosee = tosee + 1
            with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,2)+'z', 'rb') as ff:
                tosee = tosee + 1
            with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,3)+'z', 'rb') as ff:
                tosee = tosee + 1
            with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,4)+'z', 'rb') as ff:
                tosee = tosee + 1
            with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,5)+'z', 'rb') as ff:
                tosee = tosee + 1
            print("File exist. Will exit!")
            torun = 0;
        except:
            print("File does not existe. Will do!",flush=True)  
            print("... %s" % (filenames.fileRunPickle(listofvalues,simulation_num,0)))   
            torun = 1;
    else:
        print("Will do anyway!",flush=True)
        torun = 0;
        

    if(torun == 0):
        sys.exit();

    
	
	# %%
    #
    #
    # MAKE ALL INITIALIZATIONS
    #
    #
    #
    
    #set the arena size
    arena_binsize = [4,4]
    
    context_per = 0
     
    
    # set the patterns of input activity
    np.random.seed(seed_input)  
    lec_numcells = 500
    lec_activity = []
    lec_activity.append(pow(np.random.uniform(0,1,(lec_numcells,arena_binsize[0],arena_binsize[1])),2))
    lec_activity.append(pow(np.random.uniform(0,1,(lec_numcells,arena_binsize[0],arena_binsize[1])),2))
    lec_type = np.random.uniform(0,1,(lec_numcells,arena_binsize[0],arena_binsize[1]))
    lec_type[lec_type>(1-context_per)] = 1
    lec_type[lec_type<morphing_per] = 0
    lec_type[np.logical_and(lec_type<=(1-context_per),lec_type>=morphing_per)]=2
    lec_change = np.random.uniform(0,1,(lec_numcells,arena_binsize[0],arena_binsize[1]))
    
    
    # number of place cells cells
    hpc_numcells = 5000
    hpc_memories = []  
    
    # set the grid cells topology
    mec_blocksize = [2,4,6,8,10,12,14,16]
    mec_blocks = len(mec_blocksize)
    mec_numcells = np.sum(np.power(mec_blocksize,2))
    mec_indexlist = []
    init_val = 0
    for ii in arange(mec_blocks):
        mec_indexlist.append((init_val+arange(pow(mec_blocksize[ii],2))).reshape((mec_blocksize[ii],mec_blocksize[ii]))) 
        init_val = np.max(mec_indexlist[ii])+1
    del(init_val)
    random.uniform(0,1,(lec_numcells,arena_binsize[0],arena_binsize[1]))
    
    # set the different trajectories
    np.random.seed(seed_path)
    xxx,yyy = np.meshgrid(arange(arena_binsize[0]),arange(arena_binsize[1]))
    xxx = xxx.ravel()
    popo = []
    for ii in arange(100):
        popo.append(np.random.permutation(len(xxx)))
    
    
    #set the initial synaptic weights
    
    np.random.seed(seed_www)  
    
    lec_hpc_weights_mean = 1
    lec_hpc_weights = np.random.lognormal(1.0,1.0,(lec_numcells,hpc_numcells))
    lec_hpc_weights[lec_hpc_weights<0] = 0
    lec_hpc_weights = normalize_weight(lec_hpc_weights,lec_hpc_weights_mean)
    
    mec_hpc_weights_mean = 1
    mec_hpc_weights = np.random.lognormal(1.0,1.0,(mec_numcells,hpc_numcells))
    mec_hpc_weights[mec_hpc_weights<0] = 0
    mec_hpc_weights = normalize_weight(mec_hpc_weights,mec_hpc_weights_mean)
    
    hpc_mec_weights_mean = 1
    hpc_mec_weights = np.random.lognormal(1.0,1.0,(hpc_numcells,mec_numcells))
    hpc_mec_weights[hpc_mec_weights<0] = 0
    hpc_mec_weights = normalize_weight(hpc_mec_weights,hpc_mec_weights_mean)
    
    
    # set the E%MAX
    
    current_emax = 0.90
    current_emax_plast = 0
    
    
    current_lrate_hpc_mec = lrate_hpc_mec
    current_lrate_mec_hpc = lrate_mec_hpc
    current_lrate_lec_hpc = lrate_lec_hpc
     
    lec_hpc_weights_mean = 1
    mec_hpc_weights_mean = 1
    hpc_mec_weights_mean = 1     
    
    lec_noise = 0
    mec_noise = 0
    hpc_noise = 0
    
       
# %% 
    #
    #
    # PRE-LEARN - SIMULATIONS WITH LEARNING BEFORE SAVING THE DATA
    #           - THIS PART SIMULATES THE HABITUATION OF THE ANIMAL PRIOR TO IMPLANT
    #
    #

       
    lllf = [1.0,1.0]   
    mooo = [mec_ratio,mec_ratio]
    shape_vec = [0.0,1.0]
    context_vec = [0.0,0.0]
    pzzz = [0,0]   
       
    
    for sessions in arange(pre_runs):
        
        print("session %d of %d" % (sessions,pre_runs),flush=True)   
            
        hhhr = hpc_ratio * (sessions/(pre_runs-1))     
        
        lllf[0] = 1.0
        lllf[1] = 1.0
            
        pzzz[0] = sessions + 16
        pzzz[1] = sessions + 16 + pre_runs

        lec_act_vect = []
        mec_act_vect = []
        hpc_act_vect = []       
        
        mec_inact_vect = np.zeros((len(shape_vec),mec_numcells,arena_binsize[0],arena_binsize[1]))
        hpc_inact_vect = np.zeros((len(shape_vec),hpc_numcells,arena_binsize[0],arena_binsize[1]))
        lec_inact_vect = np.zeros((len(shape_vec),lec_numcells,arena_binsize[0],arena_binsize[1]))

        for ii in arange(len(shape_vec),dtype=int):
            
            print("shape %d of %d" % (ii,len(shape_vec)),flush=True)  
        
            mec_ratio = mooo[ii]
                    
            lec_act = zeros((lec_numcells,arena_binsize[0],arena_binsize[1]))
            hpc_act = zeros((hpc_numcells,arena_binsize[0],arena_binsize[1]))   
            mec_act = zeros((mec_numcells,arena_binsize[0],arena_binsize[1]))
            
            
            xxx,yyy = np.meshgrid(arange(arena_binsize[0]),arange(arena_binsize[0]))
            xxx = xxx.ravel()
            yyy = yyy.ravel()
            #ppp = np.random.permutation(len(xxx))
            ppp = popo[pzzz[ii]]          
            xxx = xxx[ppp].astype(int)
            yyy = yyy[ppp].astype(int) 
            
            #
            #    xxx = xxx + 4
            
            
            if (lllf[ii]>0):
                xxxr = []
                yyyr = []
                for arena_runss in arange(arena_runs):
                    xxxr = concatenate([xxx,xxxr])
                    yyyr = concatenate([yyy,yyyr]) 
                xxx = xxxr
                yyy = yyyr
                
            if((conna == True) and (ii > 0)):
                current_pos = current_pos - array((4,0))
            else:                    
                current_pos = array((xxx[0],yyy[0])).astype(int)  
            
            current_hpc_activity = np.zeros(hpc_numcells)
            if ((ii<1) or (conna == False)):
                current_mec_activity = np.zeros(mec_numcells)
            
            current_context = context_vec[ii]
            current_shape = shape_vec[ii]
            current_vector = lec_whichone(lec_type,lec_change,current_context,current_shape)
            base_lec = np.zeros(current_vector.shape)
            base_lec = lec_activity[0].copy()
            base_lec[current_vector==2] = lec_activity[1][current_vector==2]
            
            for pp in arange(len(xxx),dtype=int): 
            
                print("aaa %d of %d" % (pp,len(xxx)),flush=True)    
            
                current_pos_old = current_pos
                current_pos = array((xxx[pp],yyy[pp])).astype(int)
                current_speed = current_pos - current_pos_old
                
                current_lec_activity = base_lec[:,current_pos[0],current_pos[1]]
                                
                current_lec_noise = np.random.uniform(0.0,lec_noise,current_lec_activity.shape)    
                current_mec_noise = np.random.uniform(0.0,mec_noise,current_mec_activity.shape) 
                current_hpc_noise = np.random.uniform(0.0,hpc_noise,current_hpc_activity.shape)                                
            
                
                lec_inact_vect[ii.astype(int) ,:,xxx[pp].astype(int) ,yyy[pp].astype(int) ] = current_lec_activity   
            
                for kk in arange(theta_cycles):
            
                    if (kk>0): current_speed = array((0,0))    
                        
                    current_mec_input = (current_mec_activity+current_mec_noise)            
                    
                    
                    if (mec_ratio>0.0):
                        for jj in arange(mec_blocks):
                            gxx,gyy = meshgrid(arange(mec_blocksize[jj])+(-1)*current_speed[0],arange(mec_blocksize[jj])+(-1)*current_speed[1])
                            gyy[mod(divide(gxx-mod(gxx,mec_blocksize[jj]),mec_blocksize[jj]),2)>0] = gyy[mod(divide(gxx-mod(gxx,mec_blocksize[jj]),mec_blocksize[jj]),2)>0] + floor(mec_blocksize[jj]/2)
                            gxx = int0(mod(gxx,mec_blocksize[jj]))
                            gyy = int0(mod(gyy,mec_blocksize[jj]))                      
                            current_mec_input[mec_indexlist[jj]]  = current_mec_input[mec_indexlist[jj]][gyy,gxx]             
                            #mec_input_vect[ii,kk,mec_indexlist[jj],xxx[pp],yyy[pp]]  = current_mec_input[mec_indexlist[jj]]               
                         
                    h_h = np.dot(current_hpc_activity+current_hpc_noise,hpc_mec_weights)
                    if(np.max(h_h)>0.0):
                        h_h = h_h/np.max(h_h)
                    h_h[isnan(h_h)] = 0.0
                    current_mec_input = (1-mec_ratio)*h_h + mec_ratio*current_mec_input
                    
                    current_lec_noise = np.random.uniform(0.0,lec_noise,current_lec_activity.shape)    
                    current_mec_noise = np.random.uniform(0.0,mec_noise,current_mec_activity.shape) 
                    current_hpc_noise = np.random.uniform(0.0,hpc_noise,current_hpc_activity.shape)        
              
                    for jj in arange(mec_blocks):                  
                        current_mec_activity[mec_indexlist[jj]] = (current_mec_input[mec_indexlist[jj]] - current_emax*np.max(current_mec_input[mec_indexlist[jj]]))
                        current_mec_activity[current_mec_activity<0] = 0.0
                        current_mec_activity[mec_indexlist[jj]] /= np.max(current_mec_activity[mec_indexlist[jj]])
                        current_mec_activity[isnan(current_mec_activity)] = 0.0  
                        mec_inact_vect[ii.astype(int) ,mec_indexlist[jj.astype(int) ].astype(int) ,xxx[pp].astype(int) ,yyy[pp].astype(int) ] = current_mec_activity[mec_indexlist[jj.astype(int) ].astype(int) ]  
                    

                    h_l = np.dot(current_lec_activity+current_lec_noise,lec_hpc_weights) 
                    h_l = h_l/np.max(h_l)
                    h_l[isnan(h_l)] = 0.0
                    
                    if(hhhr>0):
                        h_m = np.dot(current_mec_activity+current_mec_noise,mec_hpc_weights) 
                        if(np.max(h_m)>0.0):
                            h_m = h_m/np.max(h_m)
                        h_m[isnan(h_m)] = 0.0
                    
                        current_hpc_input = (1-hhhr)*h_l + hhhr*h_m 
                    else:
                        current_hpc_input = h_l;
                    
                    if (kk>0): 

                        ddd = current_hpc_activity * 0
                        
                        
                        for mm in arange(len(hpc_memories)):
                            ccc = corrcoef(hpc_memories[mm],current_hpc_activity+current_hpc_noise)[0][1]
                            if ccc<hpc_pcompl_th: 
                                ccc=0   
                            else:
                                ddd += hpc_memories[mm]
                        
                        if (np.max(ddd) > 0):
                    
                            ddd = ddd/np.max(ddd)
                            ddd[isnan(ddd)] = 0.0
                            current_hpc_input = (1-mec_ratio)*current_hpc_input + mec_ratio*ddd


                    current_hpc_activity = (current_hpc_input - current_emax*np.max(current_hpc_input))
                    current_hpc_activity[current_hpc_activity<0] = 0.0
                    current_hpc_activity /= np.max(current_hpc_activity)
                    current_hpc_activity[current_hpc_activity<current_emax_plast] = 0
                                                            
                    hpc_inact_vect[ii.astype(int),:,xxx[pp].astype(int),yyy[pp].astype(int)] = current_hpc_activity
            
                    if (lllf[ii]>0):        
                        
                        
                        lec_hpc_weights = normalize_weight(learn_weight(lec_hpc_weights,current_lec_activity+current_lec_noise,current_hpc_activity+current_hpc_noise,current_lrate_lec_hpc),lec_hpc_weights_mean)
                        mec_hpc_weights = normalize_weight(learn_weight(mec_hpc_weights,current_mec_activity+current_mec_noise,current_hpc_activity+current_hpc_noise,current_lrate_mec_hpc),mec_hpc_weights_mean)
                        hpc_mec_weights = normalize_weight(learn_weight(hpc_mec_weights,current_hpc_activity+current_hpc_noise,current_mec_activity+current_mec_noise,current_lrate_hpc_mec),hpc_mec_weights_mean)
                        
                        
            
                if ((lllf[ii]>0) and (hpc_pcompl_th<1.0)):                     
                    hpc_memories.append(current_hpc_activity)
                
                lec_act[:,xxx[pp].astype(int),yyy[pp].astype(int)] = current_lec_activity
                mec_act[:,xxx[pp].astype(int),yyy[pp].astype(int)] = current_mec_activity
                hpc_act[:,xxx[pp].astype(int),yyy[pp].astype(int)] = current_hpc_activity
                
            mec_act_vect.append(mec_act)
            lec_act_vect.append(lec_act)
            hpc_act_vect.append(hpc_act)    
            
            
            
            
# %%
       
    #
    #
    # PREPARE THE SIMULATION... WILL SET THE ENVIRONMENTAL VARIABLES FOR EACH SESSION OF THE PROTOCOL
    #                           EMULATES THE MORPHING PROTOCOL
    #
    #
       
       
    mooo = 0.9999 * np.ones((108))   
    hooo = hpc_ratio * np.ones((108))   
    shape_vec =  0.0 * np.ones((108))
    context_vec =  0.0 * np.ones((108))
    lllf = 0.0 * np.ones((108))
    
    pzzz = np.concatenate((arange(1),arange(1),arange(0,16),arange(0,16),arange(0,16),arange(0,16),arange(0,21),arange(0,21),arange(0,21),arange(0,21),arange(0,21),arange(0,21)))    
    
    lllf[0] = 0.0
    mooo[0] = mec_ratio
    mooo[18:34] = mec_ratio
    
    lllf[1] = 0.0
    mooo[1] = mec_ratio
    mooo[50:67] = mec_ratio
    mooo[66:] = mec_ratio
    
    hooo[87:108] = 0.0

    shape_vec[1] = 1.0
    shape_vec[34:66] = 1.0
    shape_vec[66:87]=np.linspace(0.0,1.0,21)
    shape_vec[87:108]=np.linspace(0.0,1.0,21)
 
    nono = 0.0 * np.ones((108))

    
    
# %%    
    
    #
    #
    # RUN THE SIMULATION... 
    #                           
    #
    #


    num_runsss = 1       
        
    corrVectMEC1 = -1* ones(num_runsss)
    corrVectHPC1 = -1* ones(num_runsss)
    corrVectMECGRID1 = -1* ones(num_runsss)
    corrVectHPCGRID1 = -1* ones(num_runsss)
    corrVectMECvsGRID1 = -1* ones(num_runsss)
    
    corrVectMEC2 = -1* ones(num_runsss)
    corrVectHPC2 = -1* ones(num_runsss)
    corrVectMECGRID2 = -1* ones(num_runsss)
    corrVectHPCGRID2 = -1* ones(num_runsss)
    corrVectMECvsGRID2 = -1* ones(num_runsss)
    
    
    corrVectMECx = -1* ones(num_runsss)
    corrVectHPCx = -1* ones(num_runsss)
    corrVectMECGRIDx = -1* ones(num_runsss)
    corrVectHPCGRIDx = -1* ones(num_runsss)
    
    dist_pf1 = -1*ones((num_runsss,16))
    dist_pf2 = -1*ones((num_runsss,16))
    
    pvCorrelationCurveHPC1 = -1*ones((num_runsss,21))
    pvCorrelationCurveMEC1 = -1*ones((num_runsss,21))
    pvCorrelationCurveHPC2 = -1*ones((num_runsss,21))
    pvCorrelationCurveMEC2 = -1*ones((num_runsss,21))
    pvCorrelationCurveHPC = -1*ones((num_runsss,21))
    pvCorrelationCurveMEC = -1*ones((num_runsss,21))
    
    pvCorrelationCurveHPC1Lesion = -1*ones((num_runsss,21))
    pvCorrelationCurveMEC1Lesion = -1*ones((num_runsss,21))
    pvCorrelationCurveHPC2Lesion = -1*ones((num_runsss,21))
    pvCorrelationCurveMEC2Lesion = -1*ones((num_runsss,21))
    pvCorrelationCurveHPCLesion = -1*ones((num_runsss,21))
    pvCorrelationCurveMECLesion = -1*ones((num_runsss,21))
    
    
    # LOOP FOR EACH PROTOCOL DEFINED EARLIER
    
    for sessions in arange(num_runsss):
        
        print("session %d of %d" % (sessions,num_runsss),flush=True)   
            
        pzzz[0] = sessions + 16 + pre_runs
        pzzz[1] = sessions + 16 + pre_runs + num_runsss

        lec_act_vect = []
        mec_act_vect = []
        hpc_act_vect = []       
        
        mec_inact_vect = np.zeros((len(shape_vec),mec_numcells,arena_binsize[0],arena_binsize[1]))
        hpc_inact_vect = np.zeros((len(shape_vec),hpc_numcells,arena_binsize[0],arena_binsize[1]))
        lec_inact_vect = np.zeros((len(shape_vec),lec_numcells,arena_binsize[0],arena_binsize[1]))

        # RUN FOR EACH SESSION  

        for ii in arange(len(shape_vec)):
            
            print("shape %d of %d" % (ii,len(shape_vec)),flush=True)  
        
            mec_ratio = mooo[ii]
            hpc_ratio = hooo[ii]
                    
            lec_act = zeros((lec_numcells,arena_binsize[0],arena_binsize[1]))
            mec_act = zeros((mec_numcells,arena_binsize[0],arena_binsize[1]))
            hpc_act = zeros((hpc_numcells,arena_binsize[0],arena_binsize[1]))            
            
            # SET TRAJECTORY
            
            xxx,yyy = np.meshgrid(arange(arena_binsize[0]),arange(arena_binsize[0]))
            xxx = xxx.ravel()
            yyy = yyy.ravel()
            ppp = popo[pzzz[ii]]          
            xxx = xxx[ppp]
            yyy = yyy[ppp]  
            
            
            if((conna == True) and (ii == 1)):
                current_pos = current_pos - array((5,0))
            else:                    
                current_pos = array((xxx[0],yyy[0]))  
                
            
            current_hpc_activity = np.zeros(hpc_numcells)
            if ((ii!=1) or (conna == False)):
                current_mec_activity = np.zeros(mec_numcells)
            
            current_hpc_activity = np.zeros(hpc_numcells)
            current_mec_activity = np.zeros(mec_numcells)  
            
            current_context = context_vec[ii]
            current_shape = shape_vec[ii]
            current_vector = lec_whichone(lec_type,lec_change,current_context,current_shape)
            base_lec = np.zeros(current_vector.shape)
            base_lec = lec_activity[0].copy()
            base_lec[current_vector==2] = lec_activity[1][current_vector==2]
            
            #set the random seed     
            np.random.seed(seed_path+ii)
                        
            if (nono[ii]>0.0):
                ttt = floor(lec_numcells*nono[ii]);                
                base_lec[:ttt,:,:] = pow(np.random.uniform(0,1,(ttt,arena_binsize[0],arena_binsize[1])),2)   
            

            # RUN FOR EACH POSITION OF THE TRAJECTORY     
            
            for pp in arange(len(xxx)): 
            
                print("aaa %d of %d" % (pp,len(xxx)),flush=True)    
            
                # COMPUTE SPEED AND LEC ACTIVITY
                
                current_pos_old = current_pos
                current_pos = array((xxx[pp],yyy[pp]))
                current_speed = current_pos - current_pos_old
                current_lec_activity = base_lec[:,current_pos[0],current_pos[1]]
                     
                current_lec_noise = np.random.uniform(0.0,lec_noise,current_lec_activity.shape) 
                current_mec_noise = np.random.uniform(0.0,mec_noise,current_mec_activity.shape) 
                current_hpc_noise = np.random.uniform(0.0,hpc_noise,current_hpc_activity.shape)                                
            
                lec_inact_vect[ii,:,xxx[pp],yyy[pp]] = current_lec_activity   
            
                # RUN FOR EACH THETA CYCLE
            
                for kk in arange(theta_cycles):
            
                    # SET SPEED ZERO FOR THE FIRST POSITION
                    if (kk>0): current_speed = array((0,0))    
                      
                    # COMPUTE THE RECURRENT GRID CELL ACTIVITY - TWISTED TOURUS
                    current_mec_input = (current_mec_activity+current_mec_noise)            
                    if(mec_ratio>0.0):
                        for jj in arange(mec_blocks):
                            gxx,gyy = meshgrid(arange(mec_blocksize[jj])+(-1)*current_speed[0],arange(mec_blocksize[jj])+(-1)*current_speed[1])
                            gyy[mod(divide(gxx-mod(gxx,mec_blocksize[jj]),mec_blocksize[jj]),2)>0] = gyy[mod(divide(gxx-mod(gxx,mec_blocksize[jj]),mec_blocksize[jj]),2)>0] + floor(mec_blocksize[jj]/2)
                            gxx = int0(mod(gxx,mec_blocksize[jj]))
                            gyy = int0(mod(gyy,mec_blocksize[jj]))                      
                            current_mec_input[mec_indexlist[jj]]  = current_mec_input[mec_indexlist[jj]][gyy,gxx]             
                           
                    
                    # COMPUTE INPUT TO GRID CELLS
                    
                    h_h = np.dot(current_hpc_activity+current_hpc_noise,hpc_mec_weights)
                    h_h = h_h/np.max(h_h)
                    h_h[isnan(h_h)] = 0.0
                    current_mec_input = (1-mec_ratio)*h_h + mec_ratio*current_mec_input
                    
                    current_lec_noise = np.random.uniform(0.0,lec_noise,current_lec_activity.shape)    
                    current_mec_noise = np.random.uniform(0.0,mec_noise,current_mec_activity.shape) 
                    current_hpc_noise = np.random.uniform(0.0,hpc_noise,current_hpc_activity.shape)        
              
                    # COMPUTE e% MAX OF GRID CELLS
                    
                    for jj in arange(mec_blocks):                  
                        current_mec_activity[mec_indexlist[jj]] = (current_mec_input[mec_indexlist[jj]] - current_emax*np.max(current_mec_input[mec_indexlist[jj]]))
                        current_mec_activity[current_mec_activity<0] = 0.0
                        current_mec_activity[mec_indexlist[jj]] /= np.max(current_mec_activity[mec_indexlist[jj]])
                        current_mec_activity[isnan(current_mec_activity)] = 0.0  
                        mec_inact_vect[ii,mec_indexlist[jj],xxx[pp],yyy[pp]] = current_mec_activity[mec_indexlist[jj]]  
                    
                    
                    # COMPUTE THE PLACE CELLS INPUT
                    h_l = np.dot(current_lec_activity+current_lec_noise,lec_hpc_weights) 
                    h_l = h_l/np.max(h_l)
                    h_l[isnan(h_l)] = 0.0 
                    h_m = np.dot(current_mec_activity+current_mec_noise,mec_hpc_weights) 
                    h_m = h_m/np.max(h_m)
                    h_m[isnan(h_m)] = 0.0
                    current_hpc_input = (1-hpc_ratio)*h_l + hpc_ratio*h_m 
                    
                    # THE PATTERN COMPLETION ALGORITHM
                    if (kk>0): 

                        ddd = current_hpc_activity * 0  
                        
                        for mm in arange(len(hpc_memories)):
                            ccc = corrcoef(hpc_memories[mm],current_hpc_activity+current_hpc_noise)[0][1]
                            if ccc<hpc_pcompl_th: 
                                ccc=0   
                            else:
                                ddd += hpc_memories[mm]

                        
                        if (np.max(ddd) > 0):
                    
                            ddd = ddd/np.max(ddd)
                            ddd[isnan(ddd)] = 0.0
                            current_hpc_input = (1-mec_ratio)*current_hpc_input + mec_ratio*ddd
                    
                    # COMPUTE THE e% MAX OF PLACE CELLS       
                    current_hpc_activity = (current_hpc_input - current_emax*np.max(current_hpc_input))
                    current_hpc_activity[current_hpc_activity<0] = 0.0
                    current_hpc_activity /= np.max(current_hpc_activity)
                    current_hpc_activity[current_hpc_activity<current_emax_plast] = 0
                    
                    hpc_inact_vect[ii,:,xxx[pp],yyy[pp]] = current_hpc_activity
            
                    # IF LEARNING IS SET, UPDATE THE WEIGHTS
            
                    if (lllf[ii]>0):        
                        
                        lec_hpc_weights = normalize_weight(learn_weight(lec_hpc_weights,current_lec_activity+current_lec_noise,current_hpc_activity+current_hpc_noise,current_lrate_lec_hpc),lec_hpc_weights_mean)
                        mec_hpc_weights = normalize_weight(learn_weight(mec_hpc_weights,current_mec_activity+current_mec_noise,current_hpc_activity+current_hpc_noise,current_lrate_mec_hpc),mec_hpc_weights_mean)
                        hpc_mec_weights = normalize_weight(learn_weight(hpc_mec_weights,current_hpc_activity+current_hpc_noise,current_mec_activity+current_mec_noise,current_lrate_hpc_mec),hpc_mec_weights_mean)
                                         
            
                # SAVE PATTERN COMPLETION PATTERN
                if ((lllf[ii]>0) and (hpc_pcompl_th<1.0)):                     
                    hpc_memories.append(current_hpc_activity)
                
                lec_act[:,xxx[pp],yyy[pp]] = current_lec_activity
                mec_act[:,xxx[pp],yyy[pp]] = current_mec_activity
                hpc_act[:,xxx[pp],yyy[pp]] = current_hpc_activity
                
            mec_act_vect.append(mec_act)
            lec_act_vect.append(lec_act)
            hpc_act_vect.append(hpc_act)
            
            
  
# COLLECT THE STATISTICS AND SAVE            
       
            
        ooo1a = np.zeros((16,16))
        ooo2a = np.zeros((16,16))
        ooo3a = np.zeros((16,16))
        ooo4a = np.zeros((16,16))
        ooo5a = np.zeros((16))
        
        ooo1b = np.zeros((16,16))
        ooo2b = np.zeros((16,16))
        ooo3b = np.zeros((16,16))
        ooo4b = np.zeros((16,16))
        ooo5b = np.zeros((16))
        
        ooo1c = np.zeros((16,16))
        ooo2c = np.zeros((16,16))
        ooo3c = np.zeros((16,16))
        ooo4c = np.zeros((16,16))

        pfdist1 = np.zeros((16))
        pfdist2 = np.zeros((16))

       

        for xx in arange(16):
            
            pfdist1 = pfdist1 + np.histogram(np.sum(np.sum(hpc_inact_vect[xx+18,:,:,:]>0,axis=1),axis=1),arange(17))[0]
            pfdist2 = pfdist2 + np.histogram(np.sum(np.sum(hpc_inact_vect[xx+50,:,:,:]>0,axis=1),axis=1),arange(17))[0]
            
            vvv5a = np.zeros((arena_binsize[0],arena_binsize[1]))
            vvv5b = np.zeros((arena_binsize[0],arena_binsize[1]))
             
            for ii in arange(arena_binsize[0]):
                for jj in arange(arena_binsize[1]):
                    vvv5a[ii,jj] = np.corrcoef(mec_inact_vect[xx+2,:,ii,jj],mec_inact_vect[xx+18,:,ii,jj])[0,1]
                    vvv5b[ii,jj] = np.corrcoef(mec_inact_vect[xx+34,:,ii,jj],mec_inact_vect[xx+50,:,ii,jj])[0,1]
            
            ooo5a[xx] = np.mean(vvv5a)
            ooo5b[xx] = np.mean(vvv5b)
            
            
            for yy in arange(xx,16):
                
                vvv1a = np.zeros((arena_binsize[0],arena_binsize[1]))
                vvv2a = np.zeros((arena_binsize[0],arena_binsize[1]))
                vvv3a = np.zeros((arena_binsize[0],arena_binsize[1]))
                vvv4a = np.zeros((arena_binsize[0],arena_binsize[1]))
                
                vvv1b = np.zeros((arena_binsize[0],arena_binsize[1]))
                vvv2b = np.zeros((arena_binsize[0],arena_binsize[1]))
                vvv3b = np.zeros((arena_binsize[0],arena_binsize[1]))
                vvv4b = np.zeros((arena_binsize[0],arena_binsize[1]))
                
                vvv1c = np.zeros((arena_binsize[0],arena_binsize[1]))
                vvv2c = np.zeros((arena_binsize[0],arena_binsize[1]))
                vvv3c = np.zeros((arena_binsize[0],arena_binsize[1]))
                vvv4c = np.zeros((arena_binsize[0],arena_binsize[1]))
                
                vvv1d = np.zeros((arena_binsize[0],arena_binsize[1]))
                vvv2d = np.zeros((arena_binsize[0],arena_binsize[1]))
                vvv3d = np.zeros((arena_binsize[0],arena_binsize[1]))
                vvv4d = np.zeros((arena_binsize[0],arena_binsize[1]))
                 
                for ii in arange(arena_binsize[0]):
                    for jj in arange(arena_binsize[1]):
                        vvv1a[ii,jj] = np.corrcoef(mec_inact_vect[xx+2,:,ii,jj],mec_inact_vect[yy+2,:,ii,jj])[0,1]
                        vvv2a[ii,jj] = np.corrcoef(hpc_inact_vect[xx+2,:,ii,jj],hpc_inact_vect[yy+2,:,ii,jj])[0,1]
                        vvv3a[ii,jj] = np.corrcoef(mec_inact_vect[xx+18,:,ii,jj],mec_inact_vect[yy+18,:,ii,jj])[0,1]
                        vvv4a[ii,jj] = np.corrcoef(hpc_inact_vect[xx+18,:,ii,jj],hpc_inact_vect[yy+18,:,ii,jj])[0,1]

                        vvv1b[ii,jj] = np.corrcoef(mec_inact_vect[xx+34,:,ii,jj],mec_inact_vect[yy+34,:,ii,jj])[0,1]
                        vvv2b[ii,jj] = np.corrcoef(hpc_inact_vect[xx+34,:,ii,jj],hpc_inact_vect[yy+34,:,ii,jj])[0,1]
                        vvv3b[ii,jj] = np.corrcoef(mec_inact_vect[xx+50,:,ii,jj],mec_inact_vect[yy+50,:,ii,jj])[0,1]
                        vvv4b[ii,jj] = np.corrcoef(hpc_inact_vect[xx+50,:,ii,jj],hpc_inact_vect[yy+50,:,ii,jj])[0,1]
                        
                        vvv1c[ii,jj] = np.corrcoef(mec_inact_vect[xx+2,:,ii,jj],mec_inact_vect[yy+34,:,ii,jj])[0,1]
                        vvv2c[ii,jj] = np.corrcoef(hpc_inact_vect[xx+2,:,ii,jj],hpc_inact_vect[yy+34,:,ii,jj])[0,1]
                        vvv3c[ii,jj] = np.corrcoef(mec_inact_vect[xx+18,:,ii,jj],mec_inact_vect[yy+50,:,ii,jj])[0,1]
                        vvv4c[ii,jj] = np.corrcoef(hpc_inact_vect[xx+18,:,ii,jj],hpc_inact_vect[yy+50,:,ii,jj])[0,1]
                        
                        vvv1d[ii,jj] = np.corrcoef(mec_inact_vect[xx+34,:,ii,jj],mec_inact_vect[yy+2,:,ii,jj])[0,1]
                        vvv2d[ii,jj] = np.corrcoef(hpc_inact_vect[xx+34,:,ii,jj],hpc_inact_vect[yy+2,:,ii,jj])[0,1]
                        vvv3d[ii,jj] = np.corrcoef(mec_inact_vect[xx+50,:,ii,jj],mec_inact_vect[yy+18,:,ii,jj])[0,1]
                        vvv4d[ii,jj] = np.corrcoef(hpc_inact_vect[xx+50,:,ii,jj],hpc_inact_vect[yy+18,:,ii,jj])[0,1]
                      
                      
                ooo1a[xx,yy] = np.mean(vvv1a)
                ooo1a[yy,xx] = np.mean(vvv1a)
                ooo2a[xx,yy] = np.mean(vvv2a)
                ooo2a[yy,xx] = np.mean(vvv2a)
                ooo3a[xx,yy] = np.mean(vvv3a)
                ooo3a[yy,xx] = np.mean(vvv3a)
                ooo4a[xx,yy] = np.mean(vvv4a)
                ooo4a[yy,xx] = np.mean(vvv4a)
                
                ooo1b[xx,yy] = np.mean(vvv1b)
                ooo1b[yy,xx] = np.mean(vvv1b)
                ooo2b[xx,yy] = np.mean(vvv2b)
                ooo2b[yy,xx] = np.mean(vvv2b)
                ooo3b[xx,yy] = np.mean(vvv3b)
                ooo3b[yy,xx] = np.mean(vvv3b)
                ooo4b[xx,yy] = np.mean(vvv4b)
                ooo4b[yy,xx] = np.mean(vvv4b)
                
                ooo1c[xx,yy] = np.mean(vvv1c)
                ooo1c[yy,xx] = np.mean(vvv1d)
                ooo2c[xx,yy] = np.mean(vvv2c)
                ooo2c[yy,xx] = np.mean(vvv2d)
                ooo3c[xx,yy] = np.mean(vvv3c)
                ooo3c[yy,xx] = np.mean(vvv3d)
                ooo4c[xx,yy] = np.mean(vvv4c)
                ooo4c[yy,xx] = np.mean(vvv4d)
                
     
        
        corrVectMECGRID1[sessions] =  np.mean(ooo1a)
        corrVectHPCGRID1[sessions] =  np.mean(ooo2a)
        corrVectMEC1[sessions] =  np.mean(ooo3a)
        corrVectHPC1[sessions] =  np.mean(ooo4a)
        corrVectMECvsGRID1[sessions] =  np.mean(ooo5a)
    
        corrVectMECGRID2[sessions] =  np.mean(ooo1b)
        corrVectHPCGRID2[sessions] =  np.mean(ooo2b)
        corrVectMEC2[sessions] =  np.mean(ooo3b)
        corrVectHPC2[sessions] =  np.mean(ooo4b)
        corrVectMECvsGRID2[sessions] =  np.mean(ooo5b)
    
        corrVectMECGRIDx[sessions] =  np.mean(ooo1c)
        corrVectHPCGRIDx[sessions] =  np.mean(ooo2c) 
        corrVectMECx[sessions] =  np.mean(ooo3c)
        corrVectHPCx[sessions] =  np.mean(ooo4c)
      
        dist_pf1[sessions,:] = pfdist1
        dist_pf2[sessions,:] = pfdist2
        
        
        
        for xx in arange(21):
            ooo1a = np.zeros(arena_binsize)
            ooo2a = np.zeros(arena_binsize)
            ooo1b = np.zeros(arena_binsize)
            ooo2b = np.zeros(arena_binsize)
            ooo3a = np.zeros(arena_binsize)
            ooo3b = np.zeros(arena_binsize)
    
            for ii in arange(arena_binsize[0]):
                for jj in arange(arena_binsize[1]):
                    ooo1a[ii,jj] = np.corrcoef(hpc_inact_vect[66,:,ii,jj],hpc_inact_vect[xx+66,:,ii,jj])[0,1]
                    ooo1b[ii,jj] = np.corrcoef(mec_inact_vect[66,:,ii,jj],mec_inact_vect[xx+66,:,ii,jj])[0,1]
                    ooo2a[ii,jj] = np.corrcoef(hpc_inact_vect[86,:,ii,jj],hpc_inact_vect[86-xx,:,ii,jj])[0,1]
                    ooo2b[ii,jj] = np.corrcoef(mec_inact_vect[86,:,ii,jj],mec_inact_vect[86-xx,:,ii,jj])[0,1]
                    ooo3a[ii,jj] = np.mean((ooo1a[ii,jj],ooo2a[ii,jj]))
                    ooo3b[ii,jj] = np.mean((ooo1b[ii,jj],ooo2b[ii,jj]))
                    
            pvCorrelationCurveHPC1[sessions,xx] = np.mean(ooo1a)
            pvCorrelationCurveMEC1[sessions,xx] = np.mean(ooo1b)
            pvCorrelationCurveHPC2[sessions,xx] = np.mean(ooo2a)
            pvCorrelationCurveMEC2[sessions,xx] = np.mean(ooo2b)
            pvCorrelationCurveHPC[sessions,xx] = np.mean(ooo3a)
            pvCorrelationCurveMEC[sessions,xx] = np.mean(ooo3b)
            
        if (acts==True):
            actvLec1 = lec_inact_vect[66,0:100,:,:]
            actvLec2 = lec_inact_vect[86,0:100,:,:]
            actvMec1 = mec_inact_vect[66,0:100,:,:]
            actvMec2 = mec_inact_vect[86,0:100,:,:]
            actvHpc1 = hpc_inact_vect[66,:,:,:]
            actvHpc2 = hpc_inact_vect[86,:,:,:]
            with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,9)+'z', 'wb') as ff:
                pickle.dump([actvLec1,actvLec2,actvMec1,actvMec2,actvHpc1,actvHpc2] , ff)
        
            
            
        for xx in arange(21):
            ooo1a = np.zeros(arena_binsize)
            ooo2a = np.zeros(arena_binsize)
            ooo1b = np.zeros(arena_binsize)
            ooo2b = np.zeros(arena_binsize)
            ooo3a = np.zeros(arena_binsize)
            ooo3b = np.zeros(arena_binsize)
    
            for ii in arange(arena_binsize[0]):
                for jj in arange(arena_binsize[1]):
                    ooo1a[ii,jj] = np.corrcoef(hpc_inact_vect[87,:,ii,jj],hpc_inact_vect[xx+87,:,ii,jj])[0,1]
                    ooo1b[ii,jj] = np.corrcoef(mec_inact_vect[87,:,ii,jj],mec_inact_vect[xx+87,:,ii,jj])[0,1]
                    ooo2a[ii,jj] = np.corrcoef(hpc_inact_vect[107,:,ii,jj],hpc_inact_vect[107-xx,:,ii,jj])[0,1]
                    ooo2b[ii,jj] = np.corrcoef(mec_inact_vect[107,:,ii,jj],mec_inact_vect[107-xx,:,ii,jj])[0,1]
                    ooo3a[ii,jj] = np.mean((ooo1a[ii,jj],ooo2a[ii,jj]))
                    ooo3b[ii,jj] = np.mean((ooo1b[ii,jj],ooo2b[ii,jj]))
                    
            pvCorrelationCurveHPC1Lesion[sessions,xx] = np.mean(ooo1a)
            pvCorrelationCurveMEC1Lesion[sessions,xx] = np.mean(ooo1b)
            pvCorrelationCurveHPC2Lesion[sessions,xx] = np.mean(ooo2a)
            pvCorrelationCurveMEC2Lesion[sessions,xx] = np.mean(ooo2b)
            pvCorrelationCurveHPCLesion[sessions,xx] = np.mean(ooo3a)
            pvCorrelationCurveMECLesion[sessions,xx] = np.mean(ooo3b)
            
        
        
        
        with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,0)+'z', 'wb') as ff:
            pickle.dump([corrVectMECGRID1,corrVectHPCGRID1,corrVectMEC1,corrVectHPC1,corrVectMECvsGRID1] , ff)
        with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,1)+'z', 'wb') as ff:
            pickle.dump([corrVectMECGRID2,corrVectHPCGRID2,corrVectMEC2,corrVectHPC2,corrVectMECvsGRID2] , ff)
        with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,2)+'z', 'wb') as ff:
            pickle.dump([corrVectMECGRIDx,corrVectHPCGRIDx,corrVectMECx,corrVectHPCx] , ff)
        with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,3)+'z', 'wb') as ff:
            pickle.dump([dist_pf1,dist_pf2] , ff)
        with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,4)+'z', 'wb') as ff:
            pickle.dump([pvCorrelationCurveHPC,pvCorrelationCurveHPC1,pvCorrelationCurveHPC2,pvCorrelationCurveMEC,pvCorrelationCurveMEC1,pvCorrelationCurveMEC2] , ff)
        with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,5)+'z', 'wb') as ff:
            pickle.dump([pvCorrelationCurveHPCLesion,pvCorrelationCurveHPC1Lesion,pvCorrelationCurveHPC2Lesion,pvCorrelationCurveMECLesion,pvCorrelationCurveMEC1Lesion,pvCorrelationCurveMEC2Lesion] , ff)


        if(sessions>5):
            
            if(np.max(np.abs(np.diff(corrVectMECGRID1[(sessions)-3:sessions])))==0.0):
                if(np.max(np.abs(np.diff(corrVectMECGRID2[(sessions)-3:sessions])))==0.0):
                    if(np.max(np.abs(np.diff(corrVectMECGRIDx[(sessions)-3:sessions])))==0.0):    

                        corrVectMECGRID1[(sessions+1):] = -2
                        corrVectHPCGRID1[(sessions+1):] = -2
                        corrVectMEC1[(sessions+1):] = -2
                        corrVectHPC1[(sessions+1):] = -2
                        corrVectMECvsGRID1[(sessions+1):] = -2
                        
                        corrVectMECGRID2[(sessions+1):] = -2
                        corrVectHPCGRID2[(sessions+1):] = -2
                        corrVectMEC2[(sessions+1):] = -2
                        corrVectHPC2[(sessions+1):] = -2
                        corrVectMECvsGRID2[(sessions+1):] = -2

                        corrVectMECGRIDx[(sessions+1):] = -2
                        corrVectHPCGRIDx[(sessions+1):] = -2
                        corrVectMECx[(sessions+1):] = -2
                        corrVectHPCx[(sessions+1):] = -2
                        
                        with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,0)+'z', 'wb') as ff:
                            pickle.dump([corrVectMECGRID1,corrVectHPCGRID1,corrVectMEC1,corrVectHPC1,corrVectMECvsGRID1] , ff)
                        with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,1)+'z', 'wb') as ff:
                            pickle.dump([corrVectMECGRID2,corrVectHPCGRID2,corrVectMEC2,corrVectHPC2,corrVectMECvsGRID2] , ff)
                        with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,2)+'z', 'wb') as ff:
                            pickle.dump([corrVectMECGRIDx,corrVectHPCGRIDx,corrVectMECx,corrVectHPCx] , ff)
                        
                        return



if __name__ == "__main__":
   main(sys.argv[1:])
   
        
    
        
    
    
    
    
    
    













