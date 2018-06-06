# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 10:32:29 2015

@author: rennocosta

This script is part of the publication Renno-Costa & Tort, 2017, JNeurosci
This script relates to the data presented in the Figures 3 and 4
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

### function to normalize synaptic weights after learning

def normalize_weight(www,www_mean):
    www /= np.tile(np.mean(www,axis=0),(www.shape[0],1)) 
    return www

### learning rule 
   
def learn_weight(www,activity_pre,activity_pos,lrate):
    www += lrate*(np.tile(activity_pre,(activity_pos.shape[0],1)).transpose()) * (np.tile(activity_pos,(activity_pre.shape[0],1)))   
    return www

### function to manipulate the non-grid cell input accordingly to the position and context

def lec_whichone(lectype,change,ccc,sss):
    saida = np.zeros(lectype.shape)
    saida[np.logical_and(lectype==1,change<ccc)] = 2
    saida[np.logical_and(lectype==1,change>=ccc)] = 1
    saida[np.logical_and(lectype==0,change<sss)] = 2
    saida[np.logical_and(lectype==0,change>=sss)] = 1
    saida[lectype==2] = 1
    return saida

###

def main(argv):
    
    # will parse the arguments
    
    parser = argparse.ArgumentParser(description='Will run a simulation instance.')
  
  	# seed values for non-grid cell input activity pattern, initial synaptic weights value and path
                   
    parser.add_argument('seed_input', metavar='seed_input', type=int, nargs=1,
                   help='seed_input number') 
    parser.add_argument('seed_www', metavar='seed_www', type=int, nargs=1,
                   help='seed_www') 
    parser.add_argument('seed_path', metavar='seed_path', type=int, nargs=1,
                   help='seed_path') 
  
  	# length of theta cycles 
  
    parser.add_argument('theta_cycles', metavar='theta_cycles', type=int, nargs=1,
                   help='theta_cycles')
    parser.add_argument('arena_runs', metavar='arena_runs', type=int, nargs=1,
                   help='arena_runs')  
                   
                   
    parser.add_argument('pre_runs', metavar='pre_runs', type=int, nargs=1,
                   help='pre_runs')    
    parser.add_argument('true_runs', metavar='true_runs', type=int, nargs=1,
                   help='true_runs')    
                   
    # learning rate for place cells -> grid cells; grid cells -> place cells; non-grid cells -> place cells           

    parser.add_argument('lrate_hpc_mec', metavar='lrate_hpc_mec', type=int, nargs=1,
                   help='lrate_hpc_mec')                        
    parser.add_argument('lrate_mec_hpc', metavar='lrate_mec_hpc', type=int, nargs=1,
                   help='lrate_mec_hpc')                
    parser.add_argument('lrate_lec_hpc', metavar='lrate_lec_hpc', type=int, nargs=1,
                   help='lrate_lec_hpc')    
                   
    # relative values for the populations inputs: mec ratio (recurrent vs place cell input); 
    # hpc ratio (grid cell vs non-gridcell input);  hippocampus pattern completion threshold
         
    parser.add_argument('mec_ratio', metavar='mec_ratio', type=int, nargs=1,
                   help='MEC ratio (x100)')
    parser.add_argument('hpc_ratio', metavar='hpc_ratio', type=int, nargs=1,
                   help='HPC ratio (x100)')
    parser.add_argument('hpc_pcompl_th', metavar='hpc_pcompl_th', type=int, nargs=1,
                   help='HPC pattern completion th (x100)')


    parser.add_argument('morph_per', metavar='morph_per', type=int, nargs=1,
                   help='morph_per') 
             
                    
    # define the path for saving the results   
    
    parser.add_argument('-w', '--windows',dest='envir',action='store_const',default="cluster",const="windows")
    parser.add_argument('-u', '--ufrgs',dest='envir',action='store_const',default="cluster",const="UFRGS")
    parser.add_argument('-s', '--npad',dest='envir',action='store_const',default="cluster",const="NPAD")
    
    
    # without -c, will run with 1 memory (morph from memory to noise). with -c, will run with two memories   
    parser.add_argument('-c', '--connected',dest='conntype',action='store_const',default="no",const="yes")    
    
    # will save the activity of population and not only the statistics (be aware of file size)   
    parser.add_argument('-a', '--activity',dest='actsave',action='store_const',default="no",const="yes")
    
    # will kill simulation if files already exists
    parser.add_argument('-k', '--KILL',dest='tokill',action='store_const',default="no",const="yes")
    
    
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
        
    if (actsave=="yes"):
        actsaveb = True
    else:
        actsaveb = False
    
    seed_input = args.seed_input[0]
    seed_www = args.seed_www[0]
    seed_path = args.seed_path[0]
    mec_ratio = float(args.mec_ratio[0])/100
    hpc_ratio = float(args.hpc_ratio[0])/100
    hpc_pcompl_th = float(args.hpc_pcompl_th[0])/100
    morphing_per = float(args.morph_per[0])/100
    pre_runs = args.pre_runs[0]
    true_runs = args.true_runs[0]
    lrate_hpc_mec = float(args.lrate_hpc_mec[0])/1000
    lrate_mec_hpc = float(args.lrate_mec_hpc[0])/1000
    lrate_lec_hpc = float(args.lrate_lec_hpc[0])/1000
    theta_cycles = args.theta_cycles[0]
    arena_runs = args.arena_runs[0]
    
	# define the number of the simulation
    simulation_num = 68
  
    listofvalues = [ct,args.seed_input[0],args.seed_www[0],args.seed_path[0],args.theta_cycles[0],args.arena_runs[0],args.pre_runs[0],args.true_runs[0],args.lrate_hpc_mec[0],args.lrate_mec_hpc[0],args.lrate_lec_hpc[0],args.mec_ratio[0],args.hpc_ratio[0],args.hpc_pcompl_th[0],args.morph_per[0]]                 
                    
      
    filenames = rfn.remappingFileNames(envir)
    filenames.prepareSimulation(listofvalues,simulation_num)  
    
    if (tokill == "no"):
        try:
            tosee = 0;
            with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,2)+'z', 'rb') as ff:
                tosee = tosee + 1
            with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,2)+'z', 'rb') as ff:
                tosee = tosee + 1
            with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,3)+'z', 'rb') as ff:
                tosee = tosee + 1                  

            print("File exist. Will exit!")
            torun = 0;
        except:
            print("File does not existe. Will do!")  
            print("... %s" % (filenames.fileRunPickle(listofvalues,simulation_num,0)))   
            torun = 1;
    else:
        print("Will do anyway!")
        torun = 1;
        

    if(torun == 0):
        sys.exit();

    # %% will setup the network
    
    arena_binsize = [1,2]
    
    context_per = 0
    
    lec_numcells = 500
    hpc_numcells = 5000
    
    # will setup the non-grid cell input patterns
    
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
    
    hpc_memories = []  
    mec_blocksize = [2,4,6,8,10,12,14,16]
    mec_blocks = len(mec_blocksize)
    mec_numcells = np.sum(np.power(mec_blocksize,2))
    mec_indexlist = []
    init_val = 0
    for ii in arange(mec_blocks):
        mec_indexlist.append((init_val+arange(pow(mec_blocksize[ii],2))).reshape((mec_blocksize[ii],mec_blocksize[ii]))) 
        init_val = np.max(mec_indexlist[ii])+1
    del(init_val)
    
    # will setup the paths
    
    np.random.seed(seed_path)
    
    xxx,yyy = np.meshgrid(arange(arena_binsize[0]),arange(arena_binsize[1]))
    xxx = xxx.ravel()
    popo = []
    
    for ii in arange(100):
        popo.append(np.array([0,1]))
    
    # will setup the initial weights
    
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
    
            
# %% will setup the protocol
       
    # 0:20 : morphing 
    # 21:22+2*runs : learn #1 and #2
    # 2*runs+21:41 : morphing 
    # 2*runs+42:62 : morphin - hpc_lesion
       
    nummorphs = 41   
    
    sublento = (ct+1)*pre_runs + nummorphs      
    lento = nummorphs + sublento * true_runs   
       
    mooo = mec_ratio * np.ones((lento))   #[0.9999,0.9999,0.9999,0.9999,0.9999,0.9999,0.9999,0.9999]
    hooo = hpc_ratio * np.ones((lento))   #[0.9999,0.9999,0.9999,0.9999,0.9999,0.9999,0.9999,0.9999]
    shape_vec =  0.0 * np.ones((lento)) # [0.0,0.0,  0.0,0.0,0.0,1.0,1.0,1.0   ]
    context_vec =  0.0 * np.ones((lento)) #[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.0,0.5]
    lllf = 0.0 * np.ones((lento))#[1,1,   0,0,0,0,0,0,]  
    hppp = hpc_pcompl_th * np.ones((lento))
    
    r_start = concatenate([[0],nummorphs+(arange(true_runs)*sublento)+(ct+1)*pre_runs])
    r_end =  r_start.copy() +   nummorphs - 1     
    
    shape_vec[r_start[0]:(r_end[0]+1)] = np.linspace(0.0,1.0,nummorphs)    
    
    for ii in arange(true_runs):    
        lentoi = nummorphs + sublento * ii
        lllf[lentoi:(lentoi + (ct+1)*pre_runs)] = 1.0
        if (conna == True):
            shape_vec[(lentoi+1):(lentoi + (ct+1)*pre_runs):2] = 1.0
    
        shape_vec[r_start[ii+1]:(r_end[ii+1]+1)] = np.linspace(0.0,1.0,nummorphs)
        
    nono = 0.0 * np.ones((lento))

    convergethetatime = -1
        
# %% will run protocol
    
    num_runsss = 1       
        
    MECconverge = -1*ones((lento,2,theta_cycles))
    HPCconverge = -1*ones((lento,2,theta_cycles))  
    MECconvergeDist = -1*ones((lento,2,theta_cycles))
    HPCconvergeDist = -1*ones((lento,2,theta_cycles))  
    
    MECconvergetime = -1*ones((lento,2))
    HPCconvergetime = -1*ones((lento,2))  
    MECconvergeDisttime = -1*ones((lento,2))
    HPCconvergeDisttime = -1*ones((lento,2))  
    
    
    if (actsaveb):
        LECactivity = -1*ones((3,2,100))
        MECactivity = -1*ones((3,2,mec_numcells,theta_cycles))
        HPCactivity = -1*ones((3,2,100,theta_cycles)) 
    
    pvCorrelationCurveHPC1 = -1*ones((true_runs+1,2,nummorphs))
    pvCorrelationCurveMEC1 = -1*ones((true_runs+1,2,nummorphs))
    pvCorrelationCurveLEC1 = -1*ones((true_runs+1,2,nummorphs))
  
    pvCorrelationCurveHPC2 = -1*ones((true_runs+1,2,nummorphs))
    pvCorrelationCurveMEC2 = -1*ones((true_runs+1,2,nummorphs))
    pvCorrelationCurveLEC2 = -1*ones((true_runs+1,2,nummorphs))
    
    
    netconverged = False    
    
    for sessions in arange(num_runsss):
        
        print("session %d of %d" % (sessions,num_runsss))   
                   
        lec_act_vect = []
        mec_act_vect = []
        hpc_act_vect = []       
        
        mec_inact_vect = np.zeros((len(shape_vec),mec_numcells,arena_binsize[0],arena_binsize[1]))
        hpc_inact_vect = np.zeros((len(shape_vec),hpc_numcells,arena_binsize[0],arena_binsize[1]))
        lec_inact_vect = np.zeros((len(shape_vec),lec_numcells,arena_binsize[0],arena_binsize[1]))

        mec_reff_vect = np.zeros((mec_numcells))
        hpc_reff_vect = np.zeros((hpc_numcells))


        for ii in arange(len(shape_vec)):
                                    
            print("shape %d of %d" % (ii,len(shape_vec)))  

            if ((netconverged == False) or (netconverged == True) or (lllf[ii]==0)) :
        
                mec_ratio = mooo[ii]
                hpc_ratio = hooo[ii]
                        
                lec_act = zeros((lec_numcells,arena_binsize[0],arena_binsize[1]))
                mec_act = zeros((mec_numcells,arena_binsize[0],arena_binsize[1]))
                hpc_act = zeros((hpc_numcells,arena_binsize[0],arena_binsize[1]))            
                      
                xxx,yyy = np.meshgrid(arange(arena_binsize[0]),arange(arena_binsize[1]))
                xxx = xxx.ravel()
                yyy = yyy.ravel()       
                ppp = np.array([0,1])            
                xxx = xxx[ppp]
                yyy = yyy[ppp]
                
                if (lllf[ii]>0):
                    xxxr = []
                    yyyr = []
                    for arena_runss in arange(arena_runs):
                        xxxr = concatenate([xxx,xxxr])
                        yyyr = concatenate([yyy,yyyr]) 
                    xxx = xxxr
                    yyy = yyyr
                        
                
                current_pos = array((xxx[0],yyy[0]))  
                current_mec_activity = np.zeros(mec_numcells)  
                
                
                current_hpc_activity = np.zeros(hpc_numcells)
                
                current_context = context_vec[ii]
                current_shape = shape_vec[ii]
                current_vector = lec_whichone(lec_type,lec_change,current_context,current_shape)
                base_lec = np.zeros(current_vector.shape)
                base_lec = lec_activity[0].copy()
                base_lec[current_vector==2] = lec_activity[1][current_vector==2]
                
                #set the random seed     
                np.random.seed(seed_path+int(round(shape_vec[ii]*100)))
                if (nono[ii]>0.0):
                    ttt = floor(lec_numcells*nono[ii]);                
                    base_lec[:ttt,:,:] = pow(np.random.uniform(0,1,(ttt,arena_binsize[0],arena_binsize[1])),2)       
                
                for pp in arange(len(xxx)): 
                
                    print("aaa %d of %d" % (pp,len(xxx)))    
                
                    current_pos_old = current_pos
                    current_pos = array((xxx[pp],yyy[pp]))
                    current_speed = current_pos - current_pos_old
                    current_lec_activity = base_lec[:,current_pos[0],current_pos[1]]
                    
                    
                    current_lec_noise = np.random.uniform(0.0,lec_noise,current_lec_activity.shape) 
                    current_mec_noise = np.random.uniform(0.0,mec_noise,current_mec_activity.shape) 
                    current_hpc_noise = np.random.uniform(0.0,hpc_noise,current_hpc_activity.shape)                                
                
                    lec_inact_vect[ii,:,xxx[pp],yyy[pp]] = current_lec_activity   
                
                    thetaconverge = False
                
                    for kk in arange(theta_cycles):
                        
                        if(thetaconverge == False):
                
                            if (kk>0): 
                                current_speed = array((0,0)) 
                                
                            current_mec_input = (current_mec_activity+current_mec_noise)            
                            
                            
                            h_h = np.dot(current_hpc_activity+current_hpc_noise,hpc_mec_weights)
                            if(np.max(h_h)>0):                            
                                h_h = h_h/np.max(h_h)
                            h_h[isnan(h_h)] = 0.0                            
                            
                            if(mec_ratio>0):
                            
                                for jj in arange(mec_blocks):
                                    gxx,gyy = meshgrid(arange(mec_blocksize[jj])+(-1)*current_speed[0],arange(mec_blocksize[jj])+(-1)*current_speed[1])
                                    gyy[mod(divide(gxx-mod(gxx,mec_blocksize[jj]),mec_blocksize[jj]),2)>0] = gyy[mod(divide(gxx-mod(gxx,mec_blocksize[jj]),mec_blocksize[jj]),2)>0] + floor(mec_blocksize[jj]/2)
                                    gxx = int0(mod(gxx,mec_blocksize[jj]))
                                    gyy = int0(mod(gyy,mec_blocksize[jj]))                      
                                    current_mec_input[mec_indexlist[jj]]  = current_mec_input[mec_indexlist[jj]][gyy,gxx]             
                                
                            
                                current_mec_input = (1-mec_ratio)*h_h + mec_ratio*current_mec_input
                            
                            else:
                                current_mec_input = h_h
                            
                            current_lec_noise = np.random.uniform(0.0,lec_noise,current_lec_activity.shape)    
                            current_mec_noise = np.random.uniform(0.0,mec_noise,current_mec_activity.shape) 
                            current_hpc_noise = np.random.uniform(0.0,hpc_noise,current_hpc_activity.shape)        
                      
                            for jj in arange(mec_blocks):                  
                                current_mec_activity[mec_indexlist[jj]] = (current_mec_input[mec_indexlist[jj]] - current_emax*np.max(current_mec_input[mec_indexlist[jj]]))
                                current_mec_activity[current_mec_activity<0] = 0.0
                                current_mec_activity[mec_indexlist[jj]] /= np.max(current_mec_activity[mec_indexlist[jj]])
                                current_mec_activity[isnan(current_mec_activity)] = 0.0  
                                mec_inact_vect[ii,mec_indexlist[jj],xxx[pp],yyy[pp]] = current_mec_activity[mec_indexlist[jj]]  
                            
        
                            h_l = np.dot(current_lec_activity+current_lec_noise,lec_hpc_weights) 
                            h_l = h_l/np.max(h_l)
                            h_l[isnan(h_l)] = 0.0
                            
                            if(hpc_ratio>0):
                                h_m = np.dot(current_mec_activity+current_mec_noise,mec_hpc_weights) 
                                h_m = h_m/np.max(h_m)
                                h_m[isnan(h_m)] = 0.0
                            
                                current_hpc_input = (1-hpc_ratio)*h_l + hpc_ratio*h_m 
                            else:
                                current_hpc_input = h_l
                            
                                
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
                            
                            
                            hpc_inact_vect[ii,:,xxx[pp],yyy[pp]] = current_hpc_activity
                            
                            if kk==0:
                                MECconverge[ii,pp,0]= 1.0
                                HPCconverge[ii,pp,0]= 1.0
                                MECconvergeDist[ii,pp,0]= 0.0
                                HPCconvergeDist[ii,pp,0]= 0.0
                                if (sessions==0) and (actsaveb):
                                    if ii==44:                        
                                        LECactivity[0,pp,:] = current_lec_activity[0:100]
                                    if ii==54:                        
                                        LECactivity[1,pp,:] = current_lec_activity[0:100]
                                    if ii==64:                        
                                        LECactivity[2,pp,:] = current_lec_activity[0:100]
                            else:
                                MECconverge[ii,pp,kk] = np.corrcoef(mec_reff_vect,current_mec_activity)[0,1]
                                if(MECconverge[ii,pp,kk]>0.999):
                                    if (MECconvergetime[ii,pp]<0):                                
                                        MECconvergetime[ii,pp] = kk-1
                                HPCconverge[ii,pp,kk] = np.corrcoef(hpc_reff_vect,current_hpc_activity)[0,1]
                                if(HPCconverge[ii,pp,kk]>0.999):
                                    if (HPCconvergetime[ii,pp]<0): 
                                        HPCconvergetime[ii,pp] = kk-1
                                MECconvergeDist[ii,pp,kk] = np.sum(np.abs(mec_reff_vect-current_mec_activity))
                                if(MECconvergeDist[ii,pp,kk]<0.1):
                                    if (MECconvergeDisttime[ii,pp]<0): 
                                        MECconvergeDisttime[ii,pp] = kk-1
                                HPCconvergeDist[ii,pp,kk] = np.sum(np.abs(hpc_reff_vect-current_hpc_activity))
                                if(HPCconvergeDist[ii,pp,kk]<0.1):
                                    if (HPCconvergeDisttime[ii,pp]<0): 
                                        HPCconvergeDisttime[ii,pp] = kk-1
                                if (HPCconvergeDisttime[ii,pp]>=0 and HPCconvergetime[ii,pp]>=0  and MECconvergeDisttime[ii,pp]>=0  and MECconvergetime[ii,pp]>=0 ):
                                    thetaconverge = True    
                                    
                                
                            mec_reff_vect = current_mec_activity.copy()
                            hpc_reff_vect = current_hpc_activity.copy()
                            
                            if (actsaveb):
                                if ii==44:
                                    MECactivity[0,pp,:,kk] = current_mec_activity
                                    HPCactivity[0,pp,:,kk] = current_hpc_activity[0:100]
                                if ii==54:
                                    MECactivity[1,pp,:,kk] = current_mec_activity
                                    HPCactivity[1,pp,:,kk] = current_hpc_activity[0:100]
                                if ii==64:
                                    MECactivity[2,pp,:,kk] = current_mec_activity
                                    HPCactivity[2,pp,:,kk] = current_hpc_activity[0:100]
                    
                            if (lllf[ii]>0):        
                                                                
                                lec_hpc_weights = normalize_weight(learn_weight(lec_hpc_weights,current_lec_activity+current_lec_noise,current_hpc_activity+current_hpc_noise,current_lrate_lec_hpc),lec_hpc_weights_mean)
                                mec_hpc_weights = normalize_weight(learn_weight(mec_hpc_weights,current_mec_activity+current_mec_noise,current_hpc_activity+current_hpc_noise,current_lrate_mec_hpc),mec_hpc_weights_mean)
                                hpc_mec_weights = normalize_weight(learn_weight(hpc_mec_weights,current_hpc_activity+current_hpc_noise,current_mec_activity+current_mec_noise,current_lrate_hpc_mec),hpc_mec_weights_mean)
                                                
                
                    if (lllf[ii]>0) and (hppp[ii]<1.0): 
                        ccc=0
                        for mm in arange(len(hpc_memories)):
                            ccc = corrcoef(hpc_memories[mm],current_hpc_activity+current_hpc_noise)[0][1]
                            if ccc>hpc_pcompl_th: 
                                ccc=1
                        if ccc == 0:
                            hpc_memories.append(current_hpc_activity)
                    
                    lec_act[:,xxx[pp],yyy[pp]] = current_lec_activity
                    mec_act[:,xxx[pp],yyy[pp]] = current_mec_activity
                    hpc_act[:,xxx[pp],yyy[pp]] = current_hpc_activity
                
                                                
                mec_act_vect.append(mec_act)
                lec_act_vect.append(lec_act)
                hpc_act_vect.append(hpc_act)
                
                nnnM1 = np.max([HPCconvergeDisttime[ii,0],HPCconvergetime[ii,0],MECconvergeDisttime[ii,0],MECconvergetime[ii,0]])
                nnnM2 = np.max([HPCconvergeDisttime[ii,1],HPCconvergetime[ii,1],MECconvergeDisttime[ii,1],MECconvergetime[ii,1]])
                nnnZ1 = np.min([HPCconvergeDisttime[ii,0],HPCconvergetime[ii,0],MECconvergeDisttime[ii,0],MECconvergetime[ii,0]])
                nnnZ2 = np.min([HPCconvergeDisttime[ii,1],HPCconvergetime[ii,1],MECconvergeDisttime[ii,1],MECconvergetime[ii,1]])

                print('convergence times (max): %d %d %d %d' % (nnnM1,nnnM2,nnnZ1,nnnZ2)) 
                print('convergence times (0): %d %d %d %d' % (HPCconvergeDisttime[ii,0],HPCconvergetime[ii,0],MECconvergeDisttime[ii,0],MECconvergetime[ii,0])) 
                print('convergence times (1): %d %d %d %d' % (HPCconvergeDisttime[ii,1],HPCconvergetime[ii,1],MECconvergeDisttime[ii,1],MECconvergetime[ii,1])) 
                print('dist %.2f %.2f %.2f %.2f' % (HPCconvergeDist[ii,0,nnnM1+1],MECconvergeDist[ii,0,nnnM1+1],HPCconvergeDist[ii,1,nnnM2+1],MECconvergeDist[ii,1,nnnM2+1]))                
                print('corr %.2f %.2f %.2f %.2f' % (HPCconverge[ii,0,nnnM1+1],MECconverge[ii,0,nnnM1+1],HPCconverge[ii,1,nnnM2+1],MECconverge[ii,1,nnnM2+1]))                
            
                if(nnnM1==1 and nnnM2==0 and nnnZ1>=0 and nnnZ2>=0 and lllf[ii]>0 and netconverged==False):
                    netconverged = True
                    print("converge!!") 
                    convergethetatime = ii - nummorphs
        
        if (actsaveb):
            with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,0)+'z', 'wb') as ff:
                pickle.dump([MECactivity,HPCactivity,LECactivity] , ff)
        with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,1)+'z', 'wb') as ff:
            pickle.dump([MECconverge,HPCconverge,MECconvergeDist,HPCconvergeDist] , ff)
        
        for zzii in arange(len(r_start)):
        
            for xx in arange(nummorphs):
                ooo1a = np.zeros(arena_binsize)
                ooo2a = np.zeros(arena_binsize)
                ooo1b = np.zeros(arena_binsize)
                ooo2b = np.zeros(arena_binsize)
                ooo1c = np.zeros(arena_binsize)
                ooo2c = np.zeros(arena_binsize)
                                  
                for ii in arange(arena_binsize[0]):
                    for jj in arange(arena_binsize[1]):
                        ooo1a[ii,jj] = np.corrcoef(hpc_inact_vect[r_start[zzii],:,ii,jj],hpc_inact_vect[xx+r_start[zzii],:,ii,jj])[0,1]
                        ooo1b[ii,jj] = np.corrcoef(mec_inact_vect[r_start[zzii],:,ii,jj],mec_inact_vect[xx+r_start[zzii],:,ii,jj])[0,1]
                        ooo1c[ii,jj] = np.corrcoef(lec_inact_vect[r_start[zzii],:,ii,jj],lec_inact_vect[xx+r_start[zzii],:,ii,jj])[0,1]
                        ooo2a[ii,jj] = np.corrcoef(hpc_inact_vect[r_end[zzii],:,ii,jj],hpc_inact_vect[r_end[zzii]-xx,:,ii,jj])[0,1]
                        ooo2b[ii,jj] = np.corrcoef(mec_inact_vect[r_end[zzii],:,ii,jj],mec_inact_vect[r_end[zzii]-xx,:,ii,jj])[0,1]
                        ooo2c[ii,jj] = np.corrcoef(lec_inact_vect[r_end[zzii],:,ii,jj],lec_inact_vect[r_end[zzii]-xx,:,ii,jj])[0,1]
                        
                pvCorrelationCurveHPC1[zzii,:,xx] = ooo1a[0,:]
                pvCorrelationCurveMEC1[zzii,:,xx] = ooo1b[0,:]
                pvCorrelationCurveLEC1[zzii,:,xx] = ooo1c[0,:]
                pvCorrelationCurveHPC2[zzii,:,xx] = ooo2a[0,:]
                pvCorrelationCurveMEC2[zzii,:,xx] = ooo2b[0,:]
                pvCorrelationCurveLEC2[zzii,:,xx] = ooo2c[0,:]
                    
        with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,2)+'z', 'wb') as ff:
            pickle.dump([pvCorrelationCurveHPC1,pvCorrelationCurveHPC2,pvCorrelationCurveMEC1,pvCorrelationCurveMEC2,pvCorrelationCurveLEC1,pvCorrelationCurveLEC2] , ff)
        with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,3)+'z', 'wb') as ff:
            pickle.dump([MECconvergetime,HPCconvergetime,MECconvergeDisttime,HPCconvergeDisttime,convergethetatime] , ff)
            
if __name__ == "__main__":
   main(sys.argv[1:])
   
        
    
        
    
    
    
    
    
    













