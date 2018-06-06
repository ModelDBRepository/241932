# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 12:55:44 2015

@author: rennocosta
"""

import os
import errno

class remappingFileNames:

#cluster

    def __init__(self,simMode = r"default"):
                
        if (simMode == "default"):
            self.pathRoot = r"/home/user/"
            self.pathData = r"DATA/" 
            self.pathRemapping = r"Remapping/"
            self.pathSimulation = r"Simulation_{0:02d}/"
            #self.pathRun = r"Run_{0:02d}/"
            self.bar = r""
            self.bar2 = r"/"
            
        if (simMode == "cluster"):
            self.pathRoot = r"/home/cluster/"
            self.pathData = r"DATA/" 
            self.pathRemapping = r"Remapping/"
            self.pathSimulation = r"Simulation_{0:02d}/"
            #self.pathRun = r"Run_{0:02d}/"
            self.bar = r""
            self.bar2 = r"/"
			
       
        self.filenameRun = r"run"
        self.filenameParam = r"param"
            
        self.extensionPickle = r".pkl"
        self.extensionMat = r".mat"


    def prepareSimulation(self,listofvalues,simulation=0):
        try:
            os.makedirs(self.pathRoot)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
        try:
            os.makedirs(self.pathRoot + self.pathData)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
        try:
            os.makedirs(self.pathRoot+ self.pathData+self.pathSimulation.format(simulation+1))
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
        #try:
        #    os.makedirs(self.pathRoot+ self.pathData+self.pathSimulation.format(simulation+1)+ self.pathRun.format(run+1))
        #except OSError as exception:
        #    if exception.errno != errno.EEXIST:
        #        raise

        try:
            nanana = self.pathRoot + self.pathData + self.pathSimulation.format(simulation+1) + self.bar + self.filenameParam
            for ii in listofvalues:
                nanana = nanana + (r"_{0:04d}").format(ii)
            nanana = nanana + self.bar2  
            os.makedirs(nanana)
        except OSError as exception:
            print("whats now?")
            if exception.errno != errno.EEXIST:
                raise        


    def fileRunPickle(self,listofvalues,simulation=0,run=0):
        nanana = self.pathRoot + self.pathData + self.pathSimulation.format(simulation+1) + self.bar + self.filenameParam
        for ii in listofvalues:
            nanana = nanana + (r"_{0:04d}").format(ii)
        return nanana + self.bar2 + self.bar + self.filenameRun + (r"_{0:04d}").format(run) + self.extensionPickle
            
      
      #  nanana =  self.pathRoot + self.pathData + self.pathSimulation.format(simulation+1) + self.pathRun.format(run+1) + self.bar + self.filenameRun
      #  for ii in listofvalues:
      #      nanana = nanana + (r"_{0:04d}").format(ii)
      #  return nanana + self.extensionPickle
            
