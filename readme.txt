This is the readme for the model code associated with the paper:

Rennó-Costa C, Tort ABL (2017) Place and Grid Cells in a Loop:
Implications for Memory Function and Spatial Coding.
J Neurosci 37:8062-8076

This code was contributed by C Rennó-Costa.

Instructions to run:

To run the experiments, please edit the file "support_filename.py" to
include the output directory of the simulation results. Each of the
simulations will produce a pickle file with the results necessary to
produce the statistics and figures in the manuscript. We provide four
scripts that implement different versions of the model. To run you
must call a python interpreter, the name of the script and the
parameters.

a) loop_script_convergence.py will train the network with (1) one
memory pattern or (2) memory patterns and simulate the output of the
simulation considering a morph between the patterns. One can take
information about the convergence of the network. The parameters are:

<random seed for the input pattern: int>
<random seed for the initial weights: int>
<random seed for the path: int>
<number of gamma cycles for each theta cycle: int>
<number of times the agent explores the environment: int>
<number of training sessions before recorded training: int>
<number of recorded training sessions: int>
<learning rate for hippocampus to mEC: int>
<learning rate from mEC to hippocampus: int>
<learning rate from lEC to hippocampus: int>
<ratio of the input to mEC (hippocampus vs recurrent): int x100>
<ratio of the input to HPC (lEC vs mEC): int x100>
<strength of hippocampus pattern completion: int x100>
<difference between learned patterns: int>
<-c: without will teach 1 memory, with will teach 2 memories>
<-a: also save the activity data (a lot of data)>
<-k: will overwrite previous simulation>
<-s, example of different places where the simulation will save the output, see support_filename.py >

Example:

> python3  loop_script_convergence.py   10 11 12   7 1  5 50  100 10 10    50 10 90   50

this simulation will make an input pattern with random seed '10', the
initial weights with random seed '11' and the path (dummy number, no
path here) with random seed ('12'). each theta cycle will have 7 gamma cycles 
and will run the pattern only one time. will do 50 different
learning sessions (with five pre-learning in each). Learning rate will
be 100 from the hippocampus to grid cells and 10 from the EC to the
hippocampus. The grid cells will have 50% inputs from place cells and
another half from recurrent collaterals. The input to the hippocampus
will have 10% from grid cells and 90% from lEC. Patterns that match
90% of saved patterns in the hippocampus will be completed. The two
memory patterns will have 0.5 of correlation.

b) loop_script_morph.py will train the network with two environments
and simulate the output of the simulation considering a morph between
the environments. The main results of the paper are taken from
this. The parameters are:

<random seed for the input pattern: int>
<random seed for the initial weights: int>
<random seed for the path: int>
<number of gamma cycles for each theta cycle: int>
<number of times the agent explores the environment: int>
<number of training sessions before recorded training: int>
<number of recorded training sessions: int>
<learning rate for hippocampus to mEC: int>
<learning rate from mEC to hippocampus: int>
<learning rate from lEC to hippocampus: int>
<ratio of the input to mEC (hippocampus vs recurrent): int x100>
<ratio of the input to HPC (lEC vs mEC): int x100>
<strength of hippocampus pattern completion: int x100>
<difference between learned patterns: int>
<-a: also save the activity data (a lot of data)>
<-k: will overwrite previous simulation>
<-s, example of different locations where the simulation will save the output, see support_filename.py >

#example:
> python3  loop_script_morph.py   10 11 12   7 1  1  100 10 10    50 10 90   50

c) loop_script_morph_noise.py.. same as before, but with the variable 
levels of noise 

#example:
> python3  loop_script_morph_noise.py   10 11 12   7 1  1  100 10 10    50 10 90   50 

d) loop_script_morph_consistency.py.. same as before, but with variable 
level of consistency

#example:
> python3  loop_script_morph_consistency.py   10 11 12   7 1  1  100 10 10    50 10 90   50 

e) loop_script_morph_develop.py.. same as before, but with the level
of noise in the emulation of the development of grid cells. simulate
with and without grid cells.

#example:
> python3  loop_script_morph_develop.py   10 11 12   7 1  1  100 10 10    50 10 90   50
