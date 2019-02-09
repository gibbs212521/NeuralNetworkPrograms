import numpy as np
import Inputs as iput

iput.nInputs = 3
iput.nOutputs = 1
iput.nLayers = 2

Iterations = 10

import DataSelect as ds
import Application as ap


f = open('YWeight.csv','r')
YWeight = np.loadtxt('YWeight.csv',delimiter=',')
f.close()

f = open('YBias.csv','r')
YBias= np.loadtxt('YBias.csv',delimiter=',')
f.close()

for j in range(Iterations): 

    DataSetSelection = np.random.randint(0,len(ds.data))

    [NodeOutput,InputValue] = ap.ForwardPropagate(DataSetSelection)
    
    FinalOutput = NodeOutput[-1]*YWeight + YBias

    MeanLossSquared = (FinalOutput - ds.data[DataSetSelection,3])**2

    print("Test {}".format(j+1))
    print("Final Output is {} against the Target Output {}".format(FinalOutput,ds.data[DataSetSelection,3]))
    print("Square Loss Function amounts to {}\n".format(MeanLossSquared))
print("Weights")
print(ap.Weight)
print("Final Output Weight")
print(YWeight)
print("Biases")
print(ap.Bias)
print("Final Output Bias")
print(YBias)
print("Output Node of Final Test")
print(NodeOutput[-1])



