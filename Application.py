### Data Set Input Requires Alteration
### import ForwardPropagation as fp

   
import numpy as np
import DataSelect as ds
import fActivation as fa
import Inputs as iput

nLayers = iput.nLayers
nOutputs = iput.nOutputs
nInputs = iput.nInputs



f = open('Weight.csv','r')
Weight = np.loadtxt('Weight.csv',delimiter=',')
f.close()

f = open('Bias.csv','r')
Bias = np.loadtxt('Bias.csv',delimiter=',')
f.close()





def ForwardPropagate(i):
    
    NodeInput = np.array([])
    NodeOutput = np.array([])


    InputValue = {
        0 : ds.InputSelect[i].capacitance,
        1 : ds.InputSelect[i].resistance,
        2 : ds.InputSelect[i].time,
        }

    for m in range(nInputs):           # Input - Node Cycle
        NodePut = np.array([])
        for k in range(nInputs):       # Vector Input Cycle
            NodeThruPut = Weight[k + m*nInputs] * InputValue[k]
            NodePut = np.append(NodePut,NodeThruPut)
        NetStep = sum(NodePut) + Bias[m]
        NodeInput = np.append(NodeInput,NetStep)
        NodeOutput = np.append(NodeOutput,fa.Activation(NodeInput))
    

    for j in range(nLayers-1):         # Body Layer Cycle ----- Subtract 1 to account for input layer
        for m in range(nInputs):       # Node Cycle 
            NodePut= np.array([])
            for k in range(nInputs):   # Vector Input Cycle
                NodeThruPut = Weight[k + m*nInputs -1 + (j+1)*(nInputs**2)] * NodeOutput[m + (j)*nInputs]
                NodePut = np.append(NodePut,NodeThruPut)
            NetStep = sum(NodePut) + Bias[m+ (j)*nInputs]
            NodeInput = np.append(NodeInput,NetStep)
            NodeOutput = np.append(NodeOutput,fa.Activation(NodeInput))


    for m in range(nOutputs):          # Output - Node Cycle 
        NodePut= np.array([])
        for k in range(nInputs):       # Vector Input Cycle
            NodeThruPut = Weight[m + k*nOutputs + (nLayers)*(nInputs**2)] * NodeOutput[m + (nLayers)*nInputs]
            NodePut = np.append(NodePut,NodeThruPut)
        NetStep = sum(NodePut) + Bias[m + nLayers*nInputs]
        NodeInput = np.append(NodeInput,NetStep)
        NodeOutput = np.append(NodeOutput,fa.Activation(NodeInput))

    return [NodeOutput,InputValue]



#print(NodeOutput[-1])
#print(NodeOutput[-2])
#print(NodeOutput[-3])
#print(NodeOutput[-0])
#print(NodeOutput)


