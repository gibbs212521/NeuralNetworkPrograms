### Data Set Input Requires Alteration
### import ForwardPropagation as fp

   
import numpy as np
import NetworkInitialize as ni
import DataSelect as ds
import fActivation as fa

def ForwardPropagate(i):
    
    NodeInput = np.array([])
    NodeOutput = np.array([])


    InputValue = {
        0 : ds.InputSelect[i].capacitance,
        1 : ds.InputSelect[i].resistance,
        2 : ds.InputSelect[i].time,
        }

    for m in range(ni.NodesInit.nInputs):           # Input - Node Cycle
        NodePut = np.array([])
        for k in range(ni.NodesInit.nInputs):       # Vector Input Cycle
            NodeThruPut = ni.Weight[k + m*ni.NodesInit.nInputs] * InputValue[k]
            NodePut = np.append(NodePut,NodeThruPut)
        NetStep = sum(NodePut) + ni.Bias[m]
        NodeInput = np.append(NodeInput,NetStep)
        NodeOutput = np.append(NodeOutput,fa.Activation(NodeInput))
    

    for j in range(ni.NodesInit.nLayers-1):         # Body Layer Cycle ----- Subtract 1 to account for input layer
        for m in range(ni.NodesInit.nInputs):       # Node Cycle 
            NodePut= np.array([])
            for k in range(ni.NodesInit.nInputs):   # Vector Input Cycle
                NodeThruPut = ni.Weight[k + m*ni.NodesInit.nInputs -1 + (j+1)*(ni.NodesInit.nInputs**2)] * NodeOutput[m + (j)*ni.NodesInit.nInputs]
                NodePut = np.append(NodePut,NodeThruPut)
            NetStep = sum(NodePut) + ni.Bias[m+ (j)*ni.NodesInit.nInputs]
            NodeInput = np.append(NodeInput,NetStep)
            NodeOutput = np.append(NodeOutput,fa.Activation(NodeInput))


    for m in range(ni.NodesInit.nOutputs):          # Output - Node Cycle 
        NodePut= np.array([])
        for k in range(ni.NodesInit.nInputs):       # Vector Input Cycle
            NodeThruPut = ni.Weight[m + k*ni.NodesInit.nOutputs + (ni.NodesInit.nLayers)*(ni.NodesInit.nInputs**2)] * NodeOutput[m + (ni.NodesInit.nLayers)*ni.NodesInit.nInputs]
            NodePut = np.append(NodePut,NodeThruPut)
        NetStep = sum(NodePut) + ni.Bias[m + ni.NodesInit.nLayers*ni.NodesInit.nInputs]
        NodeInput = np.append(NodeInput,NetStep)
        NodeOutput = np.append(NodeOutput,fa.Activation(NodeInput))

    return [NodeOutput,InputValue]



#print(NodeOutput[-1])
#print(NodeOutput[-2])
#print(NodeOutput[-3])
#print(NodeOutput[-0])
#print(NodeOutput)

