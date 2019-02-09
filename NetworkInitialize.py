### SEE LINEs 30 THRU 33                        ### SEE BOTTOM OF CODE
### nLayers, nOutputs, & nInputs require Alteration
### import NetworkInitialize as ni

class Network:
    """Initialization of Network Skeleton"""

    # Genus Operandi
    Section = 'Initialization'

    # Initializer / Instance Attributes
    def __init__(self, name, nLayers, nOutputs, nInputs, toRandom):
        self.name = name
        self.nLayers = nLayers
        self.nOutputs = nOutputs
        self.nInputs = nInputs
        self.toRandom = toRandom
        
# name = locality
# nLayers = number of Layers
# nOutputs = number of Outputs for the Network, Must Be 0 if not applied to Outputs
# nInputs = number of Inputs for the Network
# toRandom = 1 if values are to be randomized and 0 if values are to be zeroes


    

import numpy as np
import Inputs as iput

nLayers = iput.nLayers
nOutputs = iput.nOutputs
nInputs = iput.nInputs


NodesInit = Network("Nodes", nLayers, nOutputs, nInputs, 0)
WeightsInit = Network("Weights", nLayers, nOutputs, nInputs, 1)
BiasesInit = Network("Biases", nLayers, nOutputs, nInputs, 1)



#   Initialize Network
def Initialize_Nodes(Network):
    Nodes = np.array([])
    if Network.name == "Weights" :
        Hidden_Nodes = [0.0 for i in range(Network.nInputs**2) for i in range (Network.nLayers)] # +1 for output layer's inputs
        Output_Nodes = [0.0 for i in range(Network.nInputs*Network.nOutputs)]
    else:
        Hidden_Nodes = [0.0 for i in range(Network.nInputs) for i in range (Network.nLayers )]
        Output_Nodes = [0.0 for i in range(Network.nOutputs)]
#    print('test0')
#    print(Nodes)
    if Network.toRandom == 0:
        Nodes = np.append(Nodes,np.zeros(len(Hidden_Nodes)))
        Nodes = np.append(Nodes,np.zeros(len(Output_Nodes)))
    else:
        Nodes = np.append(Nodes,np.random.rand(len(Hidden_Nodes)))
        Nodes = np.append(Nodes,np.random.rand(len(Output_Nodes)))

#    print("Initial {}".format(Network.name))
#    print(Nodes)

    return Nodes

Bias = Initialize_Nodes(BiasesInit)
Node = Initialize_Nodes(NodesInit)
Weight = Initialize_Nodes(WeightsInit)



YWeight = np.random.rand(nOutputs)
YBias = np.random.rand(nOutputs)


####                                                                #####
#                                                                       #
#   CONSIDER REWORKING THIS SEGAMENT IN ORDER TO PRINT TO .CSV FILE     #
#                                                                       #
####                                                                #####

