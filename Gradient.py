### import Gradient as gr
### SEE LINE 10
### DESIGNED TO WORK FROM i=0 to i=iMAX in Back Propagation

import numpy as np
import NetworkInitialize as ni
import DataSelect as ds
import ForwardPropagation as fp

iteration =0

GradientCounter = np.array([])
        
def Counter(GradientCounter):
    NewCount = 0
    GradientCounter = np.append(GradientCounter,NewCount)
    return GradientCounter        


class Gradient(object):
    """"""

    ## Class Attribute
    Section = 'Processing'

    def __init__(self, name, Value, DistFromOutput, NodeIteration, Origin, SwitchedOn):
        self.Value = Value
        self.name = name
        self.DistFromOutput = DistFromOutput
        self.NodeIteration = NodeIteration
        self.Origin = Origin
        self.SwitchedOn = SwitchedOn




class NodeActivation(object):
    # Class Description
    "''"
    
    ## Class Attribute
    Section = 'Processing'

    #   Does not include Y Bias (Only operates concurrently with Nodes)

    def __init__(self, name, DistFromOutput, NodeIteration, Value, SwitchedOn):
        self.name = name
        self.DistFromOutput = DistFromOutput
        self.NodeIteration = NodeIteration
        self.Value = Value
        self.SwitchedOn = SwitchedOn



def GradientOperation(iteration,dWeight,dBias):

    [NodeOutput, InputValue] = fp.ForwardPropagate(iteration)

    ActivationArray = np.array([])
    WeightArray = np.array([])
    BiasArray = np.array([])

    ActivationPut = np.array([])
    WeightPut = np.array([])
    BiasPut = np.array([])


    if (np.size(GradientCounter) == 0):     ### DEFINE TERMS of NodeActivation & BiasGradient
        for j in range(ni.nOutputs):
            ActivationThruPut = NodeActivation("Node", 0, ni.nOutputs-(1+j), NodeOutput[-(j+1)],1)
            ActivationPut = np.append(ActivationPut,ActivationThruPut)
            BiasThruPut = NodeActivation("Bias Gradient", 0, ni.nOutputs-(1+j), ni.Bias[-(j+1)], 0)
            BiasPut = np.append(BiasPut,BiasThruPut)
            for k in range(ni.nInputs):
                WeightThruPut = Gradient("Weight Gradient", ni.Weight[-(1 + k + j*ni.nOutputs)], 0, ni.nOutputs-(1+j), [ni.nInputs-(1+k),ni.nLayers-1], 0)
                WeightPut = np.append(WeightPut,WeightThruPut)

        for m in range(ni.nLayers):
            for j in range(ni.nInputs):
                ActivationThruPut = NodeActivation("Node",m+1, ni.nInputs-(1+j),NodeOutput[-(1 + j + ni.nOutputs + m*ni.nInputs)],0)
                ActivationPut = np.append(ActivationPut,ActivationThruPut)
                BiasThruPut = NodeActivation("Bias Gradient",m+1,ni.nInputs-(1+j),ni.Bias[-(1 + j + ni.nOutputs + m*ni.nInputs)],0)
                BiasPut = np.append(BiasPut,BiasThruPut)
                for k in range(ni.nInputs):
                    WeightThruPut = Gradient("Weight Gradient", ni.Weight[-( 1 + ni.nInputs*ni.nOutputs + k + j*ni.nInputs + m*ni.nInputs**2)], m+1, ni.nInputs-(1+j), [ni.nInputs-(k+1),ni.nLayers-(2+m)],0)
                    WeightPut = np.append(WeightPut,WeightThruPut)


    else:
        for j in range(ni.nOutputs):
            ActivationThruPut = NodeActivation("Node", 0, ni.nOutputs-(j+1), NodeOutput[-(j+1)],1)
            ActivationPut = np.append(ActivationPut,ActivationThruPut)
            BiasThruPut = NodeActivation("Bias Gradient", 0, ni.nOutputs-(1+j), dBias[j], 1)
            BiasPut = np.append(BiasPut,BiasThruPut)
            for k in range(ni.nInputs):
                WeightThruPut = Gradient("Weight Gradient", dWeight[k + j*ni.nOutputs], 0, ni.nOutputs-(1+j), [ni.nInputs-(1+k),ni.nLayers-1], 1)
                WeightPut = np.append(WeightPut,WeightThruPut)


        for m in range(ni.nLayers):
            for j in range(ni.nInputs):
                ActivationThruPut = NodeActivation("Node", m+1, ni.nInputs-(1+j), NodeOutput[-(1 + j + ni.nOutputs + m*ni.nInputs)], 1)
                ActivationPut = np.append(ActivationPut,ActivationThruPut)
                BiasThruPut = NodeActivation("Bias Gradient",m+1, ni.nInputs-(1+j),dBias[ j +ni.nOutputs + m*ni.nInputs],1)
                BiasPut = np.append(BiasPut,BiasThruPut)
                for k in range(ni.nInputs):
                    WeightThruPut = Gradient("Weight Gradient", dWeight[ ni.nInputs*ni.nOutputs + k + j*ni.nInputs + m*ni.nInputs**2], m+1, ni.nInputs-(j + 1 ), [ni.nInputs-(1+k),ni.nLayers-(2+m)],1)
                    WeightPut = np.append(WeightPut,WeightThruPut)

    ActivationArray = np.append(ActivationArray,ActivationPut)
    WeightArray = np.append(WeightArray,WeightPut)
    BiasArray = np.append(BiasArray,BiasPut)

    return ([ActivationArray,WeightArray,BiasArray])

[ActivationArray,WeightArray,BiasArray] = GradientOperation(iteration,0,0)



#Nodes = np.array([])
#Biases = np.array([])
#Weights = np.array([])
#for j in range(len(ActivationArray)):
#    Nodes = np.append(Nodes,ActivationArray[j].Value)
#    Biases = np.append(Biases,BiasArray[j].Value)
#
#for j in range(len(WeightArray)):
#    Weights = np.append(Weights,WeightArray[j].Value)
#
#
#print('Nodes')
#print(Nodes)
#print('Weights')
#print(Weights)
#print('Biases')
#print(Biases)



GradientCounter = Counter(GradientCounter)