### import fActivation as fa

import numpy as np

#   Calculate Neuron Activity for an input via Sigmoid Function
def Activation(x):
    activate = 1/(1+np.exp(-x))
    return activate



def Derivative_Activation(x):
#    Activation(x)*(1-Activation(x))        # Node-Output = Activation(x)
    Derivative = x*(1-x)
    return Derivative


