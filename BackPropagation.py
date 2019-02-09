### SEE LINE 5
### Data Set Input Requires Alteration
### import BackPropagation as bp


#############################################################################################################
#                                                                                                           #
#                                                                                                           #
#                                   NOT CALIBRATED FOR MORE THAN 1 OUTPUT                                   #
#                                                                                                           #
#                                                                                                           #
#############################################################################################################

import numpy as np
import NetworkInitialize as ni
import DataSelect as ds
import fActivation as fa
import ForwardPropagation as fp
import Gradient as gr

    #####################

def BackPropagate(i):

    [NodeOutput, InputValue] = fp.ForwardPropagate(i)


    dBiasSum = np.array([])

    dWeight = np.array([])
    dBias = np.array([])

    dYWeight = np.array([])
    dYBias = np.array([])




    for m in range(ni.nOutputs):              #   Y Output - Node Cycle
        YBiasGradient = 1
        dYBias = np.append(dYBias,YBiasGradient)
        YWeightGradient = gr.ActivationArray[m].Value
        dYWeight = np.append(dYWeight,YWeightGradient)

        dBiasSum = np.append(dBiasSum,YBiasGradient*ni.YWeight[-m])



    #########               IN ORDER TO CALIBRATE FOR INPUTS, TAB ALL THE BELOW


    for m in range(ni.nOutputs):                        #   Output Layer - Node Cycle
        BiasGradient = (-1) * fa.Derivative_Activation(gr.ActivationArray[m].Value) * dBiasSum[0]
        dBias = np.append(dBias,BiasGradient)
        for k in range(ni.nInputs):                    #   Output Layer - Vector Cycle
            WeightGradient = gr.ActivationArray[1+k].Value * BiasGradient
            dWeight = np.append(dWeight,WeightGradient)
        
            dBiasSum = np.append(dBiasSum,BiasGradient*gr.WeightArray[k+m*ni.nInputs].Value)
         

        ###################################################################################################################################   BELOW
    dBiasSum0 = np.array([])

    for j in range(ni.nLayers-1):                       #   Layer Cycle     ( MINUS TWO ON ACCOUNT OF THE OUTPUT & INPUT LAYERS)
        if (j!=0):
            dBiasSum = np.append(dBiasSum,dBiasSum1)
        else:
            dBiasSum = dBiasSum
        dBiasSum0 = np.array([])
        dBiasSum1 = np.empty(ni.nInputs)
        for m in range(ni.nInputs):                     #   Node Cycle
            BiasGradient = (-1) * fa.Derivative_Activation(gr.ActivationArray[ni.nOutputs + m + j*ni.nInputs].Value) * dBiasSum[ni.nOutputs + m + j*ni.nInputs]
            dBias = np.append(dBias,BiasGradient)
            for k in range(ni.nInputs):
                WeightGradient = gr.ActivationArray[ni.nOutputs + k + j*ni.nInputs].Value * BiasGradient
                dWeight = np.append(dWeight,WeightGradient)

                dBiasSum0 = np.append(dBiasSum0,BiasGradient * gr.WeightArray[ni.nInputs*ni.nOutputs + k + m*ni.nInputs + j*ni.nInputs**2].Value)
                dBiasSum1[k] += dBiasSum0[k + m*ni.nInputs]
    #    dBiasSum = np.append(dBiasSum,dBiasSum1)

    if (np.size(dBiasSum0)>0):
        dBiasSum = np.append(dBiasSum,dBiasSum1)
    else:
        dBiasSum0=dBiasSum0


    for m in range(ni.nInputs):           #   Input Layer - Node Cycle
        BiasGradient = (-1) * fa.Derivative_Activation(gr.ActivationArray[ni.nOutputs + ni.nInputs*(ni.nLayers-1) + m].Value) *dBiasSum[1 + ni.nInputs*(ni.nLayers-1) + m]
        dBias = np.append(dBias,BiasGradient)
        for k in range(ni.nInputs):       #   Input Layer - Vector Cycle
            WeightGradient = InputValue[2-k] * BiasGradient
            dWeight = np.append(dWeight, WeightGradient)

    return [dWeight, dBias, dYWeight, dYBias]
        
        