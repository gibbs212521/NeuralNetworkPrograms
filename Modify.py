


import numpy as np
import NetworkInitialize as ni
import DataSelect as ds
import ForwardPropagation as fp
import Gradient as gr
import BackPropagation as bp
import LossFunction as ls


def StochasticGradient(k):

    avgDWeight = np.empty(np.size(ni.Weight))
    avgDBias = np.empty(np.size(ni.Bias))
    avgDYWeight = np.empty(np.size(ni.YWeight))
    avgDYBias = np.empty(np.size(ni.YBias))

    GroupLossSet = np.array([])

    for i in range(k):

        v =  np.random.randint(0,len(ls.LossSet))

        LearnRate = 0.95
        ls.LossSet[v] = ls.Loss(ls.LossSet[v])
        GroupLossSet = np.append(GroupLossSet,ls.LossSet[v])


        gr.GradientCounter = np.array([])

        [ActivationArray,WeightArray,BiasArray] = gr.GradientOperation(v,0,0)
        gr.GradientCounter = np.array([1])

        [dWeight, dBias, dYWeight, dYBias] = bp.BackPropagate(v)

        [ActivationArray,dWeightArray,dBiasArray] = gr.GradientOperation(v,dWeight,dBias)



        Nodes = np.array([])
        Weights = np.array([])
        Biases = np.array([])
        dWeights = np.array([])
        dBiases = np.array([])

        for j in range(len(ActivationArray)):
            Nodes = np.append(Nodes,ActivationArray[-(j+1)].Value)
            Biases = np.append(Biases,BiasArray[-(j+1)].Value)
            dBiases = np.append(dBiases,dBiasArray[-(j+1)].Value)
            avgDBias[j] += dBiasArray[-(j+1)].Value/(np.size(dBias))


        for j in range(len(WeightArray)):
            Weights = np.append(Weights,WeightArray[-(j+1)].Value)
            dWeights = np.append(dWeights,dWeightArray[-(j+1)].Value)
            avgDWeight += dWeightArray[-(j+1)].Value/(np.size(dWeight))

        for j in range(np.size(dYWeight)):
            avgDYWeight += dYWeight[-(j+1)]/(np.size(dYWeight))
            avgDYBias += dYBias[-(j+1)]/(np.size(dYBias))

    GroupLossSet = ls.CalculateLossAvg(GroupLossSet)
    GroupLoss = GroupLossSet[0].LossAvg
    GroupLossSq = GroupLossSet[0].LossSqAvg



    return [avgDWeight, avgDBias, avgDYWeight, avgDYBias, GroupLoss, GroupLossSq]
            
        #print('Nodes')
        #print(len(Nodes))
        #print(Nodes)
        #print('Weights')
        #print(len(Weights))
        #print(Weights)
        #print('Biases')
        #print(len(Biases))
        #print(Biases)
        #print('dWeights')
        #print(len(dWeights))
        #print(dWeights)
        #print('dBiases')
        #print(len(dBiases))
        #print(dBiases)
        #print('dYWeight')
        #print(len(dYWeight))
        #print(dYWeight)
        #print('dYBias')
        #print(len(dYBias))
        #print(dYBias)

#print('dWeight')
#print(sumationDWeight)
#print('dBias')
#print(sumationDBias)
#print('dYWeight')
#print(sumationDYWeight)
#print('dYBias')
#print(sumationDYBias)

ModLossSet = np.array([])



