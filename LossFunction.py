

import numpy as np
import NetworkInitialize as ni
import DataSelect as ds
import Gradient as gr

class LossFunction(object):
    """"""
    ### CLASS ATTRIBUTE
    Section = "Output Processing"

    def __init__(self,name,Iteration, Loss,LossSquared,OutputGiven,OutputTarget,LossAvg,LossSqAvg):
        self.name = name
        self.Iteration = Iteration
        self.Loss = Loss
        self.LossSquared = LossSquared
        self.OutputGiven = OutputGiven
        self.OutputTarget = OutputTarget
        self.LossAvg = LossAvg
        self.LossSqAvg = LossSqAvg


LossSet = np.array([],dtype='float64')

for j in range(np.size(ds.InputSelect)):
    LossPart = LossFunction("Loss Values",j,0,0,0,ds.data[j,3],0,1)
    LossSet = np.append(LossSet,LossPart)

def GenLossSet(FormerLossSet):
    LossSet = np.array([])

    for j in range(np.size(ds.InputSelect)):
        LossPart = LossFunction("Loss Values",j,0,0,0,ds.data[j,3],0,1)
        LossSet = np.append(LossSet,LossPart)
    return LossSet



def Loss(LossSetPart):
    gr.GradientCounter = np.array([])
    [ActivationArray,WeightArray,BiasArray] = gr.GradientOperation(LossSetPart.Iteration,0,0)
    gr.GradientCounter = np.array([1])
    LossSetPart.OutputGiven = ActivationArray[0].Value * ni.YWeight + ni.YBias
    LossSetPart.Loss = LossSetPart.OutputGiven - LossSetPart.OutputTarget
    LossSetPart.LossSquared = LossSetPart.Loss**2
    return LossSetPart

def CalculateLossAvg(LossSet):
    LossCount = np.array([])
    LossSqCount = np.array([])
    for j in range(len(LossSet)):
        collectLoss = LossSet[j].Loss
        collectLossSquared = LossSet[j].LossSquared
        LossCount = np.append(LossCount,collectLoss)
        LossSqCount = np.append(LossSqCount,collectLossSquared)
    LossAverage = np.average(LossCount)
    LossSqAverage = np.average(LossSqCount)
    for j in range(len(LossSet)):
        LossSet[j].LossAvg = LossAverage
        LossSet[j].LossSqAvg = LossSqAverage
    return LossSet


