
# Contains Gradient Storage

import numpy as np
import NetworkInitialize as ni
import ForwardPropagation as fp
import Gradient as gr
import ForwardPropagation as fp
import BackPropagation as bp
import LossFunction as ls
import Modify as md

StochasticReductions = 1
nIterations =  300
StochasticSweep = 500
LearningRate = 1 * 10**(-4)
BiasCompensation = 0.5


Collector0 = np.array([])
Collector1 = np.array([])
Collector = np.array([])

#f = open('ErrorSqPlot.csv','w')
#np.savetxt("ErrorSqPlot.csv", np.array([]), delimiter=",", fmt='%s')
#f.close

for h in range(StochasticReductions):
    if h == 0:
       Collector1 = 0 
    else:
#        LossSqCollection = np.append(LossSqCollection,GroupLossSet[0].LossSqAvg)
#        print('Iteration {}'.format(h*nIterations))
#        print('loss')
#        print(GroupLossSet[0].LossAvg)
#        print(GroupLossSet[0].LossSqAvg)
#        print('min')
#        print(min(LossSqCollection))
        if h > 3:
            if (Collector1==min(Collector[:,1])) :
                print("New Minimum with Difference of :  {}".format(Collector1-(np.min(Collector[:h*nIterations-1,1]))))
                ############################## MINIMUM SET #######################################
                
                f = open('MinErrorPlot.csv','w')
                np.savetxt("MinErrorPlot.csv", np.column_stack((Collector[:,0],Collector[:,1])), delimiter=",", fmt='%s')
                f.close
    
                f = open('MinWeight.csv','w')
                np.savetxt('MinWeight.csv', np.column_stack((ni.Weight)), delimiter=",", fmt='%s')
                f.close

                f = open('MinBias.csv','w')
                np.savetxt("MinBias.csv", np.column_stack((ni.Bias)), delimiter=",", fmt='%s')
                f.close

                f = open('MinActivation.csv','w')
                np.savetxt("MinActivation.csv", np.column_stack((ni.Node)), delimiter=",", fmt='%s')
                f.close

                f = open('MinYWeight.csv','w')
                np.savetxt('MinYWeight.csv', np.column_stack((ni.YWeight)), delimiter=",", fmt='%s')
                f.close

                f = open('MinYBias.csv','w')
                np.savetxt('MinYBias.csv', np.column_stack((ni.YBias)), delimiter=",", fmt='%s')
                f.close

    for g in range(nIterations):          
        [avgDWeight, avgDBias, avgDYWeight, avgDYBias, GroupLoss, GroupLossSq] = md.StochasticGradient(StochasticSweep)
        print(1 + g + h*nIterations)
#        Collector = np.append(Collector,np.concatenate([GroupLossSet[0].LossAvg],[GroupLossSet[0].LossSqAvg]))
        Collector0 = GroupLoss
        Collector1 = GroupLossSq
        Collector2 = np.array([[Collector0,Collector1]])
        if g == 0:
            Collector = Collector2
        else:
            Collector = np.vstack((Collector,Collector2))
#       print(Collector)
            

#        print('Iteration {}'.format(g+1+h*nIterations))
    #    print('unapplied')
    #    print(ni.Weight)
    #    print(ni.YWeight)
        if Collector1 < 0.2:
            LearningRate = 5 * 10 **(-5)
        if Collector1 < 0.135:
            LearningRate = 5 * 10 **(-7)
        if Collector1 < 0.12:
            LearningRate = 8 * 10 **(-8)
        if Collector1 < 0.11:
            LearningRate = 1 * 10 ** (-8)
        ni.Bias = ni.Bias - avgDBias * GroupLoss*LearningRate * BiasCompensation
        ni.YBias = ni.YBias - avgDYBias * GroupLoss*LearningRate * BiasCompensation
        ni.YWeight = ni.YWeight - avgDYWeight * GroupLoss*LearningRate
        ni.Weight = ni.Weight - avgDWeight * GroupLoss*LearningRate
#   print('applied')
#   print(ni.Weight)
#   print(ni.YWeight)

    
#Collector = np.hsplit(Collector,1)
#Collector = np.concatenate((Collector0,Collector1))

print('Weight')
print(ni.Weight)
print('Bias')
print(ni.Bias)
print('Output Weight')
print(ni.YWeight)
print('Output Bias')
print(ni.YBias)
print('Collector')
print(Collector)

#Collector = np.hsplit(Collector,1)
#Collector = np.delete(Collector,Collector[0,1])
#Collector = np.delete(Collector,Collector[-1,0])



f = open('ErrorPlot.csv','w')
np.savetxt("ErrorPlot.csv", np.column_stack((Collector[:,0],Collector[:,1])), delimiter=",", fmt='%s')
f.close
    
f = open('Weight.csv','w')
np.savetxt('Weight.csv', np.column_stack((ni.Weight)), delimiter=",", fmt='%s')
f.close

f = open('Bias.csv','w')
np.savetxt("Bias.csv", np.column_stack((ni.Bias)), delimiter=",", fmt='%s')
f.close

f = open('Activation.csv','w')
np.savetxt("Activation.csv", np.column_stack((ni.Node)), delimiter=",", fmt='%s')
f.close



f = open('YWeight.csv','w')
np.savetxt('YWeight.csv', np.column_stack((ni.YWeight)), delimiter=",", fmt='%s')
f.close

f = open('YBias.csv','w')
np.savetxt('YBias.csv', np.column_stack((ni.YBias)), delimiter=",", fmt='%s')
f.close