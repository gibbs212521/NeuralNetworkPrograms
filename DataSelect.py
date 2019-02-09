### import DataSelect as ds

class Select(object):
    """Select Data Set for Activation Function """

    # Class Attribute
    Section = 'Processing'

    # Initializer / Instance Attributes
    def __init__(self, iteration, capacitance, resistance, time, outputCorrect, outputError):
        self.iteration = iteration
        self.capacitance = capacitance
        self.resistance = resistance
        self.time = time
        self.outputCorrect = outputCorrect
        self.outputError = outputError

import numpy as np
import fActivation as fa



f = open('ProjectDataSet.csv','r')
data = np.loadtxt('ProjectDataSet.csv',delimiter=',')
f.close()

#####################################################################
#                                                                   #
###############           NORMALIZED DATA             ###############
#                                                                   #
#####################################################################


C=data[:,0]
R=data[:,1]
T=data[:,2]
V=data[:,3]

#print(data[1:10,:])

# V0 = 5 Volts DC Input
meanC = np.average(C)
meanR = np.average(R)
meanT = np.average(T)

stdC = np.std(C)
stdR = np.std(R)
stdT = np.std(T)


data[:,0] = ((data[:,0]-meanC)/stdC)
data[:,1] = ((data[:,1]-meanR)/stdR)
data[:,2] = ((data[:,2]-meanT)/stdT)
data[:,3] = data[:,3]



#########################################################################################################################################################################

InputSelect = np.array([])
#TESTERSEL = np.array([])

for i in range(len(data)):
    SetInput = Select(i,data[i,0],data[i,1],data[i,2],data[i,3],0)
    InputSelect = np.append(InputSelect,SetInput)
    #TESTERSEL = np.append(TESTERSEL,SetInput.capacitance)


#print(TESTERSEL[1:10])
#for i in range(20):
#    print("Data Set {} has C = {}, R = {}, & T = {} with the expected output of {}".format(
#        InputSelect[i].iteration,InputSelect[i].cap acitance,InputSelect[i].resistance,InputSelect[i].time,InputSelect[i].outputCorrect))




