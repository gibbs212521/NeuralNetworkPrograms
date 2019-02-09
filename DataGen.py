### SEE LINE 14
### DataSetSize Requires Alteration
### import DataGen as dg

#class dataGen(object):
#    """Create or Overwrite File For Evaluation."""
   
import numpy as np


##  TRAINING SET 100,000          ### HALF HOUR TO CREAT **8
##  TESTING SET 1,000


DataSetSize = 10**4

C = np.random.randint(10,10** 6 ,DataSetSize)
C = C * 10 ** -12
V0 = 5 # 5 Volt DC Input

R = np.random.randint(10**3,10**8,DataSetSize)

for i in range (0,DataSetSize):
    C[i] = round(C[i],15)
    R[i] = round(R[i],-3)
#print(C[0:10])
#print(R[0:10])

Tester = np.random.uniform(low=0.0005, high=0.3, size = (DataSetSize,))

T = C * R * Tester

#print (T[1000:1010])

#for i in range (int(DataSetSize*0.5),int(DataSetSize*0.5+DataSetSize*0.01)):
#    print (C[i],R[i],T[i], sep=",")
    
RC = R * C
Vt = V0*np.exp(-T/RC)

f = open('ProjectDataSet.csv','w')
np.savetxt("ProjectDataSet.csv", np.column_stack((C, R, T, Vt)), delimiter=",", fmt='%s')
f.close


#for P in range (0,5):
#    print(Vt[P]/np.exp(T[P]/RC[P]))



#f = open('ProjectInputSet.csv','w')
#np.savetxt("ProjectInputSet.csv", np.column_stack((C, R, T)), delimiter=",", fmt='%s')
#f.close

#f = open('ProjectOutputSet.csv','w')
#np.savetxt("ProjectOutputSet.csv", np.row_stack((Vt)), delimiter=",", fmt='%s')
#f.close


#print (C[0])
#print (R[0])
#print (Vt[0])
#print (T[5])



