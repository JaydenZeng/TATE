import pickle
import numpy as np

traindata = pickle.load(open('./train.pkl', 'rb'))
testdata  = pickle.load(open('./test.pkl', 'rb'))

a = [0, 0, 0]
b = [0, 0, 0]


for i in range(len(traindata['L'])):
    if traindata['L'][i] == 0:
        a[0] += 1
    elif traindata['L'][i] == 1:
        a[1] += 1
    else:
        a[2] += 1

for i in range(len(testdata['L'])):
    if testdata['L'][i]==0:
        b[0] += 1
    elif testdata['L'][i]==1:
        b[1] +=1
    else:
        b[2] +=1

print (a)
print (b)
