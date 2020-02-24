import matplotlib.pyplot as plt
import numpy as np


#training and testing data
train_data = np.genfromtxt('hw3_train.dat')
test_data = np.genfromtxt('hw3_test.dat')
train_r,train_c = train_data.shape
test_r,test_c = test_data.shape

x_train = np.concatenate((np.ones((train_r, 1)), train_data[:, : -1]), axis = 1)
y_train = train_data[:,-1]
y_train = np.expand_dims(y_train, axis=1)

x_test = np.concatenate((np.ones((test_r, 1)), test_data[:, : -1]), axis = 1)
y_test = test_data[:,-1]
y_test = np.expand_dims(y_test, axis=1)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def gradient(w, x, y):
    gsum = (-y*x).T.dot(sigmoid(-y *np.dot(x,w)))
    return gsum/len(y)
    
def gradient_one(w, x, y, t):
    t = t%(len(y))
    y =  np.expand_dims(y[t], axis=0)
    x = np.expand_dims(x[t], axis=0)
    gsum = (-y*x).T.dot(sigmoid(-y *np.dot(x,w)))
    return gsum

## p19 fixed rate descent
ita = 0.01
times = 2000
FGD_in_100 = []
FGD_out_100 = []
w = np.zeros((x_train[0].size,1))

for t in range(times-1):
    w = w- ita* gradient(w, x_train, y_train)
    
    y_hat = np.dot(x_train,w)
    errorNum = np.sum(np.sign(y_hat)!=np.sign(y_train))
    FGD_in_100.append(errorNum/len(y_train))

    y_hat = np.dot(x_test,w)
    errorNum = np.sum(np.sign(y_hat)!=np.sign(y_test))
    FGD_out_100.append(errorNum/len(y_test))

## p20 stochastic gradient descent
ita = 0.001
times = 2000
SGD_in_1000 = []
SGD_out_1000 = []
w = np.zeros((x_train[0].size,1))

for t in range(times-1):
    w = w- ita* gradient_one(w, x_train, y_train,t)
    
    y_hat = np.dot(x_test,w)
    errorNum = np.sum(np.sign(y_hat)!=np.sign(y_test))
    SGD_out_1000.append(errorNum/len(y_test))

    y_hat = np.dot(x_train,w)
    errorNum = np.sum(np.sign(y_hat)!=np.sign(y_train))
    SGD_in_1000.append(errorNum/len(y_train))
    
#p7 E_in    
fig, ax = plt.subplots(figsize=(8,6))
x = np.arange(1,2000)
ax.plot(x,FGD_in_100,label="FGD_in_100")
ax.plot(x,SGD_in_1000, label = "SGD_in_1000")
ax.set_xlabel(r't')
ax.set_ylabel(r'$E_{in}({\bf w}_t)$')
ax.legend(loc='lower left')
plt.savefig("Ein.png")
 
#p8 E_out  
fig, ax = plt.subplots(figsize=(8,6))
x = np.arange(1,2000)
ax.plot(x,FGD_out_100,label="FGD_out_100")
ax.plot(x,SGD_out_1000, label="SGD_out_1000")
ax.set_xlabel(r't')
ax.set_ylabel(r'$E_{out}({\bf w}_t)$')
ax.legend(loc='lower left')
plt.savefig("Eout.png")

