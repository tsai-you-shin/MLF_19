import numpy as np
from random import randrange 
import matplotlib.pyplot as plt

#read the input training data
train_data = np.genfromtxt('hw1_6_train.dat')
data_r,data_c = train_data.shape
ptr = np.arange(data_r) 

def sign(Y):
    if Y <= 0:
        return -1
    else:
        return 1
    
def pla():
    trans_cnt = 0
    W = np.zeros(data_c)
    np.random.shuffle(ptr)
    while True:
        is_complete = True
        test=0
        for i in list(ptr):
            X = train_data[i][:-1]
            X = np.insert(X,0,1)
            Y = np.dot(X,W)
            if sign(Y) != train_data[i][-1]:
                is_complete = False
                W = W + train_data[i][-1]*np.array(X)
                trans_cnt+=1
        if is_complete == True:
            break
    return trans_cnt

if __name__ == '__main__':

	#Do PLA 1126 times
	num_of_update=[]
	for j in range (1126):
	    num_of_update.append(pla())

	#draw the histogram and calculate the average
	max_cnt = max(num_of_update)
	min_cnt = min(num_of_update)
	plt.title("Average #updates: "+str(round(np.mean(num_of_update),3)))
	plt.xlabel("# updates")
	plt.ylabel("frequency of #updates")
	plt.hist(num_of_update,range(min_cnt,max_cnt,5))
	plt.savefig('p_6.png')