import numpy as np
from random import randrange 
import matplotlib.pyplot as plt

#read the input training and testing data
train_data = np.genfromtxt('hw1_7_train.dat')
test_data = np.genfromtxt('hw1_7_test.dat')
data_r,data_c = train_data.shape # the size of training data
test_r,test_c = test_data.shape
train_data = np.insert(train_data,0,1,axis=1)
test_data = np.insert(test_data,0,1,axis=1)
ptr = np.arange(data_r) #random order of input data

def sign(Y):
    if Y <= 0:
        return -1
    else:
        return 1

def error_check(W,string):
    if string == 'train':
        data = train_data
    elif string == 'test':
        data = test_data
    else:
        print('need a string to define process: train/test')
    cnt_error = 0
    for i in range(data_r):
        X = data[i][:-1]
        Y = np.dot(X,W)
        if sign(Y) != data[i][-1]:
            cnt_error+=1
    return cnt_error


def pla_pocket():
    trans_cnt = 0
    error_curr = data_r
    W = np.zeros(data_c)
    np.random.shuffle(ptr)
    while trans_cnt < 100:
        test=0
        for i in list(ptr):
            X = train_data[i][:-1]
            Y = np.dot(X,W)
            if sign(Y) != train_data[i][-1]:
                W = W + train_data[i][-1]*np.array(X)
                trans_cnt+=1
                tmp = error_check(W,'train')
                if tmp < error_curr:
                    W_hat = W
                    error_curr = tmp
                if trans_cnt >= 100:
                    break
    return W_hat

if __name__ == '__main__':
	#Do PLA + pocket 1126 times
	W_hats=[]
	rate_of_error=[]
	#training part
	for i in range (1126):
	    W_hats.append(pla_pocket())
	#testing part
	for i in range(1126):
	    rate_of_error.append(error_check(W_hats[i],'test'))
	rate_of_error= np.array(rate_of_error)/data_r

	#draw the histogram and calculate the average of error rate
	max_rate = max(rate_of_error)
	min_rate = min(rate_of_error)
	plt.title("Average error rate: "+str(round(np.mean(rate_of_error),3)))
	plt.xlabel("error rate")
	plt.ylabel("frequency of error rate")
	plt.hist(rate_of_error)
	plt.savefig('p_7.png')