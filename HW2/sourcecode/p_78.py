import numpy as np
import matplotlib.pyplot as plt

def cal_Ein(X,Y):
    best_s = 1
    best_theta = 0
    best_err = datasize
    x1 = np.sort(X)
    for i in range (datasize+1):
        if i == 0:
            theta = 0.5 * (x1[i]-1)
        elif i == datasize:
            theta = 0.5 * (x1[i-1]+1)
        else:
            theta = 0.5 * (x1[i] + x1[i-1])
             
        y1 = np.sign(X-theta)
        y2 = (-1) * np.sign(X-theta)
        
        error1 = np.sum(y1!=Y)
        error2 = np.sum(y2!=Y)
        #print(error1, error2)
        if error1 < best_err:
            best_err = error1
            best_theta = theta
            best_s = 1
        if error2 < best_err:
            best_err = error2
            best_theta = theta
            best_s = -1
            
    return best_err/datasize, best_s, best_theta
                
def cal_Eout(s,t):
    return 0.5 + 0.3 * s * (abs(t)-1)


def algorithm(datasize, times):
    diff_Ein_out = []
    for i in range (times):
        X = np.zeros(datasize)
        Y = np.zeros(datasize)
        X = np.random.uniform(-1, 1, datasize)
        for j in range(datasize):
            if np.random.uniform(0,1) < 0.2:
                Y[j] = -1 * np.sign(X[j])
            else:
                Y[j] = np.sign(X[j])

        Ein , s, t = cal_Ein(X,Y)
        Eout =  cal_Eout(s,t)
        diff_Ein_out.append(Ein-Eout)

    return diff_Ein_out

# P_7 
datasize = 20
times = 1000
output = algorithm(datasize, times)
dis = np.arange(-0.5, 0.2, 0.02)
plt.title("Datasize = 20, Times = 1000")
plt.xlabel("E_out - E_in")
plt.ylabel("frequency")
plt.hist(output, dis)
plt.savefig("p_7.png")
print("[ p_7 average of E_in-E_out ]: "+str(np.mean(output)))

#P_8
datasize = 2000
times = 1000
output = algorithm(datasize, times)
dis = np.arange(-0.04, 0.04, 0.002)
plt.close();
plt.title("Datasize = 2000, Times = 1000")
plt.xlabel("E_out - E_in")
plt.ylabel("frequency")
plt.hist(output, dis)
plt.savefig("p_8.png")
print("[ p_8 average of E_in-E_out ]: "+str(np.mean(output)))
