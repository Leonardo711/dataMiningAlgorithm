# coding: UTF-8
from __future__ import division
import numpy as np
import scipy as sp
'''

weak classifier is based on decision Stump

'''

def sign(x):
    res = np.zeros(x.shape)
    res[x>=0] = 1
    res[x<0] = -1
    return res


class ADA(object):
    def __init__(self, X, y, Classfier):
        # X is N*M matric, N is for features, M is for cases;
        self.X = X
        self.y = y
        self.Classfier = Classfier
        self.tmp = np.zeros(np.y.shape)
        self.W = np.ones((self.X.shape[1], 1).flatten(1))/self.X.shape[1]

    def train(self, T=4):
        #default training time: 4
        self.ClassfierSet = []
        self.alpha = []
        for i in range(T):
            self.ClassfierSet[i] = self.Classfier(self.X, self,y)
            err = self.ClassfierSet[i].train(self.W)
            self.alpha[i] = 1/2 * np.log((1-err)/err)
            pred = self.ClassfierSet[i].predict(self.X)
            nW = self.W * np.exp(-self.alpha[i]*self.y*pred.transpose())
            nw = nw/nw.sum().flatten(1)
            if self.finalclassifer(i) ==0:
                print i+1, "times done"
                return 
        def finalClassfier(t):
            self.tmp += self.ClassfierSet[t].predict(self.X).flatten(1)*self.alpha[t]
            pre_y = sign(self.sums)
            count = (pre_y != self.y).sum()
            return count

    def pred(self, testSet):
        testSet = np.array(testSet)
        assert testSet.shape[0] == self.X.shape[0]
        result = np.zeros((test_set.shape[1],1)).flatten(1)
        
        for i in range(len(self.alpha)):
            result = result + self.ClassfierSet[i].predict(test_set).flatten(1)* self.alpha[i]
        pre_y = sign(result)
        return pre_y

class Classfier(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.N = X.shape[0]

    def train(self, W, steps=100):
        def findmin(i, direction, steps):
            minErr = np.inf
            maxRange, minRange = self.X[i,:].max(), self.X[i,:].min()
            st = (maxRange - minRange)/steps 
            for threshold in range(minRange, maxRange, st):
                gt = np.ones((np.array(self.X).shape[1],1))
                gt[self.X[i,:] * direction > threshold *direction]=-1
                err = np.sum((gt != self.y)* self.W)
                if err < minErr:
                    minErr = err
                    p_threshold = threshold
            return p_threshold, minErr
                

        minErr = np.inf
        self.p_demension = 0
        self.p_threshold = 0
        self.p_direction = -1 #-1 means less than p_threshold then -1
        for i in range(self.N):
            for direction in [-1, 1]:
                threshold, err = findmin(i, direction, steps)
                if err < minErr:
                    minErr = err
                    self.p_threshold = threshold
                    self.p_direction = direction
                    self.p_demension = i 
        return minErr

                    
    def predict(self, test_set):
        t = np.ones((np.array(test_set).shape[1],1))
        t[test_set[self.p_demension,:] * self.p_direction > self.p_threshold * self.p_direction] = 1
        return t



               


        

        


        


        








