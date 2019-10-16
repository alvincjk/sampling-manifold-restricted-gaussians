# definition of iteration class

import numpy as np
import time

class iteration:
    
    def __init__(self, betastar, minisize=100, epsilon=1e-2, bounds=None, autocompact=True, replace=True):
        
        self.betastar = betastar
        self.minisize = minisize
        self.epsilon = epsilon
        self.bounds = bounds
        self.autocompact = autocompact
        self.replace = replace
        
    def __call__(self, theta, alpha, jacobian, hessian=None):
        
        sdim = theta.size
        ddim = alpha.size
        self.timing = [time.time()]
        
        self.metric = np.einsum('ij,ik->jk', jacobian, jacobian)
        invmet = np.linalg.inv(self.metric)
        self.pseudoinverse = np.einsum('ij,kj->ik', invmet, jacobian)
        self.timing.append(time.time())
        
        self.compactness = 1/self.epsilon
        if self.autocompact and (self.bounds is not None):
            sqrtlam = np.einsum('ij,jk->ik', np.diag(2/np.diff(self.bounds)[:,0]), self.pseudoinverse)
            self.metricterm = np.amax(np.linalg.svd(sqrtlam)[1]**2)
            self.compactness = self.metricterm/self.epsilon
        else: self.metricterm = None
        if (self.metricterm is not None) and (hessian is not None):
            evals, evecs = np.linalg.eig(invmet)
            qmat = np.einsum('ij,jk->ik', evecs, np.diag(np.sqrt(evals)))
            ii = hessian-np.einsum('ij,jkl->ikl', jacobian, np.einsum('ij,jkl->ikl', self.pseudoinverse, hessian))
            self.second = np.linalg.norm(ii, axis=0)
            kmat = np.einsum('ji,jk->ik', qmat, np.einsum('ij,jk->ik', self.second, qmat))
            self.curvatureterm = np.amax(np.linalg.eig(kmat)[0])
            self.compactness = max(self.metricterm, self.curvatureterm)/self.epsilon
        else: self.curvatureterm = None
        self.timing.append(time.time())
        
        self.beta = alpha+np.reshape(np.random.normal(0, 1/np.sqrt(self.compactness), self.minisize*ddim), [self.minisize,ddim])
        self.timing.append(time.time())
        
        self.samples = theta+np.einsum('ij,kj->ki', self.pseudoinverse, self.beta-alpha)
        self.timing.append(time.time())
        
        self.pushforward = alpha+np.einsum('ij,kj->ki', jacobian, self.samples-theta)
        logfac = np.log((1+self.compactness)/self.compactness)*sdim/2
        lvec = np.einsum('ji,kj->ki', jacobian, self.pushforward-self.betastar)
        rvec = np.einsum('ij,kj->ki', self.pseudoinverse, self.pushforward-self.betastar)
        negexp = np.sum(lvec*rvec, axis=1)/(1+self.compactness)/2
        self.logweights = logfac-negexp
        self.timing.append(time.time())
        
        if self.replace and (self.bounds is not None):
            mask = np.any((self.samples<self.bounds[:,0])|(self.samples>self.bounds[:,1]), axis=1)
            self.samples[mask] = np.tile(theta, [np.sum(mask),1])
            self.logweights[mask] = np.tile(0, np.sum(mask))
        self.timing.append(time.time())
        
        self.timing = np.diff(np.array(self.timing))
