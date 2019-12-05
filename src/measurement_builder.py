import matplotlib.pyplot as plt
import matplotlib.axis as axis
import numpy as np
import numpy.random
from numpy import linalg as LA
from scipy import linalg as sLA
from scipy.stats import ortho_group
import scipy.stats as ss
from sklearn.preprocessing import normalize
from sklearn.decomposition import FastICA, PCA

#=======SIGNAL=====#
def build_signal(l,p,n,method,gmm_params,gmm_weights): # Here we build S = ULVt
    U = build_Umat(p,method)
    L = build_Lmat(l,p,n)
    V = build_Vmat(l,n,method,gmm_params,gmm_weights)
    S = np.dot(U,np.dot(L,V.T))
    return S, U, L, V

def build_Umat(p,method): # U is a p-by-p unitary matrix
    if (method == 'fromV'):
        U = ortho_group.rvs(p) #U = ortho_group.rvs(np.maximum(n,p))[0:p,0:n]
    else:
        U = np.zeros((p,p)) #U = np.zeros((p,n))
        U[0,0] = 1
    return U
        
def build_Lmat(l,p,n): # L is a p-by-n diagonal matrix
    r = LA.matrix_rank(np.diag(l),tol=1.e-10)
    maxd = np.maximum(n,p)
    L = np.zeros(maxd)
    L[0:r] = l
    L = np.diag(L)[0:p,0:n]
    return L

def build_Vmat(l,n,method,gmm_params,gmm_weights): # V is a n-by-n unitary matrix
    V = ortho_group.rvs(n)
    if (method == 'fromV'):
        # we populate the first r columns of V
        r = LA.matrix_rank(np.diag(l),tol=1.e-10)
        for i in np.arange(r):
            V[:,i] = draw_gmm(n,gmm_params[i,:,:],gmm_weights[i,:])
        V = sLA.orth(V)
    return V

def draw_gmm(n,gmm_params,gmm_weights):
    # Taken from https://stackoverflow.com/questions/49106806/how-to-do-a-simple-gaussian-mixture-sampling-and-pdf-plotting-with-numpy-scipy
    # Set-up.
    numpy.random.seed(0x5eed)
    n_components = gmm_params.shape[0] # Parameters of the mixture components
    weights = gmm_weights              # Weight of each component
    mixture_idx = numpy.random.choice(n_components, size=n, replace=True, p=weights) # A stream of indices from which to choose the component
    y = numpy.fromiter((ss.norm.rvs(*(gmm_params[i])) for i in mixture_idx), dtype=np.float64) # y is the mixture sample
    return y
#================#


#======NOISE====#
def build_noise(p,n,distribution):
    Y = np.random.randn(p,n)
    if (distribution == 'identity'):
        X=Y
    elif (distribution == 'gaussian'):
        E = np.random.randn(p)
        E = np.diag(E)
        X = np.dot(E,Y)
    return X
#================#


#==== MEASURE ===#
def build_measurement(X,S,s):
    M = X + s*S
    return M
#================#


#=====RETRIEVE===#
def perform_svd(M, center=False):
    if center:
        M -= np.mean(M,axis=0)
    U,L,V = LA.svd(M,full_matrices=True)
    return U,L,V.T
#================#

#======COMPONENT ANALYSIS=======#
def component_analysis(M,threshold,negent_order,icalgo,nongauss,ica_iter,ica_tol,optimize_ICset,niter):
    M_mean = np.mean(M,axis=0)
    M = M - M_mean.T
# SVD
    U, L, V = perform_svd(M)
    nPCs, var = get_truncate_order(L,threshold)
    Ul, Ll, Vl = truncate(U,L,V,nPCs)
    print("Number of components kept: ",nPCs)
# Reorder based on negentropy
    if(negent_order):
        PCscore, PCscore_var = ave_score(Vl,nPCs,niter,nongauss)
        index = np.argsort(PCscore)[::-1]
        Ul, Ll, Vl = reorder(Ul,Ll,Vl,index)
# ICA
    prj_PCs = Vl
    prj_ICs, mix_ICs, unmix_ICs, iter_cvg = perform_ica(prj_PCs,icalgo,nongauss,ica_iter,ica_tol)
    print(ica_iter,iter_cvg)
    nICs=nPCs
    # Finally, get Independent modes:
    xyz_ICs = np.dot(unmix_ICs,Ul[:,0:nICs].T)
#    
    return nPCs,nICs,var,Ul,Vl,prj_ICs,xyz_ICs,mix_ICs,unmix_ICs

def perform_ica(X,icalgo,nongauss,ica_iter,ica_tol):
    #                   X = S A.T <=> S = X W.T
    # X (n_samples, n_features)    / S (n_samples, n_components)
    # A (n_features, n_components) / W (n_components, n_features)
    #ica = FastICA(whiten=False,algorithm=icalgo,fun=nongauss,max_iter=ica_iter, tol=ica_tol)
    ica = FastICA(whiten=True,algorithm=icalgo,fun=nongauss,max_iter=ica_iter, tol=ica_tol)
    S = ica.fit_transform(X) # Fit and apply the unmixing matrix to recover the sources
    A = ica.mixing_          # The mixing matrix
    W = ica.components_      # The unmixing matrix
    tot_iter = ica.n_iter_   # number of iterations to converge
    return S,A,W,tot_iter
        

# def perform_noisy_ica (to be investigated)
    # see chapter in Hyvarinen book.
    # Whitening is done using C-Sigma instead of just C
    # where C is data covariance and Sigma is noise covariance
    #     \tilde{x} = (C-Sigma)^{-1/2}x
    # Then fixed-point iteration is slightly changed as we add 
    #     \tilde{Sigma} = (C-Sigma)^{-1/2}Sigma(C-Sigma)^{-1/2}
    #  in \star{w} = E{xg(wtx)} - (I+\tilde{Sigma})wE(g'(wtx))  <=  x is actually \tilde{x} here


    
    
#====MISC===#
    
def reorder(Ul,Ll,Vl,index):
    Ul = Ul[:,index]
    Ll = Ll[index]
    Vl = Vl[:,index]
    return Ul, Ll, Vl

def get_truncate_order(L,threshold):
    var = L**2
    var /= np.sum(var)
    for i in np.arange(0,len(var)-1,1):
        var_current = np.cumsum(var)[i]
        var_next = np.cumsum(var)[i+1]
        if(var_current < threshold and var_next > threshold):
            nPCs=i+1
    return nPCs, var

def truncate(U,L,V,nPCs):
    Ul = U[:,0:nPCs]
    Ll = L[0:nPCs]
    Vl = V[:,0:nPCs]
    return Ul, Ll, Vl

def ave_score(X,n,niter,fun):
    score_ave=[]
    score_var=[]
    for i in np.arange(0,n,1):
        score_tmp = []
        for j in np.arange(0,niter,1):
            score_tmp.append(negent_score(X[:,i],fun))
        score_ave.append(np.mean(score_tmp))
        score_var.append(np.var(score_tmp))
    return score_ave,score_var

def negent_score(X,fun):
    # We compute J(X) = [E(G(X)) - E(G(Xgauss))]**2
    # We consider X (and Xgauss) to be white, in the sense that E(X,X.T)=I
    # The expectation being approximated by the sample mean in our case: np.dot(X,X.T)/n=I
    # In practice, we assume that X has already been normalized by its length [np.dot(X,X.T)=I]
    # so we rescale by np.sqrt(n) before we take the expectation value of G(X).
    length=len(X)
    Xscale = X*np.sqrt(length)
    Xgauss = np.random.randn(length)
    if(fun == 'logcosh'):
        n1 = np.mean(f_logcosh(Xscale)) #np.sum(f_logcosh(Xscale))
        n2 = np.mean(f_logcosh(Xgauss)) #np.sum(f_logcosh(Xgauss))
    elif(fun == 'exp'):
        n1 = np.mean(f_exp(Xscale))     #np.sum(f_exp(Xscale))
        n2 = np.mean(f_exp(Xgauss))     #np.sum(f_exp(Xgauss))
    elif(fun == 'rand'):
        n1 = np.mean(f_logcosh(Xgauss)) #np.sum(f_logcosh(Xgauss))
        n2 = 0 
    negent = (n2-n1)**2
    return negent

def f_logcosh(X):
    return np.log(np.cosh(X))

def f_exp(X):
    return -np.exp(-(X**2)/2)

