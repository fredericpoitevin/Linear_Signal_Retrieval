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
### PLOT FUNCTIONS ###
def plot_signal_gen_vec(U,L,V,S,tol):
    # define dimensions 
    p, n = S.shape # retrieve size parameters
    r = LA.matrix_rank(L,tol=tol) # find rank of L based on tol
    rS = LA.matrix_rank(S,tol=1e-10) # find rank of S based on epsilon
    ################
    # start figure #
    ################
    fig = plt.figure(figsize=(12,6))
    nrow=2
    ncol=3
    # plot the eigenvector U matrix
    plt.subplot2grid((nrow,ncol),(0,0), colspan=1, rowspan=1)
    plt.title('eigenvector(s)')
    for i in np.arange(0,r,1):
        plt.plot(U[:,i])
    # plot the first eigenvalues
    plt.subplot2grid((nrow,ncol),(0,1), colspan=1)
    plt.title('(first) eigenvalues')
    plt.plot(np.arange(0,r+10),np.diag(L)[0:r+10],'X')
    #plt.bar(np.arange(0,r+10),np.diag(L)[0:r+10])
    # plot the population V matrix 
    plt.subplot2grid((nrow,ncol),(0,2), colspan=1)
    plt.title('population')
    for i in np.arange(0,r,1):
        plt.hist(V[:,i],rwidth=0.4,bins=int(n/10))
    # plot the signal matrix
    plt.subplot2grid((nrow,ncol),(1,0), colspan=3)
    plt.title('signal')
    # ... if the signal is zero, assemble it from truncated versions of U, L and V
    if(rS == 0):
        Sshow = np.dot(U[:,0:r],np.dot(L[0:r,0:r],V[:,0:r].T))
    else:
        Sshow = S
    # cosmetic rotation of the matrix
    if(p>n):
        plt.imshow(Sshow.T)
    else:
        plt.imshow(Sshow)
    plt.tight_layout()
    
def plot_signal_gen_cov(U,L,S):
    Cs = np.dot(S,S.T) 
    Cs2 = np.dot(U,np.dot(np.dot(L,L.T),U.T))
    Csrank = LA.matrix_rank(Cs,tol=1.e-10)
    Cs2rank = LA.matrix_rank(Cs2,tol=1.e-10)
    print("Rank of Cs  = ",Csrank)
    print("Rank of Cs2 = ",Cs2rank)
    Csnorm  = LA.norm(Cs)
    Cs2norm = LA.norm(Cs2)
    error   = LA.norm(Cs2-Cs)/np.sqrt(Csnorm*Cs2norm)
    print("Error = ",error)
    plt.figure(figsize=(12,3))
    nrow=1
    ncol=3
    plt.subplot(nrow,ncol,1)
    plt.title('covariance from eigenpairs')
    plt.imshow(Cs2)
    plt.colorbar()
    plt.subplot(nrow,ncol,2)
    plt.title('covariance from signal')
    plt.imshow(Cs)
    plt.colorbar()
    plt.subplot(nrow,ncol,3)
    plt.title('error')
    plt.imshow(Cs-Cs2)
    plt.colorbar()
    plt.tight_layout()
    
def plot_cov_from_vec(M,figtitle):
    n_var, n_sample = M.shape
    C = np.dot(M,M.T)/n_sample #np.dot(M,M.T)
    w, v = LA.eig(C)
    w = np.sqrt(w)
    plt.figure(figsize=(12,4))
    nrow=1
    ncol=3
    plt.subplot(nrow,ncol,1)
    plt.title('sample matrix')
    plt.imshow(M)
    plt.colorbar()
    plt.suptitle(figtitle, fontsize=24)
    plt.subplot(nrow,ncol,2)
    plt.title('covariance matrix')
    plt.imshow(C)
    plt.colorbar()
    plt.subplot(nrow,ncol,3)
    plt.title('sqrt(spectrum)')
    plt.hist(np.real(w),density=True,rwidth=0.8)
    plt.tight_layout()
    
def plot_var(var,threshold,size):
    fig = plt.figure(figsize=(size, size), dpi= 160, facecolor='w', edgecolor='k')
    plt.grid()
    plt.title('Variance ratio explained by PC')
    plt.xlabel('PC ID')
    plt.ylabel('variance ratio')
    plt.semilogy(range(1,1+len(var)), var, 'ko')
    plt.semilogy(range(1,1+len(var)), np.cumsum(var), 'k+')
    #plt.plot(range(1,1+len(var)), var, 'ko')
    #plt.plot(range(1,1+len(var)), np.cumsum(var), 'k+')
    plt.axhline(y=threshold, color='r', linestyle='-')
    plt.show()

def biplots(n,prj,plottype,nbins):
    labels = get_labels(n) 
    fig = plt.figure(figsize=(24, 24), dpi= 160, facecolor='w', edgecolor='k')
    nrow=n
    ncol=n
    color_hexbin='plasma'
    #nbins = 40
    nbins_coarse = int(nbins/1)
    nbox=1 #nrow
    for i in np.arange(0,n,1):
        for j in np.arange(0,n,1):
            ax = fig.add_subplot(nrow,ncol,nbox)
            plt.grid()
            if(i<j):
                if(j<n):
                    if(i == 0):
                        ax.set_xlabel(labels[j])   
                    if(j == n - 1):
                        ax.set_ylabel(labels[i])
                    ax.xaxis.tick_top()
                    ax.yaxis.tick_right()
                    ax.xaxis.set_label_position('top')
                    ax.yaxis.set_label_position('right')
                    Ax = prj[:,j]
                    Ay = prj[:,i]
                    if(plottype == 'scatter'):
                        plt.scatter(Ax, Ay, cmap=color_hexbin)
                    else:
                        plt.hexbin(Ax, Ay, gridsize=nbins, cmap=color_hexbin, mincnt=1)
            elif(i==j):
                Ax = prj[:,i]
                plt.hist(Ax,bins=nbins_coarse)
                #plt.hist(Ay,bins=nbins,range=xlim_array,log=True,rwidth=0.4)
            else:
                if(j == 0):
                    ax.set_ylabel(labels[i])
                if(i == n - 1):
                    ax.set_xlabel(labels[j])
                Ax = prj[:,j]
                Ay = prj[:,i]
                if(plottype == 'scatter'):
                    plt.scatter(Ax, Ay, cmap=color_hexbin)
                else:
                    plt.hexbin(Ax, Ay, gridsize=nbins, cmap=color_hexbin, mincnt=1)
            nbox=nbox+1
    plt.tight_layout()
    plt.show()
    
def plot_alignment(U,L,Usvd,tol,s,save_pngs):
    r = LA.matrix_rank(L,tol=tol) # find rank of L based on tol
    n = L.shape[0]
    Algt=abs(np.dot(U[:,0:r].T,Usvd)) #abs(np.dot(U[:,0:r].T,Usvd))
    Algt_mean=3*np.std(abs(Algt)) + np.mean(abs(Algt))
    Rfrc=3*np.std(abs(np.dot(U[:,r+1:n].T,Usvd)))+ np.mean(abs(np.dot(U[:,r+1:n].T,Usvd)))
    #ymin=0 #-0.5
    ymax=1.1
    fig = plt.figure(figsize=(12,3*r))
    nrow=r
    ncol=1
    for i in np.arange(0,r,1):
        #Algt_norm=LA.norm(Algt[i,:])
        #ax = fig.add_subplot(nrow,ncol,nbox)
        plt.subplot2grid((nrow,ncol),(i,0), colspan=1, rowspan=1)
        plt.title("signal strength %s" % s)
        plt.xlabel("eigenpair ID")
        plt.ylabel("Alignment with signal (%s)" % str(i+1))
        plt.ylim(ymax=ymax)       
        plt.grid()
        plt.plot(L,Algt[i,:],'k-o')
        plt.axhline(y=Rfrc, color='r')
        plt.axhline(y=Algt_mean, color='black')
    plt.tight_layout()
    if(save_pngs):
        plt.savefig("pngs/alignments_%s.png" % s)
        
def plot_stats_component_analysis(threshold,nongauss,var,nPCs,nICs,prj_PCs,prj_ICs,niter):
# Preplotting...
    #Plabels, Ilabels = get_labels(nPCs,nICs)
    Plabels = get_labels(nPCs)
    Ilabels = get_labels(nICs)
    prj_GVs = np.random.randn(len(prj_PCs[:,0]),nPCs)/np.sqrt(len(prj_PCs[:,0]))
    GVscore, GVscore_var = ave_score(prj_GVs,nPCs,niter,nongauss)
    PCscore, PCscore_var = ave_score(prj_PCs,nPCs,niter,nongauss)
    ICscore, ICscore_var = ave_score(prj_ICs,nICs,niter,nongauss)

# FIGURE 1: VARIANCE AND SPANS
    plt.figure(figsize=(12, 8), dpi= 160, facecolor='w', edgecolor='k')
    nrow=2
    ncol=3
# - VARIANCE of PCs
    plt.subplot(nrow,ncol,1)
    plt.grid()
    plt.title('Variance ratio explained by PC')
    plt.xlabel('PC ID')
    plt.ylabel('variance ratio')
    #plt.semilogx(range(1,1+len(var)), var, 'ko')
    #plt.semilogx(range(1,1+len(var)), np.cumsum(var), 'k+')
    plt.plot(range(1,1+len(var)), var, 'ko')
    plt.plot(range(1,1+len(var)), np.cumsum(var), 'k+')
    plt.axhline(y=threshold, color='r', linestyle='-')
    #plt.axhline(y=0.99, color='r', linestyle='-')
# - NEGENTROPY of PCs
    plt.subplot(nrow,ncol,2)
    plt.grid()
    plt.title('Negentropy of  PC')
    plt.xlabel('PC ID')
    plt.ylabel('negentropy')
    plt.errorbar(np.arange(0,nPCs,1),GVscore,yerr=np.sqrt(GVscore_var))
    plt.errorbar(np.arange(0,nPCs,1),PCscore,yerr=np.sqrt(PCscore_var))
    plt.plot(GVscore,'x-')
    plt.plot(PCscore,'o-')
    #plt.plot(np.cumsum(PCscore), 'k-')
# - SPAN of PCs
    plt.subplot(nrow,ncol,3)
    plt.grid()
    plt.title('Population of PCs')
    #plt.xlabel('it')
    plt.ylabel('sorted coordinates')
    for y_arr, label in zip(prj_PCs.T, Plabels):
        plt.plot(np.sort(y_arr), '-', label=label)
    plt.legend()
# - VARIANCE of ICs
# - NEGENTROPY of ICs
    plt.subplot(nrow,ncol,5)
    plt.grid()
    plt.title('Negentropy of  IC')
    plt.xlabel('IC ID')
    plt.ylabel('negentropy')
    plt.errorbar(np.arange(0,nPCs,1),GVscore,yerr=np.sqrt(GVscore_var))
    plt.errorbar(np.arange(0,nICs,1),ICscore,yerr=np.sqrt(ICscore_var))
    plt.plot(GVscore,'x-')
    plt.plot(ICscore,'o-')
    #plt.plot(np.cumsum(ICscore), 'k-')
# - SPAN of ICs
    plt.subplot(nrow,ncol,6)
    plt.grid()
    plt.title('Population of ICs')
    #plt.xlabel(t_axis_label)
    plt.ylabel('A(IC,t)')
    for y_arr, label in zip(prj_ICs.T, Ilabels):
        yzero = np.mean(y_arr)
        plt.plot(np.sort(y_arr-yzero), '-', label=label)
    plt.legend()
#
    plt.tight_layout()
    plt.show()
    
def get_labels(n):
    labels = []
    for i in np.arange(0,n,1):
        labels.append('v'+str(i+1))
    return labels
