from __future__ import print_function
import numpy as np
import sys
import copy
import time
from amitgroup.net import train_net as tn

def train_sv_all(expi):
     expi.pp.deltaD=expi.pp.deltaP
     numclass=len(expi.ddtr)
     [X,Y]=tn.stack_data(expi)
     #expi.NO=np.zeros((numfeat,numclass))
     expi.NO=one_against_rest_all(expi,X,Y)
     [CC, e]=tn.test_by_weights(expi.ddte,expi.pp,expi.NO)
     return CC, e
     
def one_against_rest_all(expi,X,Y):

    """
     Simultaneously train each svm of one class against
     the rest for each class as you loop over the data.

     Parameters
     ----------

     expi.pp - parameters of optimization:
               expi.pp.stoch < 0 - penalty on weights. More negative more penalty
               expi.pp.deltaP, and expi.pp.deltaD margins for positive and negative examples. (The `1' in the hinge loss.)
               expi.pp.numit - number of iterations (sweeps through all data)
               expi.pp.pltp - initial time step. Decreases as 1/iteration number.

     Returns
     -------

     expi.NO - Is the matrix of weights for the numclass classes.
    """
    pp=expi.pp
    Ntot=Y.size
    numfeat=X.shape[1]
    numclass=len(expi.ddtr)
    W=np.zeros((numfeat,numclass))
    II=range(Ntot)
    # Initial stepsize
    td=1
    # Check stepsize factor isn't too large
    eps=.01
    locltp=pp.pltp
    if pp.stoch<0:
        locltp=min(pp.pltp,abs(1./pp.stoch)-eps)
    print('Time Step', pp.pltp, locltp)
    up=0
    dup=0
    # Iterations.
    for it in range(pp.numit):
        np.random.shuffle(II)
        up=0
        dup=0
        fac1=0
        # Loop over data
        for i in II:
            # Penalized weights
            fac2=locltp/td
            if pp.stoch<0:
                fac1=(1.+pp.stoch*fac2)
                W*=fac1

            tempx=X[i,:]
            tempy=Y[i]
            h=np.dot(tempx,W)
            # We are using binary features so updates in the direction of the example occurs only when the features are nonzero
            # This is NOT the general case.
            pi=tempx>0
            # Update perceptrons for each class.
            for c in range(numclass):
                # A different weight constrain version -Max weight contraint
                if pp.stoch==0:
                    if tempy==c:
                        tempw=W[:,c]<=pp.Jmax-fac2
                        if np.sum(tempw)<len(tempw):
                            print('hit upper bound')
                        tempw.shape=pi.shape
                        pi=np.logical_and(pi, tempw)
                    else:
                        tempw=W[:,c]>=-pp.Jmax+fac2
                        if np.sum(tempw)<len(tempw):
                                print('hit lower bound')
                        tempw.shape=pi.shape
                        pi=np.logical_and(pi,tempw)
                # Update weights on class
                if (tempy==c and h[c]<=pp.deltaP):
                    dup+=1
                    # Count number of updates.
                    up+=np.count_nonzero(pi)
                    W[pi,c]=W[pi,c]+fac2
                # Update weight for off class examples.
                elif (tempy!=c and h[c]>=-pp.deltaD):
                    dup+=1
                    # Count number of updates.
                    up+=np.count_nonzero(pi)
                    W[pi,c]=W[pi,c]-fac2
        # Reduce time step after a sweep through all data.
        if pp.stoch<0:
            td=td+1
        # Show energy value.
        if (np.mod(it,pp.showing)==0):
            DD=0
            for c in range(numclass):
                Yc=2*(Y.T==c)-1
                DD+=np.sum(np.maximum(np.zeros(Ntot),pp.deltaP-np.dot(X,W[:,c])*Yc));
            PP=-.5*pp.stoch*np.sum(W*W)
            EE=DD+PP
            print('td ', td, 'it ', it, 'Number of syn changes ', up, ' at ', dup, ' Data term ', DD, 'Prior term ', PP, 'Total ', EE)
        # Nothing is changing - stop the algorithm.
        if up==0:
            break
    DD=0
    # Final energy.
    for c in range(numclass):
        DD+=np.sum(np.maximum(np.zeros(Ntot),pp.deltaP-np.dot(X,W[:,c])*(2*(Y.T==c)-1)));
    PP=-.5*pp.stoch*np.sum(W*W)
    EE=DD+PP
    print('td ', td, 'it ', it, 'Number of syn changes ', up, ' at ', dup, ' Data term ', DD, 'Prior term ', PP, 'Total ', EE)
    return W


def train_sv(expi, flag=0):
    expi.pp.deltaD=expi.pp.deltaP
    numclass=len(expi.ddtr)
    numfeat=expi.ddtr[0][0].features['V1'].size
    print(numfeat)
    expi.NO=np.zeros((numfeat,numclass))
    for c in range(numclass):
        W=one_against_rest(expi.pp,expi.ddtr,c,expi.pp.numtrain_per_class, flag)
        expi.NO[:,c]=W
    CC, e=tn.test_by_weights(expi.ddte,expi.pp,expi.NO)
    print('result ', e)
    f=open(expi.pp.out,"a")
    f.write('stoch: '+str(expi.pp.stoch) + ' Del: ' + str(expi.pp.deltaP) + ' Rate: ' + str(e) + '\n')

def one_against_rest(pp,ddtr,c,numtrain=0, flag=0):
    print('numtrain', numtrain, 'class ', c)
    if numtrain==0:
        numtrain=len(ddtr[c])
    # Rearrange data for this class with class at top of array and all the rest after.
    print(numtrain)
    XY=tn.rearrange(ddtr,c, numtrain)
    XY[1].shape=[XY[1].size]
    # just for fun make output 1/-1
    XY[1][XY[1]==0]=-1
    Ntot=XY[1].size
    numfeat=XY[0].shape[1]
    W=np.zeros(numfeat)
    II=range(Ntot)
    # Initial stepsize
    td=1
    # Check stepsize factor isn't too large
    eps=.01
    locltp=pp.pltp
    if pp.stoch<0:
        locltp=min(pp.pltp,abs(1./pp.stoch)-eps)
    print('Time Step', pp.pltp, locltp)
    up=0
    dup=0
    for it in range(pp.numit):
        np.random.shuffle(II)
        up=0
        dup=0
        fac1=0
        for i in II:
            if pp.stoch<0:
                fac1=(1.+pp.stoch*locltp/td)
            fac2=locltp/td
            tempx=XY[0][i,:]
            tempy=XY[1][i]
            h=np.dot(tempx,W)
            #print h
            #raw_input()
            pi=tempx>0
            # If L2 weight penalization
            if pp.stoch<0:
                W*=fac1
            # If Max weight contraint
            else:
                if tempy==1:
                    tempw=W<=pp.Jmax-fac2
                    if np.sum(tempw)<len(tempw):
                        print('hit upper bound')
                    tempw.shape=pi.shape
                    pi=np.logical_and(pi, tempw)
                else:
                    tempw=W>=-pp.Jmax+fac2
                    if np.sum(tempw)<len(tempw):
                        print('hit lower bound')
                    tempw.shape=pi.shape
                    pi=np.logical_and(pi,tempw)
            if (tempy==1 and h<=pp.deltaP):
                dup+=1
                up+=np.count_nonzero(pi)
                W[pi]=W[pi]+fac2
            elif (tempy==-1 and h>=-pp.deltaD):
                dup+=1
                up+=np.count_nonzero(pi)
                W[pi]=W[pi]-fac2
        if pp.stoch<0:
            td=td+1
        if (np.mod(it,pp.showing)==0):
            DD=np.sum(np.maximum(np.zeros(Ntot),pp.deltaP-np.dot(XY[0],W)*XY[1]));
            PP=-.5*pp.stoch*np.sum(W*W)
            EE=DD+PP
            print('td ', td, 'it ', it, 'Number of syn changes ', up, ' at ', dup, ' Data term ', DD, 'Prior term ', PP, 'Total ', EE)
        if up==0:
            #time.sleep(20)
            break
        
    DD=np.sum(np.maximum(np.zeros(Ntot),pp.deltaP-np.dot(XY[0],W)*XY[1]));
    PP=-.5*pp.stoch*np.sum(W*W)
    EE=DD+PP
    print('td ', td, 'it ', it, 'Number of syn changes ', up, ' at ', dup, ' Data term ', DD, 'Prior term ', PP, 'Total ', EE)
   
    return W



                

    
