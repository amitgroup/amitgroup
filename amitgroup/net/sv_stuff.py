import numpy as np
import sys
import copy
import time
import train_net as tn

def train_sv(expi, flag=0):
    expi.pp.delatD=expi.pp.deltaP
    numclass=len(expi.ddtr)
    numfeat=expi.ddtr[0][0].features['V1'].size
    print numfeat
    expi.NO=np.zeros((numfeat,numclass))
    for c in range(numclass):
        W=one_against_rest(expi.pp,expi.ddtr,c,expi.numtrain_per_class, flag)
        expi.NO[:,c]=W
    CC, e=tn.test_by_weights(expi.ddte,expi.NO)
    f=open(expi.out,"a")
    f.write('stoch: '+str(expi.pp.stoch) + ' Del: ' + str(expi.pp.deltaP) + ' Rate: ' + str(e) + '\n')

def one_against_rest(pp,ddtr,c,numtrain=0, flag=0):
    print 'numtrain', numtrain, 'class ', c
    if numtrain==0:
        numtrain=len(ddtr[c])
    # Rearrange data for this class with class at top of array and all the rest after.
    print numtrain
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
    print 'Time Step', pp.pltp, locltp
    up=0
    dup=0
    for it in range(pp.numit):
        np.random.shuffle(II)
        up=0
        dup=0
        #fac1=(1.+pp.stoch*pp.pltp/sqtd)
        #fac2=pp.pltp/sqtd
        fac1=0
        for i in II:
            if pp.stoch<0:
                fac1=(1.+pp.stoch*locltp/td)
            fac2=locltp/td
            tempx=XY[0][i,:]
            tempy=XY[1][i]
            h=np.dot(tempx,W)
            pi=tempx>0
            # If L2 weight penalization
            if pp.stoch<0:
                W*=fac1
            # If Max weight contraint
            else:
                if tempy==1:
                    tempw=W<=pp.Jmax-fac2
                    if np.sum(tempw)<len(tempw):
                        print 'hit upper bound'
                    tempw.shape=pi.shape
                    pi=np.logical_and(pi, tempw)
                else:
                    tempw=W>=-pp.Jmax+fac2
                    if np.sum(tempw)<len(tempw):
                        print 'hit lower bound'
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
            print 'td ', td, 'it ', it, 'Number of syn changes ', up, ' at ', dup, ' Data term ', DD, 'Prior term ', PP, 'Total ', EE
        if up==0:
            #time.sleep(20)
            break
        if flag:
            s=raw_input('-->')
            if s=='z':
                sys.exit()
    DD=np.sum(np.maximum(np.zeros(Ntot),pp.deltaP-np.dot(XY[0],W)*XY[1]));
    PP=-.5*pp.stoch*np.sum(W*W)
    EE=DD+PP
    print it, 'Number of syn changes ', up, ' at ', dup, ' Data term ', DD, 'Prior term ', PP, 'Total ', EE
    #N=[]
    #N.append(netout(W,W))
    return W



                

    
