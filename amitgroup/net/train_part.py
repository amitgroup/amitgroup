import numpy as np
import sys
import copy
import time
import train_net as tn


def extract_parts(expi):

   XY=tn.rearrange(expi.ddtr,0,expi.numtrain)
   X=XY[0]
   nt=X.shape[0]
   num_samps=50
   ps2=np.floor(expi.pp.part_size/2)
   imsize=np.sqrt(X.size/(8*nt))
   X.shape=[nt,8,imsize,imsize]
   ZZ=[]
   for t in range(nt):
       ii=np.floor(np.random.rand(num_samps)*(imsize-expi.pp.part_size))+ps2
       jj=np.floor(np.random.rand(num_samps)*(imsize-expi.pp.part_size))+ps2
       for s in range(num_samps):
           Z=X[t,:,ii[s]-ps2:ii[s]+ps2+1,jj[s]-ps2:jj[s]+ps2+1]
           if (np.sum(Z)>expi.pp.min_edges):
               Z.shape=[8,expi.pp.part_size,expi.pp.part_size]
               ZZ.append(Z)

   pptr=np.array(ZZ)
   
   return(pptr)
           
       


                
def train_parts(expi):

    
    pptr=extract_parts(expi)
    Jmid=expi.pp.Jmax/2
    Jqtr=expi.pp.Jmax*1/4
    numtrain=pptr.shape[0]
    numfeat=pptr.size/numtrain
    TT=range(numtrain)
    Parts=[]
    for it in range(expi.pp.numit):
        np.random.shuffle(TT)
        for t in TT:
            hits=0
            XI=(pptr[t,:].flatten())==1
            XI.shape=[numfeat,1]
            for P in Parts:
                h=np.dot(pptr[t,:].flatten(),P-Jmid)
                # Part activated, apply potentiations.
                if (h>expi.pp.theta):
                    hits+=1
                    tn.potentiate_ff(expi.pp,h,XI,P,Jmid)
                else:
                    tn.depress_ff(expi.pp,h,XI,P,Jmid)
            if (hits==0):
                J=np.ones((numfeat,1))*Jqtr
                h=np.dot(pptr[t,:].flatten(),J-Jmid)
                tn.potentiate_ff(expi.pp,h,XI,J,Jmid)
                Parts.append(J)
                print t, len(Parts)
    print len(Parts)
    return Parts
