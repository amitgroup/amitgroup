import numpy as np
import sys
import copy
import time
import train_net as tn
import pylab as plt
import pdb
import amitgroup as ag


def extract_parts(expi):

   #XY=tn.rearrange(expi.ddtr,0,expi.numtrain)
   #X=XY[0]
   aa=[]
   for i in range(expi.numtrain):
      aa.append(ag.io.load_imagep('/home/work/numer/nist/mnist/DEF/VIRT/mnist_train',i,True))
      feat=ag.features.bedges(double(aa[i].img),5,'box')
      aa[i].features=feat
   nt=expi.numtrain
   num_samps=50
   ps2=np.floor(expi.pp.part_size/2)
   imsizex=aa[0].img.shape[0]
   imsizey=aa[0].img.shape[1]
   ZZ=[]
   for t in range(nt):
       ii=np.floor(np.random.rand(num_samps)*(imsizex-expi.pp.part_size))+ps2
       jj=np.floor(np.random.rand(num_samps)*(imsizey-expi.pp.part_size))+ps2
       for s in range(num_samps):
          Z=aa[t].features[:,ii[s]-ps2:ii[s]+ps2+1,jj[s]-ps2:jj[s]+ps2+1]
          if (np.sum(Z[1:expi.pp.part_size,1:expi.pp.part_size])>expi.pp.min_edges):
             a=ag.io.image(0,aa[t].img[ii[s]-ps2:ii[s]+ps2+1,jj[s]-ps2:jj[s]+ps2+1],Z)
             ZZ.append(a)

   #   pptr=np.array(ZZ)
   
   return(ZZ)
           
       


                
def train_parts(expi):

    # Extract random windows from a number of training images
    ZZ=extract_parts(expi)
    print 'No of windows ', len(ZZ)
    raw_input()
    Jmid=expi.pp.Jmax/2
    # Initial value under 1 of each synapse. Is this necessary?
    Jqtr=Jmid*expi.pp.reduction_factor
    numtrain=len(ZZ)
    numfeat=ZZ[0].features.size
    TT=range(numtrain)

    # List of learned parts
    Parts=[]
    for it in range(expi.pp.numit):
        print 'iteration ', it, len(Parts)
        raw_input()
        np.random.shuffle(TT)
        inp=[]
        for t in TT:
            hits=0
            # Find on features, those synapses can modify.
            XI=(ZZ[t].features.flatten())==1
            #XI.shape=[numfeat,1]
            i=0
            numparts=len(Parts)            
            if (numparts>0):
               # Make one array of all part models
               Parray=np.array(Parts)-Jmid
               #print Parray
               Parray.shape=[numparts,numfeat]
               # Multiple part array by features of this image.
               H=np.dot(Parray,ZZ[t].features.flatten())
               #print H
               #raw_input()
               # Find best fit part.
               hi=np.argmax(H,0)
               # This list will be depressed.
               di=np.where(H<expi.pp.theta)[0]
               # If best fit above threshold potentiate
               if (H[hi]>expi.pp.theta):
                  hits+=np.sum(H[hi]>expi.pp.theta)
                  tn.potentiate_ff(expi.pp,H[hi],XI,Parts[hi],Jmid)
               # Depress all other models?
               else:
                  for dii in range(di.size):
                     tn.depress_ff(expi.pp,H[di[dii]],XI,Parts[di[dii]],Jmid)
            # No part model fit this data point - start new part model with potentiation
            if (hits==0):
               J=np.ones(numfeat)*Jqtr
               h=np.dot(ZZ[t].features.flatten(),J-Jmid)
               #print 'h before ', h
               tn.potentiate_ff(expi.pp,h,XI,J,Jmid)
               #print 'h after ', np.dot(ZZ[t].features.flatten(),J-Jmid)
               Parts.append(J)
               print t, len(Parts)
        # Let's get rid of parts that have no takers
        Parray=np.array(Parts)
        #print Parray
        Parray.shape=[len(Parts),numfeat]
        pptr,ims=stack_features(ZZ)
        H=np.dot(Parray-Jmid,transpose(pptr))
        ip=argmax(H,0)
        iip=unique(ip)
        Parts=[]
        Parts=list(Parray[iip,:])
        print 'After pruning num parts ',len(Parts)
    print 'Final number of parts ', len(Parts)
    n=1
    pp=len(Parts)
    # Show means of each part.
    show_clusters(expi,ZZ,Parts)
    return Parts, ZZ

# I am making this change to get it onto the mac
def stack_features(ZZ):
   numfeat=ZZ[0].features.size
   pptr=np.zeros((len(ZZ),numfeat))
   ims=np.zeros((len(ZZ),ZZ[0].img.shape[0],ZZ[0].img.shape[1]))
   for i in range(len(ZZ)):
      ims[i,:,:]=ZZ[i].img
      pptr[i,:]=ZZ[i].features.flatten()

   return pptr, ims

def show_clusters(expi,ZZ,Parts):

   
   nump=len(Parts)
   numparts=len(Parts)
   numfeat=ZZ[0].features.size
   pptr,ims=stack_features(ZZ)

   TT=np.array(Parts)
   TT.shape=[numparts,numfeat]
   Jmid=expi.pp.Jmax/2
   H=np.dot(TT-Jmid,transpose(pptr))
   print H
   raw_input()
   imax=np.argmax(H,0);
   print imax.shape, H.shape
   MM=np.zeros((numparts,numfeat))
   n=1
   num_images_per_row=20
   num_rows=int(ceil(numparts/20.))
   for ip in range(numparts):
      print 'size of cluster ', np.sum(imax==ip)

      if (np.sum(imax==ip)>0):
         Hm=np.mean(pptr[imax==ip,:],0).flatten()
         Im=np.mean(ims[imax==ip,:,:],0)
         MM[ip,:]=Hm
         G=MM[ip,:]
         G.shape=[8,expi.pp.part_size,expi.pp.part_size]
        # for a in range(8):
         plt.subplot(num_rows,num_images_per_row,n)
         n=n+1
         plt.imshow(Im,cmap=get_cmap('gray'))
         #imshow(G[a,:,:], cmap=get_cmap('gray'))
         axis('off')
         
      
