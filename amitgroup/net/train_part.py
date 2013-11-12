from __future__ import absolute_import, print_function
import numpy as np
import sys
import copy
import time
from amitgroup.net import train_net as tn
import pylab as plt
import pdb
import amitgroup as ag
import os


def make_synthetic_parts(expi):
   IM=[]
   for i in range(expi.pp.numtrain):
      a=np.zeros((7,7))
      x=np.floor(np.random.rand()*5)+1
      horiz=(np.random.rand()<.5)
      for xi in np.arange(-1,2):
         for y in range(7):
            if horiz:
               a[x+xi,y]=255
            else:
               a[y,x+xi]=255
         
      im=ag.io.image(0,a)
      IM.append(im)

   return IM
         

def extract_parts(expi, data=[]):

   #XY=tn.rearrange(expi.ddtr,0,expi.pp.numtrain)
   #X=XY[0]
   aa=[]
   sstr=""
   if (data==[]):
      sstr=os.environ['HOME']+'/Desktop/Dropbox/mnist_train'
   else:
      aa=data
   for i in range(expi.pp.numtrain):
       if (sstr!="" and (expi.pp.special_class<0 or ag.io.get_tr(sstr,i)==expi.pp.special_class)):
          aa.append(ag.io.load_imagep(sstr,i,True))
          feat=ag.features.bedges(np.double(aa[-1].img),5,'box',expi.pp.spread)
          #feat=np.ubyte(aa[-1].img>0)
          #feat.shape=[1,feat.shape[0],feat.shape[1]]
          aa[-1].features={'V1': feat}
       else:
          feat=ag.features.bedges(np.double(aa[i].img),5,'box',expi.pp.spread)
          aa[i].features={'V1' :feat}
       
   nt=len(aa)
   print('NT', nt)
   num_samps=50
   lps=expi.pp.part_size
   if (lps<0):
      num_samps=1
      lps=aa[0].img.shape[0]
      
   ps2=np.floor(lps/2)
   md=np.mod(lps,2)
   print(md)
   imsizex=aa[0].img.shape[0]
   imsizey=aa[0].img.shape[1]
   ZZ=[]

   for t in range(nt):
      if (expi.pp.part_size>0):
         ii=np.floor(np.random.rand(num_samps)*(imsizex-expi.pp.part_size))+ps2
         jj=np.floor(np.random.rand(num_samps)*(imsizey-expi.pp.part_size))+ps2
         for s in range(num_samps):
          Z=aa[t].features['V1'][:,ii[s]-ps2:ii[s]+ps2+md,jj[s]-ps2:jj[s]+ps2+md]

          if (np.sum(Z[:,1:expi.pp.part_size-1,1:expi.pp.part_size-1])>expi.pp.min_edges):
             a=ag.io.image(0,aa[t].img[ii[s]-ps2:ii[s]+ps2+md,jj[s]-ps2:jj[s]+ps2+md])
             a.features={'V1' : Z}
             ZZ.append(a)
      else:
         ZZ.append(aa[t])
   #   pptr=np.array(ZZ)
   print('Number of part training data',len(ZZ))
   return(ZZ)
           

def train_parts_EM(expi, data=[]):

   ZZ=extract_parts(expi,data)
   dd=stack_features(ZZ)
   ddi=dd[0].swapaxes(1,2).swapaxes(2,3)
   print(dd[0].shape)
   mixture = ag.stats.BernoulliMixture(expi.pp.numparts, ddi)
   mixture.run_EM(1e-3)
   dummy=np.zeros(mixture.num_mix);
   show_clusters(ZZ,dummy,mixture.mixture_components())
   return mixture
                
def train_parts(expi, data=[]):

    # Extract random windows from a number of training images
    ZZ=extract_parts(expi,data)
    print('No of windows ', len(ZZ))
    raw_input()
    Jmid=expi.pp.Jmax/2
    # Initial value under 1 of each synapse. Is this necessary?
    Jqtr=Jmid*expi.pp.reduction_factor
    numtrain=len(ZZ)
    numfeat=ZZ[0].features['V1'].size
    TT=range(numtrain)

    # List of learned parts
    Parts=[]
    totpot=0
    totdep=0
    for it in range(expi.pp.numit):
        print('iteration ', it, len(Parts))
        #raw_input()
        np.random.shuffle(TT)
        inp=[]
        tc=0
        for t in TT:
            hits=0
            # Find on features, those synapses can modify.
            XI=(ZZ[t].features['V1'].flatten())==1
            #XI.shape=[numfeat,1]
            i=0
            numparts=len(Parts)            
            if (numparts>0):
               # Make one array of all part models
               Parray=np.array(Parts)-Jmid
               #print Parray
               Parray.shape=[numparts,numfeat]
               # Multiply part array by features of this image.
               H=np.dot(Parray,ZZ[t].features['V1'].flatten())
               if (expi.pp.showing==1):
                  print('Fields', H)
               #raw_input()
               # Find best fit part.
               hi=np.argmax(H,0)
               # This list will be depressed.
               di=np.where(np.logical_and(H<expi.pp.theta,H>expi.pp.theta-expi.pp.deltaD))[0]
               if (expi.pp.showing==1):
                  print(di)
               # If best fit above threshold potentiate
               if (H[hi]>expi.pp.theta):
                  hits=1
                  s=tn.potentiate_ff(expi.pp,H[hi],XI,Parts[hi],Jmid)
                  totpot+=s
                  if (expi.pp.showing==1):
                     print('pot ', s)
               # Depress all other models where field is below threshold.
               s=0
               for dii in range(di.size):
                  s+=tn.depress_ff(expi.pp,H[di[dii]],XI,Parts[di[dii]],Jmid)
                  totdep+=s
               if (expi.pp.showing==1):
                  print('dep ', s)
            # No part model fit this data point - start new part model with potentiation
            if (hits==0):
               J=np.ones(numfeat)*Jqtr
               h=np.dot(ZZ[t].features['V1'].flatten(),J-Jmid)
               if (expi.pp.showing==1):
                  print('h before ', h)
               # Learn until field is positive.
               w=0
               while (h<=expi.pp.theta):
                  w+=1
                  tn.potentiate_ff(expi.pp,h,XI,J,Jmid)
                  h=np.dot(ZZ[t].features['V1'].flatten(),J-Jmid)
               if (expi.pp.showing==1):
                  print('h after ', w, 'iterations ', np.dot(ZZ[t].features['V1'].flatten(),J-Jmid))
               Parts.append(J)

            if (expi.pp.showing==1):
               print(tc, t, len(Parts))
               [ip,Parray,H]=get_best_part(Parts,ZZ,Jmid)
               print(H)
               print('Parray')
               print(Parray)
               raw_input()
            tc+=1
        # Let's get rid of parts that have no takers
        [ip,Parray,H]=get_best_part(Parts,ZZ, Jmid)
        bef_len=len(Parts)
        lip=np.unique(ip,True)
        if (expi.pp.showing==1):
           print('list of parts ', lip[0])
           print('list of indices ', lip[1])
           #raw_input()
        lipa=np.zeros(lip[1].size+1)
        lipa[0:lip[1].size]=lip[1]
        lipa[lip[1].size]=ip.size-1
        dlip=np.where(lipa[1:lipa.size]-lipa[0:lipa.size-1]>1)
        iip=lip[0][dlip]
        if (expi.pp.showing==1):
           print('list of surviving parts that had more than one element', iip)
        #  iip=np.unique(ip)
        Parts=[]
        Parts=list(Parray[iip,:])
        print('Before pruning ', bef_len, 'After pruning num parts ',len(Parts), 'Pot ', totpot, 'Dep ', totdep)
    print('Final number of parts ', len(Parts))
    n=1
    pp=len(Parts)
    # Show means of each part.
    Parray=np.array(Parts)
    Parray.shape=[len(Parts),numfeat]
    Parray=Parray-Jmid
    show_clusters(ZZ,Parray)
    return Parts, ZZ

# I am making this change to get it onto the mac
def stack_features(ZZ):
   numfeat=ZZ[0].features['V1'].size
   pptr=np.zeros(( (len(ZZ),)+ZZ[0].features['V1'].shape ))
   ims=np.zeros((len(ZZ),ZZ[0].img.shape[0],ZZ[0].img.shape[1]))
   for i in range(len(ZZ)):
      ims[i,:,:]=ZZ[i].img
      pptr[i,]=ZZ[i].features['V1']#.flatten()

   return pptr, ims

def get_best_part(Parray, pptr):
   
   H=np.dot(Parray,np.transpose(pptr))
   print(H.shape)
   ip=np.argmax(H,0)
   print(ip)
   ip=np.sort(ip)
   return ip, H

def show_clusters(ZZ,Parray, imax=[]):

   plt.close('all')
   numparts=Parray.shape[0]
   if (imax==[]):
      numfeat=Parray.shape[1]
      [imax,H]=get_best_part(Parray,pptr)
   
   pptr,ims=stack_features(ZZ)
   print(imax.shape)
   n=1
   num_images_per_row=20
   num_rows=int(np.ceil(numparts/20.))
   num_type=pptr[0,:].size/ims[0,:].size
   for ip in range(numparts):
      print(ip, 'size of cluster ', np.sum(imax==ip))

      if (np.sum(imax==ip)>0):

         Im=np.mean(ims[imax==ip,:,:],0)      
         plt.subplot(num_rows,num_images_per_row,n)
         plt.imshow(Im,cmap=plt.get_cmap('gray'))
         plt.axis('off')
         #plt.subplot(num_rows,num_images_per_row,n+num_rows*10)
         #pt=Parray[ip,:]
         #pt.shape=[num_type,Im.shape[0],Im.shape[1]]
         #plt.imshow(pt[0,:,:], cmap=plt.get_cmap('gray'))
         #plt.axis('off')
         n=n+1

        
   plt.hold(False)
      
