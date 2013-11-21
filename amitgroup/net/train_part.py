from __future__ import absolute_import, print_function
import numpy as np
import pylab as plt
import pdb
import amitgroup as ag

import os


def make_synthetic_parts(expi):
   IM=[]
   for i in range(expi.pp.numtrain):
      a=np.zeros((7,7))
      x=np.floor(np.random.rand()*5)+1
      print(x)
      horiz=True #(np.random.rand()<.5)
      for xi in np.arange(0,1):
         for y in range(7):
            if horiz:
               a[x+xi,y]=255
            else:
               a[y,x+xi]=255
         
      im=ag.io.image(0,a)
     
      im.features={'V1': np.ubyte(im.img>0)}
      IM.append(im)

   return IM
         

def extract_parts(expi, data=[]):

   #XY=ag.net.rearrange(expi.ddtr,0,expi.pp.numtrain)
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
          if (expi.feat=='V1'):
            feat=ag.features.bedges(np.double(aa[-1].img),5,'box',expi.pp.spread)
            aa[-1].features={'V1': feat}
          
            
          #feat=np.ubyte(aa[-1].img>0)
          #feat.shape=[1,feat.shape[0],feat.shape[1]]
          
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

   if (data==[]):
        ZZ=extract_parts(expi,data)
   else:
       ZZ=data
   dd=stack_features(ZZ)
   ddi=dd[0].swapaxes(1,2).swapaxes(2,3)
   print(dd[0].shape)
   mixture = ag.stats.BernoulliMixture(expi.pp.numparts, ddi)
   mixture.run_EM(1e-3)
   dummy=np.zeros(mixture.num_mix);
   show_clusters(ZZ,dummy,mixture.mixture_components())
   return mixture
                
def train_parts(expi, data=[]):

   ZZ=[]
   if (data!=[]):
      ZZ=data
    # Extract random windows from a number of training images
    #ZZ=extract_parts(expi,data)
   print('No of windows ', len(ZZ))
    #raw_input()
   Jmid=expi.pp.Jmax/2
   # Initial value under 1 of each synapse. Is this necessary?
   Jqtr=Jmid*expi.pp.reduction_factor
   numtrain=len(ZZ)

   TT=range(numtrain)
   numfeat=ZZ[0].features['V1'].size
   # Initialize parts
   Parts=parts()
   Parts.Nums=np.zeros(expi.pp.numparts)
   Parts.Weights=[]
   for ip in range(expi.pp.numparts):
     J=np.ones(numfeat)
     JL=range(numfeat)
     np.random.shuffle(JL)
     J[JL[0:int(np.round(numfeat*expi.pp.zero_prior))]]=0
     J[JL[numfeat-int(np.round(numfeat*expi.pp.two_prior)):numfeat-1]]=2
     Parts.Weights.append(J)
     print(J.reshape(7,7))
     
   [Zarray, Iarray]=stack_features(ZZ)
   sh=Zarray.shape
   print(sh)
   Zarray.shape=[ sh[0], sh[1]*sh[2]]
   Zarray=Zarray.transpose()
   print(Zarray.shape)

   totpot=0
   totdep=0
   for it in range(expi.pp.numit):
        print('iteration ', it, len(Parts.Weights))
        #raw_input()
        np.random.shuffle(TT)
        
        for t in TT:
          
            [Parts, Parray]=update_with_data_point(expi,Parts,Zarray,t,Jmid)
        # Let's get rid of parts that have no takers
        # End of t loop
        # ip - index of best part for each data point.
        # Get sorted list of best part for each data point.
            [ip,H]=get_best_part(Parray,Zarray)
            pdb.set_trace()
            print(Parray.shape)
            print(Zarray.shape)
            print(ip.size)
            print(Zarray[:,t].reshape(7,7))
            print(t)
            print(H[ip[t],t])
            print(ip[t])
            a=0
            for p in Parts.Weights:
              print(a) 
              print(p.reshape(7,7))
              a+=1
            raw_input()
        bef_len=len(Parts.Weights)
        # List of parts and list of indices where parts change.
        ip=np.sort(ip)
        lip=np.unique(ip,True)
        print('list of parts ', lip[0])
        print('list of indices ', lip[1])
           #raw_input()
        # Put list of indices of where parts change in an array + additional entry with length.   
        lipa=np.zeros(lip[1].size+1)
        lipa[0:lip[1].size]=lip[1]
        lipa[lip[1].size]=ip.size-1
        # Find indices where index increment greater than 1 - cluster with more than 1 element.
        dlip=np.where(lipa[1:lipa.size]-lipa[0:lipa.size-1]>1)
        iip=lip[0][dlip]
        Parts=[]

        Parray=Parray+Jmid
        Parts=list(Parray[iip,:])
        print('Before pruning ', bef_len, 'After pruning num parts ',len(Parts.Weights), 'Pot ', totpot, 'Dep ', totdep)
        for p in Parts.Weights:
           print(p.reshape(7,7))
        raw_input()
        # End of iteration loop
   print('Final number of parts ', len(Parts))
   
   # Show means of each part.
   Parray=np.array(Parts)
   Parray.shape=[len(Parts),numfeat]
   Parray=Parray-Jmid
   show_clusters(ZZ,Zarray,Parray)
   return Parts, ZZ

# I am making this change to get it onto the mac
def stack_features(ZZ):

   pptr=np.uint8(np.zeros(( (len(ZZ),)+ZZ[0].features['V1'].shape )))
   ims=np.uint8(np.zeros((len(ZZ),ZZ[0].img.shape[0],ZZ[0].img.shape[1])))
   for i in range(len(ZZ)):
      ims[i,:,:]=ZZ[i].img
      pptr[i,]=ZZ[i].features['V1']#.flatten()

   return pptr, ims



def update_with_data_point(expi,Parts,Zarray,t,Jmid):
            # Make logical
            Xt=Zarray[:,t]==1
            numfeat=np.size(Xt)
            numparts=len(Parts.Weights)            
            # Make one array of all part models
            Parray=np.array(Parts.Weights)-Jmid
            Parray.shape=[numparts,numfeat]
            # Multiply part array by features of this image.
            H=np.dot(Parray,Xt)
            # Find highest field
            hh=np.max(H,0)
            # Find indices corresponding to highest field
            hhi=np.array(np.where(H==hh)).flatten()
            #print(Parts[hi].reshape(7,7))
            thr=0
            cont=1
            # Loop through parts with highest field and if there is an update
            # stop
            print(hhi)
            print(hhi.shape)
            pdb.set_trace()
            for i in range(hhi.size):
               hi=hhi[i]
               if (cont):
                  thr=0
                  if (Parts.Nums[hi]>0):
                     thr=7
                  if (hi>=thr):
                     cont=0
                     s=ag.net.train_net.potentiate_ff(expi.pp,H[hi],Xt,Parts.Weights[hi],Jmid)
                     Parts.Nums[hi]+=1
               # Depress all other models where field is below threshold.
                     s=0
                     for di in range(numparts):
                        if (di!=hi):
                           s+=ag.net.train_net.depress_ff(expi.pp,H[di],Xt,Parts.Weights[di],Jmid)

            Parray=np.array(Parts.Weights)-Jmid
            Parray.shape=[numparts, numfeat]    
            return Parts, Parray    
            

            



def get_best_part(Parray, pptr):
   
   H=np.dot(Parray,pptr)
   ip=np.argmax(H,0)
   
   return ip, H

def show_clusters(ZZ,Zarray,Parray, imax=[]):

   plt.close('all')
   numparts=Parray.shape[0]
   if (imax==[]):
        [imax,H]=get_best_part(Parray,Zarray)
   
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
      
class parts:
    Nums=[];
    Weights=[];
    

