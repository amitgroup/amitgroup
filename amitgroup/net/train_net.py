import numpy as np
import sys
import copy
import time

def top_train(expi):

    expi.NO=[]
    if expi.type==0:
        expi.NO.append(train_net(expi))
    else:
        expi.NO.append(all_at_one_top(expi))

    print "Finished Training"


def read_data(s,numclass,numfeat,const=0):

    dd=[]
    for c in range(numclass):
        ss=s+str(c)+'.bin'
        f=open(ss,'rb')
        dd.append(np.fromfile(file=f,dtype=np.uint8))
        ll=dd[c].shape[0]/numfeat
        print dd[c].shape
        dd[c].shape=[ll,numfeat]
        if const>0:
            dd[c]=np.hstack((dd[c],np.ones((ll,1))))

    return dd

def append_nets(NO,N1):

   
    numclass=len(NO)
    N=[None]*numclass
    for c in range(numclass):
        N[c]=np.copy(NO[c])
        N[c].extend(N1[c])
    return N



def nets_to_mat(NO,Jmid,numperc=0):

    numclass=len(NO);
    if numperc==0:
        numperc=len(NO[0]);
    print numperc
    numfeat=NO[0][0].JJ.size;
    JJ=np.zeros((numfeat,numperc,numclass));
    for c in range(numclass):
       for p in range(numperc):
           NO[c][p].JJ.shape=(numfeat,);
           JJ[:,p,c]=np.double(NO[c][p].JJ)-Jmid;

    return JJ

def test_averages(ddte,pp,NO, numperc=0, numtest=0):

    numclass=len(NO);
    Jmid=np.ceil(pp.Jmax/2)
    JJ=nets_to_mat(NO,Jmid,numperc);
    WW=np.mean(JJ,1);
    CC=test_by_weights(ddte,WW,numtest)
    return CC

def test_by_weights(ddte,WW,numtest=0):
    numclass=len(ddte)
    d=WW.shape[0]
    WW.shape=[d,numclass]
    CONF=np.zeros((numclass,numclass))
    Ntot=0
    for c in range(numclass):
        if numtest==0:
            N=ddte[c].shape[0]
        else:
            N=numtest
        Ntot+=N
        H=np.zeros((N,numclass));
        H=np.dot(ddte[c][0:N,:],WW);
        i=np.argmax(H,1);
        for d in range(numclass):
            CONF[c,d]=np.double(np.sum(i==d))
        
    e=np.sum(np.diag(CONF))/Ntot
    print e
    return [CONF, e]

        
def test_net(expi, numtest=0):  #$ddte,pp,NO,numperc=0, numtest=0):

    Jmid=np.ceil(expi.pp.Jmax/2)
    print Jmid
    numclass=len(expi.ddte);    
    JJ=nets_to_mat(expi.NO[0],Jmid,expi.numperc);
    print JJ[:,:,0].shape
    CONF=np.zeros((numclass,numclass))
    Ntot=0
    for c in range(numclass):
        if numtest==0:
            N=expi.ddte[c].shape[0]
        else:
            N=numtest
        Ntot+=N
        H=np.zeros((N,numclass));
        for d in range(numclass):
            temp=np.dot(expi.ddte[c][0:N,:],JJ[:,:,d])>expi.pp.theta
            H[:,d]=np.sum(temp,1)
        
       
        i=np.argmax(H,1);
        for d in range(numclass):
            CONF[c,d]=np.double(np.sum(i==d))

    print np.sum(np.diag(CONF))/Ntot
    return CONF



                
def train_net(expi):
    f = open(expi.out,'w')
    numclass=len(expi.ddtr)
    CI=range(numclass)
    #np.random.shuffle(CI)
    NO=[None]*numclass
    for c in CI:
        f.write(str(CI[c])+'\n')
        NO[CI[c]]=ff_mult_top(f,expi.pp,expi.ddtr,CI[c],expi.numperc, expi.numtrain)

    return NO

def all_at_one_top(expi):
    
    numclass=len(expi.ddtr)
    N=expi.numtrain
    if N==0:
        N=ddtr[0].shape[0]

    X=np.copy(expi.ddtr[0][0:N,:])
    Y=np.zeros((N,1))
    for c in range(1,numclass):
        N=expi.numtrain
        if N==0:
            N=expi.ddtr[c].shape[0]
        X=np.vstack((X,expi.ddtr[c][0:N,:]))
        Y=np.vstack((Y,c*np.ones((N,1))))
    NN=ff_all_at_one(expi.out,expi.pp,X,Y,expi.numperc,numclass)
    return NN
    
def ff_all_at_one(out,pp,X,Y,numperc,numclass):
    #sys.stdout = open('out','w')
    f=open(out,'w')
    Ntot=X.shape[0]
    numfeat=X.shape[1]
    print Ntot, numfeat
    # Synapses are positive and Jmid is the `middle'. Instead of being symmetric around 0.
    Jmid=np.ceil(pp.Jmax/2)
    # Feed forward synspases - initial value 1 -> 0.
    J=[]
    Jfb=[]
    for c in range(numclass):
        J.append(np.ones((numfeat,numperc))*Jmid)
        # Feedback synapses
        Jfb.append(np.ones((numfeat,numperc))*Jmid)
    # Iterate
    II=range(Ntot)
    for it in range(pp.numit):
        # Random arrangement of examples. Stochastic gradient.
        np.random.shuffle(II)
        # Variables to keep track of changes
        up=0
        down=0
        # Loop over examples
        for i in range(Ntot):
            ii=II[i]
            XI=X[ii,:]==1
            XIz=X[ii,:]==0
            # Prepare for matrix multiplication.
            
            # Field at each perceptron for this class.
            for c in range(numclass):

                h=np.dot(X[ii,:],J[c]-Jmid)
                h.shape=[1,numperc] 
                XI.shape=[pp.d,1]
                if Y[ii]==c:
                    # Update in up direction.
                    up+=potentiate_ff(pp,h,XI,J[c],Jmid)
                    Jfb[c]=modify_fb(pp,XI,XIz,Jfb[c],Jmid)
                else:
                    down+=depress_ff(pp,h,XI,J[c],Jmid)
        # up+down
        f.write('updown '+str(np.double(up)+np.double(down))+'\n')
        
    N=[]
    for c in range(numclass):
        NN=[]
        for p in range(numperc):
            NN.append(netout(J[c][:,p],Jfb[c][:,p]))
        N.append(NN)
            
    return N            
# Train the network for each class.
def ff_mult_top(f,pp,ddtr,c,numperc, numtrain=0):
    if numtrain==0:
        numtrain=ddtr[c].shape[0]
    # Rearrange data for this class with class at top of array and all the rest after.
    XY=rearrange(ddtr,c, numtrain)
    # Train class against the rest perceptron/s
    NO=ff_mult(pp,XY, numperc,f)    
    return NO

def ff_mult(pp,XY,numperc,f):

    # Features
    X=XY[0]
    # Labels 1/0
    Y=XY[1]
    Ntot=X.shape[0]



    # Simple learing rule or field learning rule.
    # Synapses are positive and Jmid is the `middle'. Instead of being symmetric around 0.
    Jmid=np.ceil(pp.Jmax/2)
    # Feed forward synspases - initial value 1 -> 0.
    J=np.ones((pp.d,numperc))*Jmid
    # Feedback synapses
    Jfb=np.ones((pp.d,numperc))*Jmid
    II=range(Ntot)
    # Iterate
    for it in range(pp.numit):
        # Random arrangement of examples. Stochastic gradient.
        np.random.shuffle(II)
        # Variables to keep track of changesi
        up=0
        down=0
        # Loop over examples
        for i in range(Ntot):
            ii=II[i]
            # Field at each perceptron for this class.
            h=np.dot(X[ii,:],J-Jmid)
            # Set of active input features
            XI=X[ii,:]==1
            XIz=X[ii,:]==0
            # Prepare for matrix multiplication.
            XI.shape=[pp.d,1]
            h.shape=[1,numperc]
            # A class example
            if (Y[ii]==1):
                # Update in up direction.
                up+=potentiate_ff(pp,h,XI,J,Jmid)
                Jfb=modify_fb(pp,XI,XIz,Jfb,Jmid)
            else:
                # Update in down direction.
                down+=depress_ff(pp,h,XI,J,Jmid)
        # Report fraction of modified synapses (potentiated, depressed)
        #print [np.double(up)/(numperc*pp.d), np.double(down)/(numperc*pp.d)]
        ## s=raw_input('-->')
        ## if s=='z':
        ##     sys.exit()
        f.write('updown '+str(np.double(up)+np.double(down))+'\n')
        if up+down==0:
            break
    N=[]
    for p in range(numperc):
        N.append(netout(J[:,p],Jfb[:,p]))
    return N

def modify_fb(pp,XI,XIz,Jfb,Jmid):

    # All feedback synapses connected to active features can be potentiated if less than max.
    
    XI.shape=XI.size
    temp=Jfb[XI,:]
    
    IJ=temp<pp.Jmax
    
    g=temp[IJ]
    
    g+=np.random.rand(g.size)<pp.pltp
    temp[IJ]=g
    Jfb[XI,:]=temp
   
    # All feedback synapses connected to inactive features can be depressed if greater than 0.
    XIz.shape=XIz.size
    temp=Jfb[XIz,:]
    IJ=temp>0
    g=temp[IJ]
    g-=np.random.rand(g.size)<pp.pltd
    temp[IJ]=g
    Jfb[XIz,:]=temp
    return Jfb
    
def potentiate_ff(pp,h,XI,J,Jmid):
    # Perceptrons with field below potentiation threshold.
    hii=h<=pp.theta+pp.delta;
    # Logical matrix of all synapses that can be potentiated ... below potentiation threshold
    # and the feature is on. (Synapses with off features don't create a change.
    imat=np.dot(XI,hii)
    # Extract changeable synapses.
    Jh=J[imat]

    # If less than maximal synaptic value
    IJ=Jh<pp.Jmax
    g=Jh[IJ]
    # Modify with stochastic ltp probability.
    RR=(np.random.rand(g.size)<pp.pltp)
    g=g+RR*pp.pinc
    Jh[IJ]=g
    r=sum(RR)
    J[imat]=Jh
    return r

def depress_ff(pp,h,XI,J,Jmid):
     # Perceptrons with field above depression threshold.
    hii=h>=pp.theta-pp.delta;
    # Logical matrix of all synapses that can be depressed ...above depression threshold
    # and the feature is on. (Synapses with off features don't create a change.)
    imat=np.dot(XI,hii)
    Jh=J[imat];
    # If greater than minimal synaptic value
    IJ=Jh>0
    g=Jh[IJ]
    # Modify with stochastic ltd probability.
    RR=(np.random.rand(g.size)<pp.pltd)
    Jh[IJ]=g-RR*pp.pinc
    r=sum(RR)
    J[imat]=Jh
    return r

def rearrange(dd,c,numtrain=0):

    XY=[]   
    ic=range(len(dd))
    ic.remove(c)
    if numtrain==0:
        XY.append(dd[c])
    else:
        XY.append(dd[c][0:numtrain,:])

    N=XY[0].shape[0]

    for ii in ic:
        if numtrain==0:
            XY[0]=np.vstack((XY[0],dd[ii]))
        else:
            XY[0]=np.vstack((XY[0],dd[ii][0:numtrain,]))
        
    Ntot=XY[0].shape[0]
    Nbgd=Ntot-N
    XY.append(np.vstack((np.ones((N,1)),np.zeros((Nbgd,1)))))

    return XY


class pars:
    
        d=None
        N=None
        Jmax=None
        pobj=None
        numit=None
        pltp=None
        pltd=None
        pinc=None
        stoch=None
        nofield=None
        theta=None
        delta=None
        pt=None
        sh=None
        min_edges=None
        part_size=None

        def __init__(self):
            self.d=7200
            self.N=1000
            self.Jmax=2
            self.pobj=.5
            self.numit=5
            self.pltp=.01
            self.pltd=.01
            self.stoch=1
            self.nofield=0
            self.theta=0
            self.delta=5
            self.pt=0
            self.showing=100000
            self.min_edges=40
            self.part_size=7
            self.pinc=1
            self.reduction_factor=.9
            
def compress_nets(NN):

    for Nc in NN:
        for P in Nc:
            P.JJ=np.ubyte(P.JJ)


class experiment:
    ddtr=[]
    ddte=[]
    pp=[]
    numperc=[]
    numtrain=[]
    type=[]
    out=[]
    NO=[]
    def __init__(self,ddtr,ddte,pp,numperc,numtrain,type,out):
        self.ddtr=ddtr
        self.ddte=ddte
        self.pp=pp
        self.numperc=numperc
        self.numtrain=numtrain
        self.type=type
        self.out=out

class netout:
    JJ=[];
    JJfb=[];
    def __init__(self,J,Jfb):
        self.JJ=np.ubyte(J)
        self.JJfb=np.ubyte(Jfb)


class modell:
    NN=[]
    par=[]
    def __init__(self,N,pp):
        self.NN=N
        self.par=pp
        
    


    
    


