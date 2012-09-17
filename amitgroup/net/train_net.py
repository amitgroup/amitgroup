import numpy as np

def read_data(s,numclass,numfeat):

    dd=[]
    for c in range(numclass):
        ss=s+str(c)+'.bin'
        f=open(ss,'rb')
        dd.append(np.fromfile(file=f,dtype=np.uint8))
        ll=dd[c].shape[0]/numfeat
        dd[c].shape=[ll,numfeat]

    return dd

def test_net(ddte,pp,NO,numtest=0):

    numperc=len(NO[0]);
    numclass=len(ddte);


    JJ=np.zeros((ddte[0].shape[1],numperc,numclass));
    for c in range(numclass):
       for p in range(numperc):
           JJ[:,p,c]=np.double(NO[c][p].JJ)-1
    print JJ[:,:,0].shape
    CONF=np.zeros((numclass,numclass))
    for c in range(numclass):
        if numtest!=0:
            N=numtest 
        else:
            N=ddte[c].shape[0]
        
        H=np.zeros((N,numclass));
        for d in range(numclass):
            temp=np.dot(ddte[c][0:N,:],JJ[:,:,d])>pp.theta
            H[:,d]=np.sum(temp,1)
        
   
        i=np.argmax(H,1);
        
        for d in range(numclass):
            CONF[c,d]=np.double(np.sum(i==d))/N
        
    print np.mean(np.diag(CONF))
    return CONF

def train_net(ddtr,pp,numperc,numtrain):

    numclass=len(ddtr)
    CI=range(numclass)
    np.random.shuffle(CI)
    NO=[None]*numclass
    for c in CI:
        print CI[c]
        NO[CI[c]]=ff_mult_top(pp,ddtr,CI[c],numperc, numtrain)

    return NO

# Train the network for each class.
def ff_mult_top(pp,ddtr,c,numperc, numtrain=0):

    if numtrain==0:
        numtrain=ddtr[c].shape[0]
    # Rearrange data for this class with class at top of array and all the rest after.
    XY=rearrange(ddtr,c, numtrain)
    NO=ff_mult(pp,XY, numperc)

    return NO

def ff_mult(pp,XY,numperc):

    X=XY[0]
    Y=XY[1]
    Ntot=X.shape[0]



    # Simple learing rule or field learning rule.
    Jmid=np.ceil(pp.Jmax/2)
    # Feed forward synspases
    J=np.ones((pp.d,numperc))*Jmid
    # Feedback synapses
    Jfb=np.ones((pp.d,numperc))*Jmid
    II=range(Ntot)
    for it in range(pp.numit):
        np.random.shuffle(II)
        up=0
        down=0
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
            if (Y[ii]==1):
                up+=potentiate_ff(pp,h,XI,J)
                modify_fb(pp,XI,XIz,Jfb)
            else:
                down+=depress_ff(pp,h,XI,J)
        # Report fraction of modified synapses (potentiated, depressed)
        print [np.double(up)/(numperc*pp.d), np.double(down)/(numperc*pp.d)]
    N=[]
    for p in range(numperc):
        N.append(netout(J[:,p],Jfb[:,p]))
    return N

def modify_fb(pp,XI,XIz,Jfb):

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
    
def potentiate_ff(pp,h,XI,J):
    # Perceptrons with field below potentiation threshold.
    hii=h<=pp.theta+pp.delta;
    # Logical matrix of all synapses that can be potentiated ... 
    # If less than maximal synaptic value
    imat=np.dot(XI,hii)
    Jh=J[imat]
    # Final list of synapses that can be potentiated
    IJ=Jh<pp.Jmax
    g=Jh[IJ]
    # Modify with stochastic ltp probability.
    RR=np.random.rand(g.size)<pp.pltp
    g=g+RR;
    Jh[IJ]=g;
    J[imat]=Jh;
    return sum(RR)

def depress_ff(pp,h,XI,J):
     # Perceptrons with field above depression threshold.
    hii=h>=pp.theta-pp.delta;
    # Logical matrix of all synapses that can be depressed ... 
    # If greater than minimal synaptic value
    imat=np.dot(XI,hii)
    Jh=J[imat];
    # Final list of synapses that can depress
    IJ=Jh>0
    g=Jh[IJ]
    # Modify with stochastic ltd probability.
    RR=np.random.rand(g.size)<pp.pltd
    g=g-RR;
    Jh[IJ]=g;
    J[imat]=Jh;
    return sum(RR);



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
    
        d=7200
        N=1000
        Jmax=2
        pobj=.5
        numit=5
        pltp=.01
        pltd=.01
        nofield=0
        theta=0
        delta=5
        pt=0
        

        def __init__(self):
            self.cat=[] #np.array([[.1, .8, .1], [.4, .3, .2], [.3, .3, .3]])

class netout:
    JJ=[];
    JJfb=[];
    def __init__(self,J,Jfb):
        self.JJ=J
        self.JJfb=Jfb
    
