from __future__ import absolute_import, print_function
import numpy as np
import sys
import copy
import time
import os
import pdb
import amitgroup as ag
import pickle




def top_train(expi):

    """
    Call two types of network training. One which
    updates perceptrons of one class against the rest
    and then moves on to other classes. The second which
    updates all the perceptrons at each iteration (this
    is more natural). The first will probably be removed at somepoint.

    Parameters
    ----------

    The input is an `experiment' class that has the training
    and test data as submembers (.ddtr, .ddte) a list of lists.
    len(ddtr) is number of classes. For each class len(ddtr[c]) is
    number of training points. If exp.pp.numtrain_per_class>0 then
    exactly that number of training points per class are used.
    exp.pp.slant=1 - deslant the digits.
    exp.out = output file for some printouts.
    exp.pp = parameter class for network training, stochastic svm training, part training. (TODO break pp into sub parameter classes)

    Returns
    -------

    The function add a list of lists of network classes into the input class and doesn't return anything. List length is number of classes.
    For each class list length is number of perceptrons. Each perceptron
    synapse matrix is given by a network class.

    """
    expi.NO=[]
    if expi.pp.type==0:
        expi.NO.append(train_net(expi))
    else:
        expi.NO.append(all_at_one_top(expi))

    [C,e]=test_averages(expi)
    print("Test classification rate ", e)
    print("Finished Training")



def extract_feature_matrix(ddt,s,num=0):

    """

    Create one feature array from the features of a list of images.
    The feature for each image is flattened and added as a row of the array.

    Parameters
    ----------

    ddt - The list of images. s - a string denoting which level of features to use.
    (For now we only have V1)
    num=0 - number of images to extract if not all of them.

    Returns
    -------

    Returns the features array.

    """
  
    if (num==0):
        l=len(ddt)
    else:
        l=num
    if (s!='B'):
        print(s)
        numfeat=ddt[0].features[s].size
        MM=np.zeros((l,numfeat), dtype=np.ubyte)
        i=0
        for i in range(l):
            MM[i,:]=ddt[i].features[s].flatten()
        return MM
    else:
        numfeat=ddt[0].img.size;
        MM=np.zeros((l,numfeat), dtype=np.ubyte)
        i=0
        for i in range(l):
            MM[i,:]=ddt[i].img.flatten()> 20
        return MM

def read_data_b(expi,numclass):

    """

    Read training and test images  from path 's' and process them for features
    put result in expi.ddtr, expi.ddte.

    if expi.pp.numtrain_per_class>0 extract exactly that number of training
    examples for each class. Otherwise just take first exp.pp.numtrain exaples
    from training et.
    if expi.pp.slant=1 deslant the images.

    Parameters
    ----------

    s-path, expi-experiment, numclass- number of classes.

    Returns
    -------

    Nothing.

    """

    print('Hello')
    s=os.environ['HOME']+'/Desktop/Dropbox/'
    sstr=s+'/mnist_train'
    sste=s+'/mnist_test'
    expi.ddtr=[]
    expi.ddte=[]
    for i in range(numclass):
        bb=[]
        expi.ddtr.append(bb)
        cc=[]
        expi.ddte.append(cc)

    if (expi.pp.numtrain_per_class==0):
        for i in range(expi.pp.numtrain):
            tim=ag.io.load_imagep(sstr,i,True)
            tim.img=ag.io.process_im(tim.img, expi.pp.slant, expi.pp.DIM)
            feat=ag.features.bedges(np.double(tim.img),5,'box',expi.pp.spread)
            tim.features={'V1': feat}
            tr=tim.truth
            expi.ddtr[tr].append(tim)
    else:
        for c in range(numclass):
            i=0
            while len(expi.ddtr[c])<expi.pp.numtrain_per_class:
                if (ag.io.get_tr(sstr,i)==c):
                    tim=ag.io.load_imagep(sstr,i,True)
                    tim.img=ag.io.process_im(tim.img, expi.pp.slant, expi.pp.DIM)
                    feat=ag.features.bedges(np.double(tim.img),5,'box',expi.pp.spread)
                    tim.features={'V1': feat}
                    tr=tim.truth
                    expi.ddtr[tr].append(tim)
                i+=1

    for i in range(10000):
        tim=ag.io.load_imagep(sste,i,True)
        tim.img=ag.io.process_im(tim.img, expi.pp.slant, expi.pp.DIM)
        feat=ag.features.bedges(np.double(tim.img),5,'box',expi.pp.spread)
        tim.features={'V1': feat}
        tr=tim.truth
        expi.ddte[tr].append(tim)


def append_nets(NO,N1):

    """
    Append the perceptron lists of two net lists to create
    a net with more perceptrons.

    Parameters
    ----------
    Two input network lists. (This needs to be fixed.)
    """
   
    numclass=len(NO)

    N=[]
    for c in range(numclass):
        a=[]
        N.append(a)
    
    for c in range(numclass):
        N[c].extend(NO[c])
        N[c].extend(N1[c])
    return N



def nets_to_mat(NO,Jmid,numperc=0):

    """
    Convert a list of numclass lists of numperc networks 
    to one big 3d matrix. numfeatxnumpercxnumclass

    Parameters
    ----------
    NO-network list of lists. 
    Jmid - middle synaptic value.
    numperc - if not zero determines the number of networks
    to take for each class.

    Returns
    -------

    The array.

    """

    numclass=len(NO);
    if numperc==0:
        numperc=len(NO[0]);
    print(numperc)
    numfeat=NO[0][0].JJ.size;
    JJ=np.zeros((numfeat,numperc,numclass), dtype=np.int8);
    JJfb=np.zeros((numfeat,numperc,numclass), dtype=np.int8);
    for c in range(numclass):
       for p in range(numperc):
           NO[c][p].JJ.shape=(numfeat,);
           NO[c][p].JJfb.shape=(numfeat,)
           JJ[:,p,c]=np.double(NO[c][p].JJ)-Jmid;
           JJfb[:,p,c]=np.double(NO[c][p].JJfb)-Jmid
    return [JJ, JJfb]

def test_averages(expi, numtest=0):

    """

    Test a network using the mean weights for each class
    instead of the individual perceptrons. Make
    the big matrix of synaptic binary weights. Get its mean for each
    class and send to simple linear classifier.

    Parameters
    ----------

    expi - experiment class, expi.numperc (if not zero) - number of perceptrons to use
    numtest=0 - if not zero number of test examples per class to run.

    Returns
    -------

    Confusion matrix and error.

    """

    numclass=len(expi.NO);
    Jmid=np.ceil(expi.pp.Jmax/2)
    [JJ,JJfb]=nets_to_mat(expi.NO[0],Jmid,expi.pp.numperc);
    WW=np.mean(JJ,1);
    [CC, e]=test_by_weights(expi.ddte,expi.pp,WW,numtest)
    return CC, e


def test_by_weights(ddte,pp,WW,numtest=0):

    """
    Simple linear tester. For each class apply weight
    vector inner product with data and return highest value.

    Parameters
    ----------

    ddte - test data.
    WW - weight matrix.
    numtest - if non zero - number of test examples per class

    Returns
    -------
    Confusion matrix and error.

    """

    numclass=len(ddte)
    d=WW.shape[0]
    WW.shape=[d,numclass]
    CONF=np.zeros((numclass,numclass))
    Ntot=0
    for c in range(numclass):
        if numtest==0:
            N=len(ddte[c])
        else:
            N=numtest
        Ntot+=N
        H=np.zeros((N,numclass));
        H=np.dot(extract_feature_matrix(ddte[c],pp.feat,N),WW);
        i=np.argmax(H,1);
        for d in range(numclass):
            CONF[c,d]=np.double(np.sum(i==d))
        
    e=np.sum(np.diag(CONF))/Ntot
    print(e)
    return [CONF, e]

        
def test_net(expi, tr=False, numtest=0):

    """
    Test a network using the vote of the perceptrons


    Parameters
    ----------

    expi - experiment class. expi.pp.numperc - number of perceptrons to use if non-zero
    numtest=0 - if not zero number of test examples per class to run.

    Returns
    -------

    Confusion matrix and error.

    """


    Jmid=np.ceil(expi.pp.Jmax/2)
    print(Jmid)
    numclass=len(expi.ddte);    
    [JJ, JJfb]=nets_to_mat(expi.NO[0],Jmid,expi.pp.numperc);
    print(JJ[:,:,0].shape)
    CONF=np.zeros((numclass,numclass))
    Ntot=0
    ddt=expi.ddte
    if tr:
        ddt=expi.ddtr
    for c in range(numclass):
        if numtest==0:
            N=len(ddt[c])
        else:
            N=numtest
        Ntot+=N
        H=np.zeros((N,numclass));
        FF=extract_feature_matrix(ddt[c],expi.pp.feat,N)
        for d in range(numclass):
            temp=np.dot(FF,JJ[:,:,d])>expi.pp.theta
            H[:,d]=np.sum(temp,1)
        i=np.argmax(H,1);
        for d in range(numclass):
            CONF[c,d]=np.double(np.sum(i==d))

    print(np.sum(np.diag(CONF))/Ntot)
    print('Class rates')
    for d in range(numclass):

        print(CONF[d,d]/np.sum(CONF[d,:]))
    
    return CONF



                
def train_net(expi):

    """

    Train each class separately. So rerun over everything numclass times.
    This will probably be phased out.

    Paremters
    ---------

    expi - experiment.

    Returns
    -------

    Returns list of perceptrons (numperc) 

    """
    f = open(expi.pp.out,'w')
    numclass=len(expi.ddtr)
    CI=range(numclass)
    #np.random.shuffle(CI)
    NO=[None]*numclass
    for c in CI:
        f.write(str(CI[c])+'\n')
        NO[CI[c]]=ff_mult_top(f,expi.pp,expi.ddtr,CI[c],expi.pp.numperc, expi.pp.numtrain_per_class)

    return NO

def stack_data(expi):
    numclass=len(expi.ddtr)
    N=expi.pp.numtrain_per_class
    if N==0:
        N=len(expi.ddtr[0])
    # Get the full data matrix for class 0
    X=extract_feature_matrix(expi.ddtr[0],expi.pp.feat,N)
    Y=np.zeros((N,1), dtype=np.ubyte)
    # Stack up the data matrices for the other classes.
    for c in range(1,numclass):
        N=expi.pp.numtrain_per_class
        if N==0:
            N=len(expi.ddtr[c])
        print('Loading class ', c, N)
        X=np.vstack((X,extract_feature_matrix(expi.ddtr[c],expi.pp.feat,N)))
        Y=np.vstack((Y,c*np.ones((N,1))))
    return X,Y

def all_at_one_top(expi):

    """

    Train everything together. Each  data point triggers
    potentiatiation of perceptrons of its class and depression
    on perceptrons of ALL other classes.

    Parameters
    ----------

    expi - experiment. 
    expi.pp.numperc - number of perceptrons per class
    expi.pp.numtrain_per_class>0 - number of training data per class.

    Returns
    ------

    List of lists of perceptrons (one list for each class.)

    """
    # stack data of all classes in one array, with an accompanying label array
    print('Going to stack')
    [X,Y]=stack_data(expi)
    # Call the training routine
    numclass=len(expi.ddtr)
    NN=ff_all_at_one(expi.pp,X,Y,expi.pp.numperc,numclass)
    return NN
    
def ff_all_at_one(pp,X,Y,numperc,numclass):

    """

    Actually loop through a random ordering of the data
    and potentiate synapses to perceptron of same class 
    depress synapses to perceptrons of other classes and
    potentiate feedback synapses for same class from perceptron to features.

    Parameters
    ----------

    out - file name for some printouts
    pp - parameters of learning (pltp, pltd, deltaP, deltaD
    X - feature data
    Y - class labels
    numperc - number of perceptrons per class
    numclass - number of classes.

    Returns
    -------

    List of list of perceptrons.

    """

    #sys.stdout = open('out','w')
    Ntot=X.shape[0]
    numfeat=X.shape[1]
    print(Ntot, numfeat)

    # Synapses are positive and Jmid is the `middle'. Instead of being symmetric around 0.
    Jmid=np.ceil(pp.Jmax/2)
    # Feed forward synspases - initial value 1 -> 0.
    J=[]
    Jfb=[]
    for c in range(numclass):
        J.append(np.ones((numfeat,numperc), dtype=np.int8)*Jmid)
        # Feedback synapses
        Jfb.append(np.ones((numfeat,numperc), dtype=np.int8)*Jmid)
    # Iterate
    II=range(Ntot)
    h=np.zeros(numperc)
    rnumclass=range(numclass)
    rNtot=range(Ntot)
    for it in range(pp.numit):
        print('iteration ', it)
        # Random arrangement of examples. Stochastic gradient.
        np.random.shuffle(II)
        # Variables to keep track of changes
        up=0
        down=0
        # Loop over examples
        for i in rNtot:
            ii=II[i]
            # Booleanize the data.
            XI=X[ii,:]==1
            XIz=X[ii,:]==0
            # Prepare for matrix multiplication.
            
            # Field at each perceptron for this class.
            
            for c in rnumclass:
                h=(np.dot(X[ii,:],J[c]-Jmid)).T
                #h.shape=[1,numperc] 
                if Y[ii]==c:
                    # Update in up direction.
                    up+=potentiate_ff(pp,h,XI,J[c],Jmid)
                    Jfb[c]=modify_fb(pp,XI,XIz,Jfb[c],Jmid)
                else:
                    down+=depress_ff(pp,h,XI,J[c],Jmid)
                    
        # up+down
        
        #write('updown '+str(np.double(up)+np.double(down))+'\n')
        
    N=[]
    for c in range(numclass):
        NN=[]
        for p in range(numperc):
            NN.append(netout(J[c][:,p],Jfb[c][:,p]))
        N.append(NN)
            
    return N            



def modify_fb(pp,XI,XIz,Jfb,Jmid):

    """

    Potentiate or depress the feedback synapses.

    Parameters:
    ----------

    pp - learning parameters.
    XI - Which features are on.
    XIz - Which features are off.
    Jfb - array of synaptic values.
    Jmid = pp.Jmax/2

    Returns
    -------

    Returns update synaptic value array.

    """
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


    """

    If field is below pp.theta+pp.deltaP
    Potentiate feed forward synapses that have active features (XI)

    Parameters
    ----------
    pp - learning parameters - pp.pltp, pp.deltaP
    h - current field.
    XI - active features.
    J - synaptic array
    Jmid = pp.Jmax/2

    Returns
    -------

    Updates J but returns number of modifications.

    """
    
    # Perceptrons with field below potentiation threshold.
    hii=h<=pp.theta+pp.deltaP
    if (len(np.nonzero(hii)[0])==0):
        return 0

    # Logical matrix of all synapses that can be potentiated ... below potentiation threshold
    # and the feature is on. (Synapses with off features don't create a change.
    imat=np.outer(XI,hii)
    if (len(J.shape)==1):
        imat=imat.flatten()
        
    # Extract changeable synapses.
    Jh=J[imat]

    # If less than maximal synaptic value
    IJ=Jh<pp.Jmax
    g=Jh[IJ]
    # Modify with stochastic ltp probability.
    RR=(np.random.rand(g.size)<pp.pltp)
    g=g+np.double(RR)*pp.pinc
    Jh[IJ]=g
    r=len(np.nonzero(RR)[0])
    J[imat]=Jh
    return r

def depress_ff(pp,h,XI,J,Jmid):

    """

    If field is above pp.theta-pp.deltaD
    depress feed forward synapses that have active features (XI)

    Parameters
    ----------
    pp - learning parameters - pp.pltd, pp.deltaP
    h - current field.
    XI - active features.
    J - synaptic array
    Jmid = pp.Jmax/2

    Returns
    -------

    Updates J but returns number of modifications.

    """

     # Perceptrons with field above depression threshold.
    hii=h>=pp.theta-pp.deltaD;
    if (len(np.nonzero(hii)[0])==0):
        return 0
    # Logical matrix of all synapses that can be depressed ...above depression threshold
    # and the feature is on. (Synapses with off features don't create a change.)
    imat=np.outer(XI,hii)
    if (len(J.shape)==1):
        imat=imat.flatten()
   
    Jh=J[imat];
    # If greater than minimal synaptic value
    IJ=Jh>0
    g=Jh[IJ]
    # Modify with stochastic ltd probability.
    RR=(np.random.rand(g.size)<pp.pltd)
    Jh[IJ]=g-RR*pp.pinc
    r=len(np.nonzero(RR)[0])
    J[imat]=Jh
    return r

def rearrange(dd,c,s,numtrain=0):

    """

    Arrange data matrix of class c at top of array
    and then data of all other classes.

    Parameters
    ----------

    dd - list of datas 
    c - class to put on top.
    numtrain - number per class to use.

    Returns:
    -------

    Returns the matrix.

    """

    XY=[]   
    ic=range(len(dd))
    ic.remove(c)
    n=len(dd[c])
    if numtrain>0:
        n=min(numtrain,len(dd[c]))

    XY.append(extract_feature_matrix(dd[c],s,n))


    N=XY[0].shape[0]

    for ii in ic:
        n=len(dd[ii])
        if numtrain>0:
            n=min(numtrain,len(dd[ii]))
        XY[0]=np.vstack((XY[0],extract_feature_matrix(dd[ii],s,n)))

        
    Ntot=XY[0].shape[0]
    Nbgd=Ntot-N
    XY.append(np.vstack((np.ones((N,1), dtype=np.ubyte),np.zeros((Nbgd,1), dtype=np.int8))))

    return XY


# Train the network for each class.
def ff_mult_top(f,pp,ddtr,c,numperc, numtrain=0):
    if numtrain==0:
        numtrain=len(ddtr[c])
    # Rearrange data for this class with class at top of array and all the rest after.
    XY=rearrange(ddtr,c, pp.feat, numtrain)
    # Train class against the rest perceptron/s
    NO=ff_mult(pp,XY, numperc,f)    
    return NO

def ff_mult(pp,XY,numperc,f):

    # Features
    X=XY[0]
    # Labels 1/0
    Y=XY[1]
    Ntot=X.shape[0]
    numfeat=X.shape[1]


    # Simple learing rule or field learning rule.
    # Synapses are positive and Jmid is the `middle'. Instead of being symmetric around 0.
    Jmid=np.ceil(pp.Jmax/2)
    # Feed forward synspases - initial value 1 -> 0.
    J=np.ones((numfeat,numperc))*Jmid
    # Feedback synapses
    Jfb=np.ones((numfeat,numperc))*Jmid
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
        f.write('updown '+str(np.double(up)+np.double(down))+'\n')
        if up+down==0:
            break
    N=[]
    for p in range(numperc):
        N.append(netout(J[:,p],Jfb[:,p]))
    return N


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
        deltaP=None
        deltaD=None
        pt=None
        sh=None
        min_edges=None
        part_size=None
        spread=None
        slant=None
        type=None
        numperc=None
        numtrain=None
        numtrain_per_class=None
        DIM=None
        out=None
        numparts=None

        def __init__(self):
            self.d=7200
            self.N=1000
            self.Jmax=2
            self.pobj=.5
            self.numit=5
            self.pltp=.01
            self.zero_prior=.1
            self.two_prior=.1
            self.pltd=.01
            self.stoch=1
            self.nofield=0
            self.theta=0
            self.deltaP=5.
            self.deltaD=5.
            self.pt=0
            self.showing=100000
            self.min_edges=40
            self.part_size=7
            self.pinc=1
            self.spread=2
            self.special_class=-1
            self.reduction_factor=.9
            self.numperc=1
            self.numtrain=0
            self.numtrain_per_class=100
            self.type=1
            self.slant=1
            self.DIM=0
            self.feat='V1'
            self.out='out'
            self.numparts=0
            
        def write(self,f):
            pickle.dump(self,f)

        def write(self,s):
            f=open(s,'w')
            pickle.dump(self,f)
            f.close()

        def read(self,f):
            self=pickle.load(f)
            return self

        def read(self,s):
            f=open(s,'r')
            self=pickle.load(f)
            f.close()
            return self
            
def compress_nets(NN):

    for Nc in NN:
        for P in Nc:
            P.JJ=np.ubyte(P.JJ)

    
class experiment:
    ddtr=[]
    ddte=[]
    pp=[]    
    NO=[]
    def __init__(self):
        self.pp=pars()
        
       
    def ecopy(self,ine):
        self.ddtr=ine.ddtr
        self.ddte=ine.ddte
        self.pp=copy.copy(ine.pp)

    def write_pars(self,s):
        self.pp.write(s)

    def read_pars(self,s):
        pr=pars()
        pr=pr.read(s)
        self.pp=pr
    
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
        
    


    
    


