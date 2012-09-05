import train_net as tn

pp=tn.pars()
s='/home/amit/Desktop/Dropbox/rawdata1/tr'
ddtr=tn.read_data(s,10,7200)
s='/home/amit/Desktop/Dropbox/rawdata1/te'
ddte=tn.read_data(s,10,7200)

NO=tn.train_net(ddtr,pp,1,100)
CONF=tn.test_net(ddte,pp,NO)
