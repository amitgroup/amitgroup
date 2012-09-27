import train_net as tn
import Kthread

def bgd(f,expi,joi):

    thread=Kthread.KThread(None,f,None,(expi,))

    thread.start()
    if joi:
        thread.join()

    return thread

    
