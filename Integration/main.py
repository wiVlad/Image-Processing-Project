import multiprocessing
from queue import Queue

from GameEngine.GameApp import PongApp
from ImageProc.DetectMeanShift import DetectMeanShift
from ImageProc.test import test 
#from ImageProc.ImProc import maskHSV


# A thread that produces data
def producer(out_q):
    # DetectCamShift(out_q)q
    DetectMeanShift(out_q)
    # maskHSV(out_q)
    # test(out_q)

# A thread that consumes data


def consumer(in_q):
    while True:
        PongApp().run(in_q)


if __name__ == '__main__':  
        # Create the shared queue and launch bqoth q
    q = multiprocessing.Queue()
    t1 = multiprocessing.Process(target=consumer, args=(q,))
    t2 = multiprocessing.Process(target=producer, args=(q,))
    t1.start()
    t2.start()
