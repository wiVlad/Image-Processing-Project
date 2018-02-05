import multiprocessing
from queue import Queue

from ImageProc.ImProc import maskHSV
from ImageProc.camShift import CamShift
from GameEngine.GameApp import PongApp 


# A thread that produces data
def producer(out_q):
	CamShift(out_q)
	#maskHSV(out_q)

# A thread that consumes data
def consumer(in_q):
    while True:
    	PongApp().run(in_q)

if __name__ == '__main__':
	# Create the shared queue and launch both threads
	q = multiprocessing.Queue()
	t1 = multiprocessing.Process(target=consumer, args=(q,))
	t2 = multiprocessing.Process(target=producer, args=(q,))
	t1.start()
	t2.start()