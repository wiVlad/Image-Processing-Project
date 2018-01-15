
from queue import Queue
from threading import Thread
import random
import time
# A thread that produces data
def producer(out_q):
    while True:
        # Produce some data
        x = random.random()*10000
        time.sleep(1)
        out_q.put(x)

# A thread that consumes data
def consumer(in_q):
    while True:
        # Get some data
        data = in_q.get()
        # Process the data
        print("Here's the number: "+str(data))

# Create the shared queue and launch both threads
q = Queue()
t1 = Thread(target=consumer, args=(q,))
t2 = Thread(target=producer, args=(q,))
t1.start()
t2.start()