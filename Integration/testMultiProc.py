import multiprocessing
import random
import time

def spawn():
	print('Spawned!')

def reciever(i, conn):
	while(True):
		got = conn.recv()
		print("Hey, I'm {} and I got {}!".format(i, got))
		#sleep(int(random.random()*5))

def sender(i, conn):
	while(True):
		conn.send(random.random()*100)
		#print("Hey, I'm {} and I got {}!".format(i, got))
		sleep(int(random.random()*5))

if __name__ == '__main__':
	parent_conn, child_conn = Pipe()


	for i in range(50):
		p1 = multiprocessing.Process(target=reciever, args=(1, child_conn))
		p2 = multiprocessing.Process(target=sender, args=(2,parent_conn))
		p1.start()
		p2.start()
		p.join()