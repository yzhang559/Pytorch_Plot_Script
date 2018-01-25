#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""


@author: yzhang559
"""

import matplotlib.pyplot as plt
import numpy as np
import argparse

def plot_loss(args):
	# Initialize
	train_iter=np.linspace(1,args.n_epoch,args.n_epoch)
	iters=args.data_size/args.batch_size
	loss=[]
	train_loss=[[] for i in xrange(iters)]
	lines=0
	
	
	
	# Input file
	f = open(args.filename, 'r')

	for line in f:
	 if('Loss: 'in line):	   
		temp= line.split("Loss: ",1)
		temp2=float(temp[1])
		loss.append(temp2)
		lines=lines+1

	
	for m in xrange(iters):
	 for n in xrange(lines):	
		if (n+1) % iters == m:
			train_loss[m].append(loss[n])	



	train_mean=np.zeros(shape=(iters,lines/iters))
	
	# Calculate mean and std
	for k in xrange(iters):
		a=np.asarray(train_loss[k]).reshape(-1,lines/iters)
		train_mean[k]=a
	
	print train_mean.shape
	mean=np.mean(train_mean,axis=0)
	std=np.std(train_mean,axis=0)
	plus_std=mean+std
	m_std=mean-std

		   
	ftrain_loss = plt.figure()
	plt.plot(train_iter,train_loss[0],train_iter,train_loss[1],train_iter,train_loss[2])
	plt.plot(train_iter,mean,'r-',train_iter,plus_std,'black',train_iter,m_std,'black')
	
	plt.ylim(0,1.5e6)
	ftrain_loss.suptitle('Train_loss vs Train_iteration', fontsize=14)
	plt.xlabel('iteration', fontsize=12)
	plt.ylabel('loss', fontsize=12)
	#plt.show()
	ftrain_loss.savefig("{}.jpg".format(args.filename),format='jpg')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    
    parser.add_argument('--filename', nargs='?', type=str, 
			default='', help='input file')

    parser.add_argument('--figpath', nargs='?', type=str, 
			default='', help='save path')

    parser.add_argument('--data_size', nargs='?', type=int, default=1, 
                        help='Data Size')

    parser.add_argument('--batch_size', nargs='?', type=int, default=1, 
                        help='Batch Size')

    parser.add_argument('--n_epoch', nargs='?', type=int, default=100, 
                        help='# of the epsochs')

    args = parser.parse_args()
    plot_loss(args)



