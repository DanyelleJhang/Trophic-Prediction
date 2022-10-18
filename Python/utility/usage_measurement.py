#!/usr/bin/env python
# coding: utf-8
import psutil
from time import time, strftime, gmtime

def hardware_usage(func):
# This function shows the execution time of 
# the function object passed
	def wrap_func(*args, **kwargs):
		m1= psutil.virtual_memory().used
		t1 = time()
		result = func(*args, **kwargs)        
		m2= psutil.virtual_memory().used
		c_count = psutil.cpu_count()
		c_percent = psutil.cpu_percent(interval=1)
		m = round(abs((m1-m2)),3)
		print("-----------------------------------------------------")
		print("----------------  Time and Memory  ------------------")
		print(f'Function: {func.__name__!r}')
		print(f'takes Virsualal Memory in: ',str(round(m / (1024.0 ** 3),3))," GB usage")
		t2 = time()
		print(f'executed in: {strftime("%H:%M:%S", gmtime(t2-t1))} seconds')
		return result
	return wrap_func
