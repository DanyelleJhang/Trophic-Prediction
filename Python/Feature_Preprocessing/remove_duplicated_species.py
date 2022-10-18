#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import os,argparse,sys
parser = argparse.ArgumentParser(
	description= "Code to:\n" +
	"ML Classification")

req_group = parser.add_argument_group(title='REQUIRED INPUT')
req_group.add_argument('-file',required=True)

if len(sys.argv) == 1:
	parser.print_help()
	sys.exit(0)
args = parser.parse_args()

file = args.file

r = pd.read_csv(file,sep="\t")
r_2 = r.drop_duplicates(subset=['Species']).reset_index(drop=True)
r_2.to_csv(file+".drop_duplicated.txt",sep="\t",header=True,index=False)

