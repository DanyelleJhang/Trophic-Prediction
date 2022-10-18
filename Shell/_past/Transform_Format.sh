#!/bin/bash
cd /data/home/kaijyun/Feature_Preprocessing/Peptide_data/Protein_seq/
for f in *.fa; 
do
	if [ -f "${f}.out" ]; then
		echo "${f}.out exists."
	else 
		echo "${f}"
		hmmscan --cpu 4 /data/home/HCL/db/pfam/Pfam-A.hmm $f > $f.out
	fi 
done


for o in *.out;
do
	if [ -f "${o}.format" ]; then
		echo "${o}.format exists."
	else 
		echo "${o}"
		perl /data/home/HCL/scripts/summarize_pfam.pl $o > $o.format
	fi 
done

mv *.out /data/home/kaijyun/Feature_Preprocessing/Peptide_data/Pfm_summary/
mv *.format /data/home/kaijyun/Feature_Preprocessing/Peptide_data/Peptide_domain/
