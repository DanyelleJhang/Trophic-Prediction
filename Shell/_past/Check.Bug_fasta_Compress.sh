#!/bin/bash
#壓縮出問題，要重寫，跑小的試看看

cd /data/home/kaijyun/Feature_Production/Data_input/
for f in *.fasta;  
do  
  echo "${f}";
  tar -zcvf ${f}.tar.gz /data/home/kaijyun/Feature_Production/Data_input/;
  rm ${f};
done;