cd /data/home/kaijyun/Test/
mkdir /Data_out/
input_array=(AAC EAAC CKSAAP DPC DDE TPC BINARY GAAC EGAAC CKSAAGP GDPC GTPC AAINDEX ZSCALE BLOSUM62 NMBroto Moran Geary CTDC CTDT CTDD CTriad KSCTriad SOCNumber QSOrder PAAC APAAC KNNprotein KNNpeptide PSSM SSEC SSEB Disorder DisorderC DisorderB ASA TA)
for i in "${input_array[@]}";
 do
 python /data/home/kaijyun/Feature_Production/iFeature/iFeature.py --type ${i} --file /data/home/kaijyun/Test/Verticillium_longisporum_gca_001268165.vl2.denovo.v1.pep.all.fa.out.format_info.fasta --out /data/home/kaijyun/Test/Data_out/Verticillium_longisporum_gca_001268165.vl2.denovo.v1.pep.all.fa.out.format_info.fasta.${i}.txt
done