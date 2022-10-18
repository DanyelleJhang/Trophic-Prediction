# Data Collection

- Require Package

```textile
pandas
numpy
biopython
```

#### 1.Download EmsemblFungi speciest from : [Species List](https://fungi.ensembl.org/species.html)

- 1014 species in total

- output file name: Species.csv

| Name           | Classification | Taxon ID | Assembly | Accession       | ... |
| -------------- | -------------- | -------- | -------- | --------------- | --- |
| Absidia glauca | Mucoromycotina | 4829     | AG_v1    | GCA_900079185.1 | ... |
| ...            | ...            | ...      | ...      | ...             | ... |

#### 2.Retrieve Taxon ID to txt file

- Due to species' names in EmsemblFungi are not consistent to FUNGuild database which naming rule is according to NCBI, Taxon ID, howeverm remains identical number among three database

```bash
################
# Bash command #
################
taxon_col_num=$(head -1 Species.csv | tr ',' '\n' | cat -n | grep "Taxon ID" | cut -f1)
# extract Taxon ID content
sed '/Taxon ID/d' Species.csv | tr ',' '\t' | cut -f $taxon_col_numm} > taxids.txt
```

- output file name: taxids.txt

#### 3.Download package taxonkit

- The package has been developed and publised recently

- Shen, W., & Ren, H. (2021). TaxonKit: A practical and efficient NCBI taxonomy toolkit. *Journal of Genetics and Genomics*.

- [TaxonKit - NCBI Taxonomy Toolkit](https://bioinf.shenwei.me/taxonkit/)

```bash
################
# Bash command #
################
conda install -c bioconda taxonkit
wget -c ftp://ftp.ncbi.nih.gov/pub/taxonomy/taxdump.tar.gz 
tar -zxvf taxdump.tar.gz

mkdir -p $HOME/.taxonkit
cp names.dmp nodes.dmp delnodes.dmp merged.dmp $HOME/.taxonk 
```

#### 4.Run taxonkit to find species name and which taxonomy classification

```bash
################
# Bash command #
################
# Find completely lineage from NCBI
taxonkit lineage taxids.txt > lineage.txt
# Reformat data comforming to FUNGuild.py 
taxonkit reformat lineage.txt | tee lineage.reformat.txt | cut -f 1,3 > lineage.FUNGuild.txt
sed -i -e "s+;+\t+g" lineage.FUNGuild.txt
# the first column represents taxonoy id
# but FUNGuild.py requires first column as OTU_ID
# it wouldnt make bias in the future
sed -i -e "1i OTU_ID\tKingdom\tPhylum\tClass\tOrder\tFamily\tGenus\tSpecies" lineage.FUNGuild.txt
```

- output file name: lineage.FUNGuild.txt

#### 5.Drop duplicated species

- However, there are different taxon id but annotated as same species; with regard to FUNGuild searching rule based on species name, i decide to drop ot duplicated species name

- 637 species in total

```python
#!/usr/bin/env python
# It run by """"""PYTHON envroment""""""
# bash script such as 
# awk 'a[$8]++'
# sort -rk8 lineage.FUNGuild.txt | awk '!seen[$8]++'
# the bash command doesnt work well
# but why ?
python ./Feature_Preprocessing/remove_duplicated_species.py -file lineage.FUNGuild.txt
```

- output file name: lineage.FUNGuild.drop_duplicated.txt

#### 6.Run FUNGuild.py to find species' function

```bash
################
# Bash command #
################
python ./Feature_Preprocessing/FUNGuild/FUNGuild.py guild -taxa ./lineage.FUNGuild.drop_duplicated.txt
```

- output file name:lineage.FUNGuild.drop_duplicated.guilds.txt

#### 7.Download protein sequence from EnsemblFungi

```bash
################
# Bash command #
################
# mkdir ./Protein_seq
rsync -av rsync://ftp.ebi.ac.uk/ensemblgenomes/pub/release-51/fungi/fasta/*/pep/*.fa.gz ./Protein_seq
rsync -av rsync://ftp.ebi.ac.uk/ensemblgenomes/pub/release-51/fungi/fasta/*/*/pep/*.fa.gz ./Protein_seq
```

- All protein sequence will be sotred at ./pep_data

#### 8.Find FUNGuild needed protein data

- the file name wasnt follow by one rule, so it will be filter by python first and then check missing information.

- 57 species wasnt found, so missing protein file will be mannually lood after by website information https://fungi.ensembl.org/info/data/ftp/index.html 

- All 637 protein files relative to FUNGuild will be move to **./FUNGuild_peptide/Peptide_data/**

- All species and which protein file FUNGuild needed is found and relaitve information will be integert to **./FUNGuild_peptide/FUNGuild_with_pep_filename_merge.csv**

#### 9.Find protein domain by Pfam

- http://pfam.xfam.org/

- Pfam has been installed at server

- summarize_pfam.pl was written by HCL lab

- create shell script: **transform_format.sh**

```shell
# /usr/bin/env bash
#########################
## transform_format.sh ##
#########################
for f in *.fa; 
do
    if [ -f "${f}.out" ]; then
        echo "${f}.out exists."
    else 
        echo "${f}"
        hmmscan --cpu 3 /data/home/HCL/db/pfam/Pfam-A.hmm $f > $f.out
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
```

#### 9.Submit job to server and meaning of output data

```bash
################
# Bash command #
################
nohup bash transform_format.sh &
```

- output data example :

| Column 1      | Column 2                                                         |
| ------------- | ---------------------------------------------------------------- |
| Mycgr3P87220  | CoaE:3-193                                                       |
| Mycgr3P108294 | Spt20:58-304 Spt20:423-487 Spt20:697-753 Spt20:771-826           |
| Mycgr3P93972  | FRQ:25-940                                                       |
| Mycgr3P42345  | Peptidase_M1_N:25-69 Peptidase_M1_N:173-241 Peptidase_M1:371-475 |
| ...           | ...                                                              |

- Take Mycgr3P42345 as example
  
  - Mycgr3P42345 is ***Protein ID*** named by Ensembl database
  
  - Peptidase_M1_N, Peptidase_M1_N and Peptidase_M1 represent different ***domain family***
  
  - 25-69, 173-241 and 371-475 indicate ***different domain families' location***
  
  - 25 denotes ***25th amino-acid*** as Start site of peptide, 69 denotes ***69th amino-acid*** as End site of peptide

#### 10.Extract domain peptide squence

```bash
nohup bash summarize_pfam_table.sh &> summarize_pfam_table_log.txt &
```

- remove no_hit domain
- save missing data file name

```shell
#!/bin/shell
## remember to revise direcory ##
defult_path='/mnt/c/Users/Danyelle/Jupyter_code/2021_12_3/'
protein_dir='Protein_data/'
##
domain_dir='_Peptide_domain/'
seq_dir='_Protein_seq/'
##
python_code='1_summarize_pfam_table_3.py'

# search domain information as priority
# not protein full length sequence
# there will cause tremendous negative effect

cd ${defult_path}${protein_dir}${domain_dir}

string="hello-world.out.format"
suffix=".fa.out.format"

echo "${new_string}"
for f in *.fa.out.format;
do
    {
        file_name_prefix=${f%"$suffix"}
        ## Try 
        echo "${file_name_prefix}"
        python ${defult_path}${python_code} \
    -protein_domain ${defult_path}${protein_dir}${domain_dir}${file_name_prefix}.fa.out.format \
    -protein_full_len ${defult_path}${protein_dir}${seq_dir}${file_name_prefix}.fa \
    -output_format pandas_table \
    -drop_no_hit True
    } || { ## Catch 
        echo "${f}.fa.out.format" >> ${defult_path}summarize_pfam_table_MISS.txt
        set -e ## error but continued; keep diving in ''set --help''
    }
done
```

##### - Pandas Table

- output data example :

| domain_name   | domain_family | domain_location | domain_seq                      |
| ------------- | ------------- | --------------- | ------------------------------- |
| Mycgr3P87220  | CoaE          | 3-193           | LLGLTGSIATGKSTVSNILSSPPYN...... |
| Mycgr3P108294 | Spt20         | 58-304          | LKKFAGRPPSLVVHLHQNYFKFD.......  |
| Mycgr3P108294 | Spt20         | 423-487         | MVAQQEHSRKLMNQRQQLAR.......     |
| Mycgr3P91042  | no_hit        | 1               | no_hit                          |

##### - Fasta-like Format

- output data example :

```textile
>CRJ87205||no_hit||1
no_hit
>CRJ87118||AA_permease_2||104-119
ISSTAMNSSGGMGGGTL
>CRJ87118||AA_permease_2||187-272
FYTPIPALMLNAALTTAYILVGEFGTLITFYGVAGYTFYFLTVLGLIILRVREPTLERPY
RTWITTPIIFCCVSLFLLSRAVFAQPL
>CRJ86435||adh_short_C2||30-276
GTICSAQTRALVRLGANACIIGRNVEKTETVAKDIATARSGAKVIGIGACDVRNPKSLQD
AADRCVKELGAIDFVIAGAAGNFVAPISGMSSNAFKAVMDIDVLGTFNTIKATVPYLVES
AKRNPNPSTNGLTGGRIMFVSATFHYTGMPLQAHVSAAKAAVDSLMASVALEYGPYGITS
NVVAPGPIKDTEGMQRLSSSSVDMKAAEAAIPLGRWGLVRDIADSTVYLFSDAGSLVNGQ
VIPVDGGA
```
