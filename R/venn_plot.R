library(ggvenn)
path <- "C:/Users/fabia/Local_Work/Data/Feature_Selection_Production/"

LabelDiscrimination <- colnames(read.delim(paste0(path,"Genome_Combination_Count_Table.DomainList100.SimpleSelection.1637.txt.LabelDiscrimination.dropUnknown.selectedBy.Saprotroph.txt")))
LabelFeatureCorreltion <- colnames(read.delim(paste0(path,"Genome_Combination_Count_Table.DomainList100.SimpleSelection.1637.txt.LabelFeatureCorreltion.selectedBy.Saprotroph.txt")))
LabelInformative <- colnames(read.delim(paste0(path,"Genome_Combination_Count_Table.DomainList100.SimpleSelection.1637.txt.LabelInformative.dropUnknown.selectedBy.Saprotroph.txt")))
LabelPermutation <- colnames(read.delim(paste0(path,"Genome_Combination_Count_Table.DomainList100.SimpleSelection.1637.txt.LabelPermutation.dropUnknown.selectedBy.Saprotroph.txt")))

remove_vector_element<-function(x,v){
  x <- x[! x %in% v]
  return(x)
}


LabelDiscrimination<-remove_vector_element(LabelDiscrimination,c("genome_file_name"))
LabelFeatureCorreltion<-remove_vector_element(LabelFeatureCorreltion,c("genome_file_name"))
LabelInformative<-remove_vector_element(LabelInformative,c("genome_file_name"))
LabelPermutation<-remove_vector_element(LabelPermutation,c("genome_file_name"))



venn <- list(Discrimination = LabelDiscrimination,
             Correltion = LabelFeatureCorreltion,
             Informative = LabelInformative,
             Permutation = LabelPermutation
             )

ggvenn(venn, c("Discrimination","Correltion","Informative","Permutation"))
