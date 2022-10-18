library(FactoMineR)
library(tidyverse)
library(dendextend)
library(readr)
suppressPackageStartupMessages(library("argparse"))

parser <- ArgumentParser()
parser$add_argument("--Data_Path")
parser$add_argument("--Data_Name")
parser$add_argument("--Sampling", default="True",type="character")
parser$add_argument("--balance_proportion", default=1.0, type="double")
parser$add_argument("--Save_Path")
parser$add_argument("--Sigle_Label_Name")
parser$add_argument("--Tag")

# get command line options, if help option encountered print help and exit,
# otherwise if options not found on command line then set defaults, 
args <- parser$parse_args()


data_path<-args$Data_Path
data_name<-args$Data_Name
sampling<-args$Sampling
n_proportion<-args$balance_proportion
save_path<-args$Save_Path
sigleLabelName<-args$Sigle_Label_Name
Tag<-args$Tag


noLabel <- function(x) {
  if (stats::is.leaf(x)) {
    attr(x, "label") <- NULL }
  return(x)
}




feature_selected_data <- read_delim(paste0(data_path,data_name), delim = "\t", escape_double = FALSE,trim_ws = TRUE)

names(feature_selected_data)[names(feature_selected_data) == sigleLabelName] <- "single_y"
feature_selected_data <- subset(feature_selected_data, select=-c(genome_file_name))
genome_sample_size<-round(min(table(feature_selected_data$single_y))*n_proportion)

if (sampling == "True"){
  feature_selected_data.sampling<- feature_selected_data %>% 
    group_by(single_y) %>% 
    sample_n(genome_sample_size)
}else{
  feature_selected_data.sampling<- feature_selected_data
}


feature_selected_data.sampling.the_bars<-feature_selected_data.sampling$single_y
feature_selected_data.sampling<-subset(feature_selected_data.sampling,select = -c(single_y))
feature_selected_data.sampling.PCA.ind.contrib<-PCA(feature_selected_data.sampling, scale.unit = TRUE, ncp = 2, graph = FALSE)$ind$contrib



### dist
# "euclidean", "maximum", "manhattan", "canberra", "binary" or "minkowski"
### hclust
# "ward.D", "ward.D2", "single", "complete", 
# "average" (= UPGMA), "mcquitty" (= WPGMA), 
# "median" (= WPGMC) or "centroid" (= UPGMC).

dend_mtcars.sampling <- feature_selected_data.sampling%>% 
  dist(method = "euclidean") %>% 
  hclust(method = "average") %>% 
  as.dendrogram %>% 
  highlight_branches %>%
  set("branches_k_color", k=2) 

dend_mtcars.sampling.PCA.ind.contrib <- feature_selected_data.sampling.PCA.ind.contrib%>% 
  dist(method = "euclidean") %>% 
  hclust(method = "average") %>% 
  as.dendrogram %>% 
  highlight_branches %>%
  set("branches_k_color", k=2) 






jpeg(paste0(save_path,data_name,"_HierarchicalClusteringSampling_genome",as.character(n_proportion),"--Sample--",sampling,"--",Tag,".jpg"), width = 1800, height = 1200)
par(mar = c(10,2,1,1))
plot(stats::dendrapply(dend_mtcars.sampling, noLabel), cex.axis = 3,main=data_name,cex.main=2)
the_bars <- ifelse(feature_selected_data.sampling.the_bars, "grey", "blue")
colored_bars(colors = the_bars, dend = dend_mtcars.sampling, rowLabels = sigleLabelName,cex.rowLabels=3)
dev.off() 

jpeg(paste0(save_path,data_name,"_PCAHierarchicalClusteringSampling_genome",as.character(n_proportion),"--Sample--",sampling,"--",Tag,".jpg"), width = 1800, height = 1200)
par(mar = c(10,2,1,1))
plot(stats::dendrapply(dend_mtcars.sampling.PCA.ind.contrib, noLabel), cex.axis = 3,main=data_name,cex.main=2)
the_bars <- ifelse(feature_selected_data.sampling.the_bars, "grey", "blue")
colored_bars(colors = the_bars, dend = dend_mtcars.sampling.PCA.ind.contrib, rowLabels = sigleLabelName,cex.rowLabels=3)
dev.off() 

  
