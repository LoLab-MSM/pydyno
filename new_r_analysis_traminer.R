library(TraMineR)
library(cluster)
library(ggplot2)
library(Hmisc)
library(erer)
library(colorspace)
library(dendextend)
setwd('/home/oscar/tropical_project_new/data_frames5000')

#list of the dataframes (species) that are dominant under all different parameter values
df_list = list.files('/home/oscar/tropical_project_new/data_frames5000')


#Here we read the species that have dynamics where there are more than 1 dominant monomial at any time point.
for (i in 1:length(df_list)){
  sp = read.csv(df_list[i], header=T, row.names=1)
  print (sum(sp == 0))
  if ((sum(sp == 0)) < nrow(sp)*ncol(sp)){
    sp.seq = seqdef(sp)
    name1 = paste('/home/oscar/tropical_project_new/df_results/', unlist(strsplit(df_list[i],"\\."))[1], 'sdp','.png',sep='')
    png(file=name1)
    seqfplot(sp.seq, withlegend = T, border = NA, title = "Sequence Frequency plot")
    dev.off()
    name2 = paste('/home/oscar/tropical_project_new/df_results/', unlist(strsplit(df_list[i],"\\."))[1], 'dp','.png',sep='')
    png(file=name2)
    seqdplot(sp.seq, withlegend = T, border = NA, title = "State distribution plot")
    dev.off()
    name3 = paste('/home/oscar/tropical_project_new/df_results/', unlist(strsplit(df_list[i],"\\."))[1], 'hp','.png',sep='')
    png(file=name3)
    seqHtplot(sp.seq, withlegend = T, title = "Entropy index")
    dev.off()
  }
}

####################################
# CLUSTER ANALYSIS
###################################

#Load species dataframe and generate traminar sequence
sp1 = read.csv(df_list[1], check.names = FALSE, header=T, row.names=1)
sp1.seq = seqdef(sp1)
#seqfplot(sp.seq, withlegend = T, border = NA, title = "Sequence Frequency plot")
#clustering

sp1.lcp = seqdist(sp1.seq,method="OM", sm="CONSTANT")
clusterward1 <- agnes(sp1.lcp, diss = TRUE, method = "ward")
plot(clusterward1, which.plots = 2)
cluster1 <- stats::cutree(clusterward1, k = 3)
cluster1.fac <- factor(cluster1, labels = c("Type 1", "Type 2", "Type 3"))

dend=as.dendrogram(clusterward1)
dend=color_branches(dend,k=3)
labels_colors(dend) = rainbow_hcl(3)[sort_levels_values(
  as.numeric(cluster1.fac)[order.dendrogram(dend)])]
labels(dend) <- paste(as.character(cluster1.fac)[order.dendrogram(dend)],
                      "(",labels(dend),")", sep = "")
dend <- set(dend, "labels_cex", 0.5)

jpeg('/home/oscar/Desktop/cytoc_clusters.jpeg', width=1000, height=700, quality=100)
plot(dend, 
     main = "Clustered CytoC tropical signatures", 
     horiz =  FALSE,  nodePar = list(cex = .007))
legend("topleft", legend = c("Type 1", "Type 2", "Type 3"), fill = rainbow_hcl(3), cex=1.5)
dev.off()

pars_sp1_type1 = sp1[cluster1.fac=="Type 1",]
pars_sp1_type2 = sp1[cluster1.fac=="Type 2",]
pars_sp1_type3 = sp1[cluster1.fac=="Type 3",]

pars_sp1_type1[sample(nrow(pars_sp1_type1),1),]
pars_sp1_type2[sample(nrow(pars_sp1_type2),1),]
pars_sp1_type3[sample(nrow(pars_sp1_type3),1),]

sp1_monomials = c('__s19*__s60*pore_transport_complex_BakA_4_CytoCM_kf', 
                  '__s19*__s64*pore_transport_complex_BaxA_4_CytoCM_kf',
                  '__s62*pore_transport_complex_BakA_4_CytoCM_kr', 
                  '__s69*pore_transport_complex_BaxA_4_CytoCM_kr')

jpeg('/home/oscar/tropical_project_new/clusters/cytoc.jpg', width=1000, height=700)
seqfplot(sp1.seq, group = cluster1.fac, border=NA, title='CytoC clustering', xlab='time (min)', ltext=sp1_monomials, cex.legend=1.5, cex.plot=1.5)
dev.off()

seqIplot(sp1.seq, group = cluster1.fac, sortv = "from.start", title='CytoC clustering')

jpeg('/home/oscar/tropical_project_new/clusters/cytoc2.jpg', width=1000, height=700, quality=100 )
seqrplot(sp1.seq, dist.matrix = sp1.lcp , group = cluster1.fac, border = NA, title='CytoC clustering', 
         xlab='time (min)', cex.lab=1.8, ltext=sp1_monomials, cex.legend=1.6, cex.plot=1.8, cex.main=1.8)
dev.off()

###############
sp2 = read.csv(df_list[2], check.names = FALSE, header=T, row.names=1)
sp2.seq = seqdef(sp2)
#seqfplot(sp.seq, withlegend = T, border = NA, title = "Sequence Frequency plot")
#clustering

sp2.lcp = seqdist(sp2.seq,method="OM", sm="TRATE")
clusterward2 <- agnes(sp2.lcp, diss = TRUE, method = "ward")
plot(clusterward2, which.plots = 2)
cluster2 <- cutree(clusterward2, k = 3)
cluster2.fac <- factor(cluster2, labels = c("Type 1", "Type 2", "Type 3"))
table(cluster2)

pars_sp2_type1 = sp2[cluster2.fac=="Type 1",]
pars_sp2_type2 = sp2[cluster2.fac=="Type 2",]
pars_sp2_type3 = sp2[cluster2.fac=="Type 3",]

pars_sp2_type1[sample(nrow(pars_sp2_type1),1),]
pars_sp2_type2[sample(nrow(pars_sp2_type2),1),]
pars_sp2_type3[sample(nrow(pars_sp2_type3),1),]

sp2_monomials = c('__s20*__s60*pore_transport_complex_BakA_4_SmacM_kf', 
                  '__s20*__s64*pore_transport_complex_BaxA_4_SmacM_kf', 
                  '__s63*pore_transport_complex_BakA_4_SmacM_kr', 
                  '__s70*pore_transport_complex_BaxA_4_SmacM_kr')

seqfplot(sp2.seq, group = cluster2.fac, border=NA)
seqIplot(sp2.seq, group = cluster2.fac, sortv = "from.start")

jpeg('/home/oscar/tropical_project_new/clusters/smac2.jpg', width=1000, height=700, quality=100)
seqrplot(sp2.seq, dist.matrix = sp2.lcp , group = cluster2.fac, border = NA, title='Smac clustering', 
         xlab='time (min)', cex.lab=1.8, ltext=sp2_monomials, cex.legend=1.6, cex.plot=1.8, cex.main=1.8)
dev.off()


#############
sp5 = read.csv(df_list[5], check.names = FALSE, header=T, row.names=1)
sp5.seq = seqdef(sp5)
#seqfplot(sp.seq, withlegend = T, border = NA, title = "Sequence Frequency plot")
#clustering

sp5.lcp = seqdist(sp5.seq,method="OM", sm="CONSTANT")
clusterward5 <- agnes(sp5.lcp, diss = TRUE, method = "ward")
plot(clusterward5, which.plots = 2)
cluster5 <- cutree(clusterward5, k = 3)
cluster5.fac <- factor(cluster5, labels = c("Type 1", "Type 2", "Type 3"))
table(cluster5)

pars_sp5_type1 = sp5[cluster5.fac=="Type 1",]
pars_sp5_type2 = sp5[cluster5.fac=="Type 2",]
pars_sp5_type3 = sp5[cluster5.fac=="Type 3",]

pars_sp5_type1[sample(nrow(pars_sp5_type1),1),]
pars_sp5_type2[sample(nrow(pars_sp5_type2),1),]
pars_sp5_type3[sample(nrow(pars_sp5_type3),1),]

sp5_monomials = c('__s37*__s47*bind_BidM_BaxM_to_BidMBaxM_kf',
                  '__s41*catalyze_BidMBaxC_to_BidM_BaxM_kc', 
                  '__s47*__s54*bind_BaxA_BaxM_to_BaxABaxM_kf', 
                  '__s49*bind_BidM_BaxM_to_BidMBaxM_kr', 
                  '__s56*bind_BaxA_BaxM_to_BaxABaxM_kr')

seqfplot(sp5.seq, group = cluster5.fac, border=NA)
seqIplot(sp5.seq, group = cluster5.fac, sortv = "from.start")

jpeg('/home/oscar/tropical_project_new/clusters/Bax2.jpg', width=1000, height=700, quality=100)
seqrplot(sp5.seq, dist.matrix = sp5.lcp , group = cluster5.fac, border = NA, title='Bax clustering', 
         xlab='time (min)', cex.lab=1.8, ltext=sp5_monomials, cex.legend=1.6, cex.plot=1.8, cex.main=1.8)
dev.off()


#############
sp6 = read.csv(df_list[6], check.names = FALSE, header=T, row.names=1)
sp6.seq = seqdef(sp6)
#seqfplot(sp.seq, withlegend = T, border = NA, title = "Sequence Frequency plot")
#clustering

sp6.lcp = seqdist(sp6.seq,method="OM", sm="TRATE")
clusterward6 <- agnes(sp6.lcp, diss = TRUE, method = "ward")
plot(clusterward6, which.plots = 2)
cluster6 <- cutree(clusterward6, k = 3)
cluster6.fac <- factor(cluster6, labels = c("Type 1", "Type 2", "Type 3"))
table(cluster6)

pars_sp6_type1 = sp6[cluster6.fac=="Type 1",]
pars_sp6_type2 = sp6[cluster6.fac=="Type 2",]
pars_sp6_type3 = sp6[cluster6.fac=="Type 3",]

pars_sp6_type1[sample(nrow(pars_sp6_type1),1),]
pars_sp6_type2[sample(nrow(pars_sp6_type2),1),]
pars_sp6_type3[sample(nrow(pars_sp6_type3),1),]

seqfplot(sp6.seq, group = cluster6.fac, border=NA)
seqIplot(sp6.seq, group = cluster6.fac, sortv = "from.start")
seqrplot(sp6.seq, dist.matrix = sp6.lcp , group = cluster6.fac, border = NA)

mudata = read.table("/home/oscar/Documents/tropical_project/parameters_2000/pars_embedded_715.txt", sep = ',' )

############################# Parameter distributions

iterations_sp5 = nrow(pars_sp5_type1)
output_sp5 = matrix(ncol=126,nrow = iterations_sp5)

iterations_sp52 = nrow(pars_sp5_type2)
output_sp52 = matrix(ncol=126,nrow = iterations_sp52)

iterations_sp53 = nrow(pars_sp5_type3)
output_sp53 = matrix(ncol=126,nrow = iterations_sp53)

for(i in 1:iterations_sp5){
  par = rownames(pars_sp5_type1)[i]
  getting_path = unlist(strsplit(par, "new", fixed = TRUE))
  real_path = paste("/home/oscar/tropical_project_new", getting_path, sep="")
  mydata_sp5 = read.table(getting_path, sep=',' )
  output_sp5[i,] = as.numeric(unlist(mydata_sp5['V2']))
}

for(i in 1:iterations_sp52){
  par = rownames(pars_sp5_type2)[i]
  getting_path = unlist(strsplit(par, "new", fixed = TRUE))
  real_path = paste("/home/oscar/tropical_project_new", getting_path, sep="")
  mydata_sp52 = read.table(getting_path )
  output_sp52[i,] = as.numeric(unlist(mydata_sp52['V2']))
}

for(i in 1:iterations_sp53){
  par = rownames(pars_sp5_type3)[i]
  getting_path = unlist(strsplit(par, "new", fixed = TRUE))
  real_path = paste("/home/oscar/tropical_project_new", getting_path, sep="")
  mydata_sp53 = read.table(getting_path )
  output_sp53[i,] = as.numeric(unlist(mydata_sp53['V2']))
}

output_sp5=data.frame(output_sp5,row.names = rownames(pars_sp5_type1))
output_sp52=data.frame(output_sp52,row.names = rownames(pars_sp5_type2))
output_sp53=data.frame(output_sp53,row.names = rownames(pars_sp5_type3))

for (i in names(output_sp5)){
#  par(mfrow=c(2, 2))
#  hist(log(output_sp5[[i]]), breaks=6, freq=FALSE, main=i)
#  hist(log(output_sp52[[i]]), breaks=6, freq=FALSE, main=i)
#  hist(log(output_sp53[[i]]), breaks=6, freq=FALSE, main=i)
  
  
  prac1 = data.frame(par = log10(output_sp5[[i]]))
  prac2 = data.frame(par = log10(output_sp52[[i]]))
  prac3 = data.frame(par = log10(output_sp53[[i]]))
  
  prac1$veg = 'type1'
  prac2$veg = 'type2'
  prac3$veg = 'type3'
  
  prac = rbind(prac1,prac2,prac3)
  print(ggplot(prac, aes(par,fill=veg))+geom_histogram(alpha = 0.5, aes(y = ..density..), position = 'identity')+ggtitle(i))
  remove(prac1,prac2,prac3,prac)
}

sp1_type1_pars = rownames(pars_sp1_type1)
sp1_type2_pars = rownames(pars_sp1_type2)
sp1_type3_pars = rownames(pars_sp1_type3)
sp2_type1_pars = rownames(pars_sp2_type1)
sp2_type2_pars = rownames(pars_sp2_type2)
sp2_type3_pars = rownames(pars_sp2_type3)
sp5_type1_pars = rownames(pars_sp5_type1)
sp5_type2_pars = rownames(pars_sp5_type2)
sp5_type3_pars = rownames(pars_sp5_type3)
sp6_type1_pars = rownames(pars_sp6_type1)
sp6_type2_pars = rownames(pars_sp6_type2)
sp6_type3_pars = rownames(pars_sp6_type3)

cluster_parameters = c(clus1_sp1 = list(sp1_type1_pars), clus2_sp1 = list(sp1_type2_pars), clus3_sp1 = list(sp1_type3_pars), 
                       clus1_sp2 = list(sp2_type1_pars), clus2_sp2 = list(sp2_type2_pars), clus3_sp2 = list(sp2_type3_pars),
                       clus1_sp5 = list(sp5_type1_pars), clus2_sp5 = list(sp5_type2_pars), clus3_sp5 = list(sp5_type3_pars), 
                       clus1_sp6 = list(sp6_type1_pars), clus2_sp6 = list(sp6_type2_pars), clus3_sp6 = list(sp6_type3_pars))

for (i in names(cluster_parameters)){
  write.table(cluster_parameters[[i]], file = paste("/home/oscar/tropical_project_new/parameters_in_cluster5000/",i), row.names = FALSE, col.names = FALSE, sep = ',')
  
}
##################### Merging function, fix this for the correlations

multmerge = function(filenames){
#  filenames=list.files(path=mypath, full.names=TRUE)
  datalist = lapply(filenames, function(x){t(data.frame(read.table(file=x,header=F, row.name=1)))})
  Reduce(function(x, y) merge(x, y, all=TRUE), datalist)
}

aa = t(multmerge(sp5_type3_pars))

####################  

hist(output_sp5$X126)
hist(output_sp52$X126)
hist(output_sp53$X126)


############################################################# correlation within the clusters

# ++++++++++++++++++++++++++++
# flattenCorrMatrix
# ++++++++++++++++++++++++++++
# cormat : matrix of the correlation coefficients
# pmat : matrix of the correlation p-values
flattenCorrMatrix <- function(cormat, pmat) {
  ut <- upper.tri(cormat)
  data.frame(
    row = rownames(cormat)[row(cormat)[ut]],
    column = rownames(cormat)[col(cormat)[ut]],
    cor  =(cormat)[ut ],
    p = pmat[ut]
  )
}

parameter_correlations_cluster = function(pars_list, number_pars){
  iterations = length(pars_list)-1
  variables = number_pars
  
  output <- matrix(ncol=variables, nrow=iterations)
  
  for(i in 1:iterations){
    output[i,] <- t(read.table(pars_list[i], row.names = 1)) 
    
  }
  pars_values = data.frame(output)
  colnames(pars_values) = colnames(t(read.table(pars_list[1], row.names = 1)) ) 
  
  for(i in names(pars_values)){
    if (sd(pars_values[[i]]) == 0){
      pars_values[[i]] = NULL
    }
  }
  res<-rcorr(as.matrix(pars_values), type = 'pearson')
  final_par_info = flattenCorrMatrix(res$r, res$P)
  final_par_info = final_par_info[order(final_par_info$cor, decreasing = TRUE),]
}

all_parameters = list.files('/home/oscar/Documents/tropical_project/parameters_3000')
parameter_correlations_cluster(sp5_type2_pars, 126)
##################################################################

