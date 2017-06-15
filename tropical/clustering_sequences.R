library(WeightedCluster)
library(TraMineR)
library(seqdist2)
library(gplots)
library(cluster)
library(colorspace)
setwd('/Users/dionisio/Desktop')
ClusterDynS <- function(signatures, sm=NA, nclusters=2, clustered_pars_path='', dissMethod="OM", clusterMethod="ward", numcpu=1 ){
  # Loading dynamic signatures
  spSignatures <- read.csv(signatures, check.names=FALSE, header=T, row.names=1)
  colnames(spSignatures) = round(as.numeric(colnames(spSignatures))/3600, digits=2) # This is only for apoptosis. it transforms seconds to hours

  # Aggregating and weighting dynamic signatures that are the same
  aggSp <- wcAggregateCases(spSignatures)
  uniqueSignatures <- spSignatures[aggSp$aggIndex,]

  # Check that there are at least 2 different sequences in the species signatures
  if (dim(uniqueSignatures)[1] == 1){
    stop("Sequences are identical")
  }

  # Converting signature to TraMineR sequence and getting similarity matrix with optimal matching(OM)
  sp.seq = seqdef(uniqueSignatures[,1:40], weights = aggSp$aggWeights)
  # cpal(sp.seq) = rainbow_hcl(length(alphabet(sp.seq)))
  cpal(sp.seq) = c('aliceblue', 'antiquewhite4', 'coral2', 'aquamarine4 ', 'aquamarine',
                   'azure4', 'blueviolet') #, 'black', 'blue ', 'blue4')
                   # 'beige', 'hotpink', 'brown4 ', 'darkgoldenrod1', 'chartreuse')
                   #'red', 'green4', 'olivedrab4')
  if (sm == 'CONSTANT' || sm == 'TRATE' || is.matrix(sm)){
    diss = seqdistOO(sp.seq, method=dissMethod, sm=sm, numcpu=numcpu)
  }
  else if(file.exists(sm)){
    sm = read.csv(sm, check.names = FALSE, header=T, row.names=1)
    sm_ready = as.matrix(sm[alphabet(sp.seq), alphabet(sp.seq)])
    indel = max(sm_ready)/2
    diss = seqdistOO(sp.seq, method=dissMethod, indel=indel, sm=sm_ready, numcpu=numcpu)
  }
  else{stop("A valid value must be given for optimal matching")}

  # Clustering
  if(clusterMethod == "ward"){
    wardCluster = hclust(as.dist(diss), method='ward.D2', members=aggSp$aggWeights)
    # plot(wardCluster)
    # groups <- cutree(wardCluster, k=9)
    # rect.hclust(wardCluster, k=9, border="red")
    clust <- cutree(wardCluster, k = nclusters)
    clust.fac <- factor(clust, labels = paste("Type", 1:nclusters))
      seqdplot(sp.seq, group=clust.fac,border=NA)
    seqIplot(sp.seq, group=clust.fac, sortv = "from.start")
    wardRange = as.clustrange(wardCluster, diss=diss, weights=aggSp$aggWeights, ncluster=15)
    print(summary(wardRange, max.rank=2))
    # heatmap.2(as.matrix(diss), Rowv=as.dendrogram(wardCluster), Colv= as.dendrogram(wardCluster), scale="none", revC=T, labRow=F, labCol=F, main='Bid Mitochondria')
    if(clustered_pars_path != ''){
      clust <- cutree(wardCluster,k=nclusters)
      types = paste("Type", 1:nclusters, sep='')
      cluster.fac <- factor(clust, labels = types)
      for (k in types){
        pars_sp_type = sp.seq[cluster.fac==k,]
        sp_type_pars = rownames(pars_sp_type)
        name_cluster_pars = unlist(strsplit(signatures, "[/.]"))[length(unlist(strsplit(signatures, "[/.]")))-1]
        write.table(sp_type_pars, file = paste(clustered_pars_path, name_cluster_pars,'_' , k, sep=''), row.names = FALSE, col.names = FALSE, sep = ',')

      }
    }
  }
  else if (clusterMethod == "PAM"){
    pamRange = wcKMedRange(diss, kvals=2:15, weights=aggSp$aggWeights)
    print(summary(pamRange, max.rank=2))
    pamclust<-wcKMedoids(diss,k=nclusters,weights= aggSp$aggWeights)
    # print(unique(pamclust$clustering))
    # types = paste("Type", 1:nclusters, sep='')
    types=1:nclusters
    cluster.fac <- factor(pamclust$clustering, labels=types)
    jpeg('mbid_clusters.jpg', width=1280, height=1280, quality = 200, pointsize = 24)
    seqIplot(sp.seq, group = cluster.fac, sortv='from.start', border=NA)
    # seqdplot(sp.seq, group=cluster.fac, xlab='Time(hours)')
    dev.off()
    
    # seqmtplot(sp.seq, group = cluster.fac, withlegend=FALSE)
    # seqrplot(sp.seq, dist.matrix = diss, group = cluster.fac, border = NA)
    seqlegend(sp.seq, ncol=5)
    if(clustered_pars_path != ''){
      all_seqs_cluster = pamclust$clustering[aggSp$disaggIndex]
      cluster2.fac = factor(all_seqs_cluster, labels=types)
      cluster_vals_info = as.numeric(levels(cluster2.fac)[cluster2.fac])
      name_cluster_pars = unlist(strsplit(signatures, "[/.]"))[length(unlist(strsplit(signatures, "[/.]")))-1]
      write.table(cluster_vals_info, file = paste(clustered_pars_path, name_cluster_pars, '.csv', sep=''), row.names = FALSE, col.names = FALSE, sep = ',')
    }
    
  }
  else{stop('A valid method of clustering must be provided')}
  
}
# df_list = '/Users/dionisio/Documents/data_frame_global_signature.csv'
# df_list = '/Users/dionisio/Documents/PhD/tropical_earm_IC/KPAR_0/dataframes_IC_consumption/data_frame37.csv'
# df_list = '/Users/dionisio/PycharmProjects/tropical/examples/EARM/dfs_diff_IC_bax/data_frame37.csv'
# df_list = '/Users/dionisio/Documents/PhD/tropical_earm_IC/KPAR_541/signatures_cons_20170501/data_frame37.csv'
# df_list = '/Users/dionisio/Documents/PhD/tropical_earm_IC/KPAR_624/signatures_cons_20170612/data_frame37.csv'
df_list = '/Users/dionisio/Documents/PhD/tropical_earm_IC/KPAR_0/signatures_cons_k0_20170612/data_frames/data_frame37.csv'

ClusterDynS(df_list, sm='CONSTANT', nclusters=2, clusterMethod='PAM', clustered_pars_path = '/Users/dionisio/Desktop/', numcpu=4)


