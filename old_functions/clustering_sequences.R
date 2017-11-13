library(WeightedCluster)
library(TraMineR)
library(gplots)
library(cluster)
library(colorspace)
library(randomcoloR)

setwd('/Users/dionisio/Desktop')
ClusterDynS <- function(signatures, sm=NA, nclusters=2, dissMethod="LCS", clusterMethod="ward", clustered_pars_path=''){
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

  # Converting signature to TraMineR sequence and getting similarity matrix with defined metric
  sp.seq = seqdef(uniqueSignatures[,0:51], weights = aggSp$aggWeights)
  set.seed(001)
  ncolors = length(alphabet(sp.seq))
  cpal(sp.seq) = distinctColorPalette(ncolors)

  if (sm == 'CONSTANT' || sm == 'TRATE' || is.matrix(sm)){
    diss = seqdist(sp.seq, method=dissMethod, sm=sm, indel = 5)
  }
  else if(file.exists(sm)){
    sm = read.csv(sm, check.names = FALSE, header=T, row.names=1)
    sm_ready = as.matrix(sm[alphabet(sp.seq), alphabet(sp.seq)])
    indel = max(sm_ready)/2
    diss = seqdist(sp.seq, method=dissMethod, indel=indel, sm=sm_ready)
  }
  else{stop("A valid value must be given for optimal matching")}
  
  # Clustering
  if(clusterMethod == "ward"){
    wardCluster = hclust(as.dist(diss), method='ward.D2', members=aggSp$aggWeights)
    wardRange = as.clustrange(wardCluster, diss=diss, weights=aggSp$aggWeights, ncluster=20)
    # jpeg('ward_statistics.jpg', width=1280, height=1280, quality = 200, pointsize = 24)
    # plot(wardRange, stat=c("ASWw", "HG", "PBC", "HC"))
    # dev.off()
    print(summary(wardRange, max.rank=2))
    
    clust <- cutree(wardCluster, k = nclusters)
    types=1:nclusters
    cluster.fac <- factor(clust, labels = types)
    for (k in types){
      print(k)
      print(seqmodst(sp.seq[cluster.fac==k,]))
    }
    graphics.off()
    par("mar")
    par(mar=c(1,1,1,1))
    jpeg('mbid_clusters.jpg', width=1280, height=1280, quality = 200, pointsize = 24)
    # seqrplot(sp.seq, dist.matrix = diss, group = cluster.fac, trep = 0.35, border = NA, with.legend = FALSE)
    # seqlegend(sp.seq, ncol=5)
    # seqmsplot(sp.seq, group = cluster.fac, sortv='from.start', border=NA, with.legend = FALSE)
    
    sil = wcSilhouetteObs(diss, clust, weights=aggSp$aggWeights, measure="ASWw") 
    seqIplot(sp.seq, group = cluster.fac, sortv=sil, border=NA, with.legend = FALSE)
    
    # seqpcplot(sp.seq, group = cluster.fac, sortv='from.start', border=NA, with.legend = FALSE)
    # seqmtplot(sp.seq, group = cluster.fac, sortv='from.start', border=NA, with.legend = FALSE)
    # seqHtplot(sp.seq, group = cluster.fac, sortv='from.start', border=NA, with.legend = FALSE)
    # seqfplot(sp.seq, group = cluster.fac, sortv='from.start', border=NA, with.legend = FALSE)
    # seqdplot(sp.seq, group=cluster.fac, xlab='Time(hours)', with.legend = FALSE)
    dev.off()

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
    pamRange = wcKMedRange(diss, kvals=10:25, weights=aggSp$aggWeights)
    print(summary(pamRange, max.rank=2))
    pamclust<-wcKMedoids(diss,k=nclusters,weights= aggSp$aggWeights)
    # print(unique(pamclust$clustering))
    # types = paste("Type", 1:nclusters, sep='')
    types=1:nclusters
    cluster.fac <- factor(pamclust$clustering, labels=types)
    ##### this is to see the states of the modal sequences
    # for (k in types){
    #   print(k)
    #   print(seqmodst(sp.seq[cluster.fac==k,]))
    # }
    graphics.off()
    par("mar")
    par(mar=c(1,1,1,1))
    sil = wcSilhouetteObs(diss, cluster.fac, weights=aggSp$aggWeights, measure="ASWw") 
    jpeg('mbid_clusters.jpg', width=1280, height=1280, quality = 200, pointsize = 24)
    # seqrplot(sp.seq, dist.matrix = diss, group = cluster.fac, trep = 0.35, border = NA, with.legend = FALSE)
    # seqlegend(sp.seq, ncol=5)
    # seqmsplot(sp.seq, group = cluster.fac, sortv='from.start', border=NA, with.legend = FALSE)
    seqIplot(sp.seq, group = cluster.fac, sortv=sil, border=NA, with.legend = FALSE)
    # seqpcplot(sp.seq, group = cluster.fac, sortv='from.start', border=NA, with.legend = FALSE)
    # seqmtplot(sp.seq, group = cluster.fac, sortv='from.start', border=NA, with.legend = FALSE)
    # seqHtplot(sp.seq, group = cluster.fac, sortv='from.start', border=NA, with.legend = FALSE)
    # seqfplot(sp.seq, group = cluster.fac, sortv='from.start', border=NA, with.legend = FALSE)
    # seqdplot(sp.seq, group=cluster.fac, xlab='Time(hours)', with.legend = FALSE)
    dev.off()
    
    # seqmtplot(sp.seq, group = cluster.fac, withlegend=FALSE)
    # seqrplot(sp.seq, dist.matrix = diss, group = cluster.fac, border = NA)
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

# signatures = '/Users/dionisio/Documents/PhD/tropical_earm_IC/signatures_within_range_function/signatures_all_kpars_consumption1/data_frames/data_frame37_copy.csv'
signatures = '/Users/dionisio/Documents/PhD/tropical_earm_IC/signatures_within_range_function/signatures_all_kpars_consumption_nums/data_frame37_copy.csv'
ClusterDynS(signatures, sm='CONSTANT', nclusters=10, clusterMethod='PAM', dissMethod = 'LCS', clustered_pars_path = '/Users/dionisio/Desktop/')


