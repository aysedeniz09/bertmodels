---
title: "BERTopicANTMN"
author: "Ayse Deniz Lokmanoglu"
date: "2022-09-19"
output: github_document
---

```{r setup, }
knitr::opts_chunk$set(echo = TRUE)
knitr::opts_chunk$set(message = FALSE)
knitr::opts_chunk$set(warning = FALSE, message = FALSE) 
knitr::opts_chunk$set(eval = FALSE)
```

### This code runs ANTMN on BERTopic in R Studio following the BERTopic Model [Code](https://github.com/aysedeniz09/bertmodels/blob/main/Bert_Antmn_Github.ipynb)

## The method is from the [supplemental code](https://github.com/DrorWalt/ANTMN), citation: Walter, D., & Ophir, Y. (2019). News Frame Analysis: An Inductive Mixed-Method Computational Approach. Communication Methods and Measures. <https://doi.org/10.1080/19312458.2019.1639145>.
1. Load the packages
```{r}
library(dplyr)
library(tidyverse)
library(readr)
library(igraph)
library(corpustools)
library(lsa)
```
2. Load topic names from the BERTopic outputs, document saved as *BERTopic_ANTMN_TopicNamesandFreq.csv* from the *freq* variable in BERTopic
```{r}
url_freq <- c("https://raw.githubusercontent.com/aysedeniz09/bertmodels/main/data/BERTopic_ANTMN_TopicNamesandFreq.csv")
bert_topic<-read.csv(url_freq)
##drop the negative row which is the outlier
bert_topic <- bert_topic[-c(1), ]
# create a list
topic_names<-bert_topic$Name
```
3. You can calculate topic size in two different ways
3a. First by using the frequency count within the BERTopic
```{r}
topicsize_freq<-bert_topic$Count
```
3b. Or using the probability document in Step 4
4. Load the Probability dataframe
```{r}
url_prob <- c("https://raw.githubusercontent.com/aysedeniz09/bertmodels/main/data/BERTopic_ANTMN_Probabilities.csv")
probs <- readr::read_csv(url_prob)
# remove the index column created by csv
probs<-probs %>%
  dplyr::select(-...1)
```
4a. (also step 3b) You can calculate topic size using the probabilit document, by getting the mean probability for each topic
```{r}
topicsize_prob<-colMeans(probs[,1:ncol(probs)])
```
5. Create [ANTMN](https://github.com/DrorWalt/ANTMN) function
```{r}
# load libraries
library(igraph)
library(corpustools)
library(lsa)

network_from_LDA<-function(Probobject,deleted_topics=c(),topic_names=c(),save_filename="",topic_size=c(),bbone=FALSE) {
  # Importing needed packages
  require(lsa) # for cosine similarity calculation
  require(dplyr) # general utility
  require(igraph) # for graph/network management and output
  require(corpustools)
  
  print("Importing model")
  
  # first extract the theta matrix form the BERTopic Probability object
  theta<-Probobject
  
  # calculate the adjacency matrix using cosine similarity on the theta matrix
  mycosine<-cosine(as.matrix(theta))
  colnames(mycosine)<-colnames(theta)
  rownames(mycosine)<-colnames(theta)
  
  # Convert to network - undirected, weighted, no diagonal
  
  print("Creating graph")
  
  topmodnet<-graph.adjacency(mycosine,mode="undirected",weighted=T,diag=F,add.colnames="label") # Assign colnames
  # add topicnames as name attribute of node - importend from prepare meta data in previous lines
  if (length(topic_names)>0) {
    print("Topic names added")
    V(topmodnet)$name<-topic_names
  } 
  # add sizes if passed to funciton
  if (length(topic_size)>0) {
    print("Topic sizes added")
    V(topmodnet)$topic_size<-topic_size
  }
  newg<-topmodnet
  
  # delete 'garbage' topics
  if (length(deleted_topics)>0) {
    print("Deleting requested topics")
    
    newg<-delete_vertices(topmodnet, deleted_topics)
  }
  
  # Backbone
  if (bbone==TRUE) {
    print("Backboning")
    
    nnodesBASE<-length(V(newg))
    for (bbonelvl in rev(seq(0,1,by=0.05))) {
      #print (bbonelvl)
      nnodes<-length(V(backbone_filter(newg,alpha=bbonelvl)))
      if(nnodes>=nnodesBASE) {
        bbonelvl=bbonelvl
        #  print ("great")
      }
      else{break}
      oldbbone<-bbonelvl
    }
    
    newg<-backbone_filter(newg,alpha=oldbbone)
    
  }
  
  # run community detection and attach as node attribute
  print("Calculating communities")
  
  mylouvain<-(cluster_louvain(newg)) 
  mywalktrap<-(cluster_walktrap(newg)) 
  myfastgreed<-(cluster_fast_greedy(newg)) 
  myeigen<-(cluster_leading_eigen(newg)) 
  
  V(newg)$louvain<-mylouvain$membership 
  V(newg)$walktrap<-mywalktrap$membership 
  V(newg)$fastgreed<-myfastgreed$membership 
  V(newg)$eigen<-myeigen$membership 
  V(newg)$degree <- degree(newg)                        # Degree centrality
  V(newg)$eig <- evcent(newg)$vector                    # Eigenvector centrality
  V(newg)$hubs <- hub.score(newg)$vector                # "Hub" centrality
  V(newg)$authorities <- authority.score(newg)$vector   # "Authority" centrality
  V(newg)$closeness <- closeness(newg)                  # Closeness centrality
  V(newg)$betweenness <- betweenness(newg)  
  # if filename is passsed - saving object to graphml object. Can be opened with Gephi.
  if (nchar(save_filename)>0) {
    print("Writing graph")
    write.graph(newg,paste0(save_filename,".graphml"),format="graphml")
  }
  
  # graph is returned as object
  return(newg)
}
```
6. Run the ANTMN function with the probability, topic name and size dataframes
```{r}
mynewnet<-network_from_LDA(Probobject=probs,
                           topic_names=topic_names,
                           topic_size=topicsize_freq,
                     save_filename="BERT_ANTMN_Graph",
                           bbone=TRUE)

save(mynewnet, file="BERTopic_Antmn_Graph.Rda")
```
7. Open file in [Gephi](https://gephi.org/) to visualize it with Walktrap Algorithm.![Network Graph](https://github.com/aysedeniz09/bertmodels/blob/main/images/BERT_ANTMN_Graph.png?raw=true) 
