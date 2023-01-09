#Libraries
install.packages('caret', dependencies = TRUE)
install.packages("devtools")
install.packages('tidyverse', dependencies = TRUE)
install.packages('parallel')
install.packages('doParallel')
install.packages('MLmetrics')
install.packages('factoextra', dependencies = TRUE)
install.packages("corrplot")
install.packages("gridExtra")
install.packages("GGally")
install.packages("knitr")
library(factoextra)
library(caret)
library(tidyverse)
library(parallel)
library(doParallel)
library(MLmetrics)
library(ggplot2)
library(stats)
library(cluster)
library(corrplot)
library(gridExtra)
library(GGally)
library(knitr)

#Read data
data <- read.csv2("data.csv", dec = ".")
data$Pos <- as.factor(data$Pos)
data <- data %>% select(-c(Rk, Player, Nation, Squad, Comp, Age, Born))
data <- subset(data, MP>10)
data <- data %>% select(-c(MP,Starts,Min,X90s))
data <- droplevels(data)

data
dim(data)
class(data)

lapply(data, class)

data = data[ , !(names(data) %in% c('Pos'))]

#scaled value 
scaled = scale(data)
scaled
summary(scaled)

# Eligiendo el valor de k para el k-Means
vector_compactacion<-0
for(i in 1:11){
  km_puntos_aux2<-kmeans(scaled,center=i,nstar=20)
  vector_compactacion[i] <- km_puntos_aux2$tot.withinss
}
# Construye rejilla 1x1
par(mfrow = c(1,1)) 
# Representamos sum of squares vs. number of clusters
plot(1:11, vector_compactacion, type = "b", 
     xlab = "Numero de clusters", 
     #ylab = "Within groups sum of squares")
     ylab = "Compactacion")

# Aseguramos la reproducibilidad 
seed_val = 39128
set.seed(seed_val)
# Tres clusters
k = 3
# Primera ejecución del k-Means
first_clust = kmeans(scaled, centers = k, nstart = 1)
# Vemos cuantos pacientes hay en cada grupo
first_clust$size


# Aseguramos la reproducibilidad 
seed_val = 38
set.seed(seed_val)
# Tres clusters
k = 3
second_clust = kmeans(scaled, k, nstart=1)
# Vemos cuantos pacientes hay en cada grupo
second_clust$size

# Añadimos columnas adicionales
data['first_clust'] = first_clust$cluster # Clúster al que pertenece en la primera ejecución del kMeans
data['second_clust'] = second_clust$cluster # Clúster al que pertenece en la primera ejecución del kMeans
# Cargamos la librería ggplot2
library(ggplot2)

plot_one = ggplot(data, aes(x=Shots, y=Goals, color=as.factor(first_clust))) + geom_point()
plot_one 

plot_two = ggplot(data, aes(x=Shots, y=Goals, color=as.factor(second_clust))) + geom_point()
plot_two

plot_three = ggplot(data, aes(x=TouDef3rd, y=PresMid3rd, color=as.factor(second_clust))) + geom_point()
plot_three

first_clust$cluster

dist_E=dist(scaled) #distanza euclidea
dist_MH=dist(scaled,method="manhattan")
dist_MH

#clustering completo
data_co=hclust(dist_E,method="complete")
summary(data_co)

#dendogramma
hc_1_assign <- cutree(data_co, 3)
plot(data_co)


data_co$merge #restituisce i passi degli abbinamenti :passo 1 combinati elemento 317 con elemento 1188 e così via
#costruisce un cluster ad ogni passo. Tutti gli elementi non preceduti da un meno sono cluster


#clustering single
data_si=hclust(dist_E,method="single")
summary(data_si)
data_si$merge #restituisce i passi degli abbinamenti :passo 1 combinati elemento 317 con elemento 1188 e così via
#costruisce un cluster ad ogni passo. Tutti gli elementi non preceduti da un meno sono cluster

#dendogramma
hc_2_assign <- cutree(data_si,3)
plot(data_si)

data_mediana=hclust(dist_E,method="median")
data_average=hclust(dist_E,method="average")
data_centr=hclust(dist_E,method="centroid")

par(mfrow=c(2,3))
plot(data_co)
plot(data_si)
plot(data_mediana)
plot(data_average)
plot(data_centr)

# adding assignments of chosen hierarchical linkage
data['hc_clust'] = hc_1_assign

hd_simple = data[, !(names(data) %in% c( 'first_clust', 'second_clust'))]
# getting mean and standard deviation summary statistics
clust_summary = do.call(data.frame, aggregate(. ~hc_clust, data = hd_simple, function(x) c(avg = mean(x), sd = sd(x))))
clust_summary
# plotting Shots and Goals
plot_one_hc = ggplot(hd_simple, aes(x=Shots, y=Goals, color=as.factor(hc_clust))) + geom_point()
plot_one_hc 

#####

desc_stats <- data.frame(
  Min = apply(data, 2, min), # minimum
  Med = apply(data, 2, median), # median
  Mean = apply(data, 2, mean), # mean
  SD = apply(data, 2, sd), # Standard deviation
  Max = apply(data, 2, max) # Maximum
)
desc_stats <- round(desc_stats, 1)
head(desc_stats)

res <- get_clust_tendency(scaled, 40, graph = FALSE)
# Hopskin statistic
res$hopkins_stat
res$plot

#set.seed(123)
# Compute the gap statistic
#gap_stat <- clusGap(scaled, FUN = kmeans, nstart = 25, 
#                    K.max = 2, B = 500) 
#fviz_gap_stat(gap_stat)

fviz_nbclust(scaled, kmeans, method = "silhouette") #vediamo numero di cluster da applicare 

fviz_nbclust(scaled, kmeans, method = "wss") #+ #vediamo numero di cluster da applicare 
#geom_vline(xintercept = 3, linetype = 2)

set.seed(123)
km.res <- kmeans(scaled, 3, nstart = 25)
head(km.res$cluster, 20)
fviz_cluster(km.res, scaled)

sil <- silhouette(km.res$cluster, dist(scaled))
rownames(sil) <- rownames(data)
head(sil[, 1:3])
fviz_silhouette(sil)

neg_sil_index <- which(sil[, "sil_width"] < 0)
sil[neg_sil_index, , drop = FALSE]

# Compute k-means
res.km <- eclust(scaled, "kmeans")
# Gap statistic plot
fviz_gap_stat(res.km$gap_stat)

# Silhouette plot
fviz_silhouette(res.km)

# Enhanced hierarchical clustering
res.hc <- eclust(scaled, "hclust") # compute hclust

##

#Correlation matrix
corrplot(cor(scaled), type="upper", method="ellipse", tl.cex=0.9)
M = cor(scaled)
(order.AOE = corrMatOrder(M, order = 'AOE'))
(order.FPC = corrMatOrder(M, order = 'FPC'))
(order.hc = corrMatOrder(M, order = 'hclust'))
(order.hc2 = corrMatOrder(M, order = 'hclust', hclust.method = 'ward.D'))
M.AOE = M[order.AOE, order.AOE]
M.FPC = M[order.FPC, order.FPC]
M.hc  = M[order.hc, order.hc]
M.hc2 = M[order.hc2, order.hc2]

corrplot(M)
corrplot(M.AOE)
corrplot(M.FPC)
corrplot(M.hc)
corrplot(M.hc)
corrRect.hclust(corr = M.hc, k = 3)
corrplot(M.hc)
corrRect.hclust(corr = M.hc, k = 3)
corrplot(M.hc2)
corrRect.hclust(M.hc2, k = 2, method = 'ward.D')



ggplot(data, aes(x=Goals, y=Shots)) +
  geom_point() +
  geom_smooth(method="lm", se=FALSE) +
  labs(
    subtitle="Relationship between Shots and Goals") +
  theme_bw()

# Normalization
xNorm <- as.data.frame(scale(data))
# Original data
p1 <- ggplot(data, aes(x=Goals, y=Shots)) +
  geom_point() +
  labs(title="Original data") +
  theme_bw()
# Normalized data 
p2 <- ggplot(xNorm, aes(x=Goals, y=Shots)) +
  geom_point() +
  labs(title="Normalized data") +
  theme_bw()
# Subplot
grid.arrange(p1, p2, ncol=2)
# Execution of k-means with k=2
set.seed(1234)
xKmeans <- kmeans(xNorm, centers=3)

# Estudio de xKmeans

# Cluster to which each point is allocated
xKmeans$cluster
# Cluster centers
xKmeans$centers
# Cluster size
xKmeans$size
# Between-cluster sum of squares
xKmeans$betweenss
# Within-cluster sum of squares
xKmeans$withinss
# Total within-cluster sum of squares 
xKmeans$tot.withinss
# Total sum of squares
xKmeans$totss
bss <- numeric()
wss <- numeric()
set.seed(1234)
for(i in 1:10){
  # For each k, calculate betweenss and tot.withinss
  bss[i] <- kmeans(xNorm, centers=i)$betweenss
  wss[i] <- kmeans(xNorm, centers=i)$tot.withinss
}
# Between-cluster sum of squares vs Choice of k
p3 <- qplot(1:10, bss, geom=c("point", "line"), 
            xlab="Number of clusters", ylab="Between-cluster sum of squares") +
  scale_x_continuous(breaks=seq(0, 10, 1)) +
  theme_bw()
# Total within-cluster sum of squares vs Choice of k
p4 <- qplot(1:10, wss, geom=c("point", "line"),
            xlab="Number of clusters", ylab="Total within-cluster sum of squares") +
  scale_x_continuous(breaks=seq(0, 10, 1)) +
  theme_bw()
grid.arrange(p3, p4, ncol=2)
# Execution of k-means with k=3
set.seed(1234)
xKmeans3 <- kmeans(xNorm, centers=3)
# Mean values of each cluster
aggregate(data, by=list(xKmeans3$cluster), mean)
# Clustering 
ggpairs(cbind(data, Cluster=as.factor(xKmeans3$cluster)),
        columns=1:9, aes(colour=Cluster, alpha=0.5),
        lower=list(continuous="points"),
        upper=list(continuous="blank"),
        axisLabels="none", switch="both") +
  theme_bw()

