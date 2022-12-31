#Libraries
install.packages(caret)
install.packages(tidyverse)
install.packages(parallel)
install.packages(doParallel)
install.packages(MLmetrics)
library(caret)
library(tidyverse)
library(parallel)
library(doParallel)
library(MLmetrics)
library(ggplot2)
library(randomForest)


#Parallel processing
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)

#Read data
data <- read.csv2("data.csv", dec = ".")
data$Pos <- as.factor(data$Pos)
data <- data %>% select(-c(Rk, Player, Nation, Squad, Comp, Age, Born))
data <- subset(data, MP>10)
data <- data %>% select(-c(MP,Starts,Min,X90s))
data <- droplevels(data)

dim(data)
class(data)

#Set random seed
set.seed(30493)


#Vistazo inicial
str(data)
summary(data)

#Preprocessing
trainIndex <- createDataPartition(data$Pos, p= 0.7, list = FALSE)
trainingSet <- data[trainIndex,]
testSet <- data[-trainIndex,]


y <- trainingSet[,1]
x<-trainingSet[,-1]

sapply(x, function(x) sum(is.na(x)))

#TrainControl
fitControl_1 <- trainControl(method="cv",number=10, savePredictions="final",search="random", allowParallel= TRUE)
fitControl_2 <- trainControl(method="repeatedcv",number=10, repeats=10,search = "random", savePredictions="final", allowParallel= TRUE)
fitControl_gbm1 <- trainControl(method="cv",number=10, savePredictions="final", allowParallel= TRUE)
fitControl_gbm2 <- trainControl(method="repeatedcv",number=10, repeats=3, savePredictions="final", allowParallel= TRUE)

#Training
#Random forest
bestMtry1000 <- tuneRF(x,y, stepFactor = 1.5, improve = 1e-5, ntree = 1000)
bestMtry1500 <- tuneRF(x,y, stepFactor = 1.5, improve = 1e-5, ntree = 1500)
bestMtry2000 <- tuneRF(x,y, stepFactor = 1.5, improve = 1e-5, ntree = 2000)

print(bestMtry1000); print(bestMtry1500); print(bestMtry2000)

grid1000 <- data.frame(mtry = 16); grid1500 <- data.frame(mtry = 16); grid2000 <- data.frame(mtry = 11)

-----------------------------------------------------------------------------------------------------------------------------------------------------
rf_b <- train(x, y, method="rf", data= data, metric="Accuracy", ntree=1000, tuneGrid = grid1000) #0.7786355
rf_b1 <- train(x, y, method="rf", data= data, metric="Accuracy", ntree=1000, trControl= fitControl_1, tuneGrid = grid1000) #0.7915388
rf_b2 <- train(x, y, method="rf", data= data, metric="Accuracy", ntree=1000, trControl= fitControl_2, tuneGrid = grid1000)#0.7870106

rf_b$resample; print(rf_b)
rf_b1$resample; print(rf_b1)
rf_b2$resample; print(rf_b2)

res_b1<-as_tibble(rf_b$results[which.min(rf_b$results[,2]),]); res_b2<-as_tibble(rf_b1$results[which.min(rf_b1$results[,2]),])
res_b3<-as_tibble(rf_b2$results[which.min(rf_b2$results[,2]),]); 

df_rfb<-tibble(Model=c('RF 1000 Trees','RF 1000 Trees CV', "RF 1000 Trees RCV"),Accuracy=c(res_b1$Accuracy,res_b2$Accuracy,res_b3$Accuracy))
df_rfb %>% arrange(Accuracy)

bAccuracy <- rf_b$results$Accuracy; b1Accuracy <- rf_b1$results$Accuracy; b2Accuracy <- rf_b2$results$Accuracy
b_accuracy_list <- c(bAccuracy, b1Accuracy, b2Accuracy)
plot(b_accuracy_list, lty=1, lwd=10)

#Seleccionamos el que mayor accuracy tenga
rf1000 <- rf_b; rf1000<-rf_b1; rf1000<- rf_b2;
-----------------------------------------------------------------------------------------------------------------------------------------------------
  
rf_c <- train(x, y, method="rf", data= data, metric="Accuracy", ntree=1500, tuneGrid = grid1500) #0.7827391
rf_c1 <- train(x, y, method="rf", data= data, metric="Accuracy", ntree=1500, trControl= fitControl_1, tuneGrid = grid1500) #0.7887743
rf_c2 <- train(x, y, method="rf", data= data, metric="Accuracy", ntree=1500, trControl= fitControl_2, tuneGrid = grid1500)#0.7889702

rf_c$resample; print(rf_c)
rf_c1$resample; print(rf_c1)
rf_c2$resample; print(rf_c2)

res_c1<-as_tibble(rf_c$results[which.min(rf_c$results[,2]),]); res_c2<-as_tibble(rf_c1$results[which.min(rf_c1$results[,2]),])
res_c3<-as_tibble(rf_c2$results[which.min(rf_c2$results[,2]),]); 

df_rfc <- tibble(Model=c('RF 1500 Trees','RF 1500 Trees CV', "RF 1500 Trees RCV"),Accuracy=c(res_c1$Accuracy,res_c2$Accuracy,res_c3$Accuracy))
df_rfc %>% arrange(Accuracy)

cAccuracy <- rf_c$results$Accuracy; c1Accuracy <- rf_c1$results$Accuracy; c2Accuracy <- rf_c2$results$Accuracy
c_accuracy_list <- c(cAccuracy, c1Accuracy, c2Accuracy)
mean_c <- mean(c_accuracy_list)
plot(c_accuracy_list, lty=1, lwd=10)

#Seleccionamos el que mayor accuracy tenga
rf1500 <- rf_c; rf1500<-rf_c1; rf1500<- rf_c2;
-----------------------------------------------------------------------------------------------------------------------------------------------------
  
rf_d <- train(x, y, method="rf", data= data, metric="Accuracy", ntree=2000, tuneGrid = grid2000) #0.7801005
rf_d1 <- train(x, y, method="rf", data= data, metric="Accuracy", ntree=2000, trControl= fitControl_1, tuneGrid = grid2000) #0.789384
rf_d2 <- train(x, y, method="rf", data= data, metric="Accuracy", ntree=2000, trControl= fitControl_2, tuneGrid = grid2000)#0.7868483

rf_d$resample; print(rf_d)
rf_d1$resample; print(rf_d1)
rf_d2$resample; print(rf_d2)

res_d1<-as_tibble(rf_d$results[which.min(rf_d$results[,2]),]); res_d2<-as_tibble(rf_d1$results[which.min(rf_d1$results[,2]),])
res_d3<-as_tibble(rf_d2$results[which.min(rf_d2$results[,2]),]); 

df_rfd <- tibble(Model=c('RF 2000 Trees','RF 2000 Trees CV', "RF 2000 Trees RCV"),Accuracy=c(res_d1$Accuracy,res_d2$Accuracy,res_d3$Accuracy))
df_rfd %>% arrange(Accuracy)

dAccuracy <- rf_d$results$Accuracy; d1Accuracy <- rf_d1$results$Accuracy; d2Accuracy <- rf_d2$results$Accuracy
d_accuracy_list <- c(dAccuracy, d1Accuracy, d2Accuracy)
mean_d <- mean(d_accuracy_list)
plot(d_accuracy_list, lty=1, lwd=10)

#Seleccionamos el que mayor accuracy tenga
rf2000 <- rf_d; rf2000<-rf_d1; rf2000<- rf_d2;
-----------------------------------------------------------------------------------------------------------------------------------------------------

#Escogemos como modelo de RF el que mayor accuracy tenga de todo lo anterior
plot(c(max(b_accuracy_list),max(c_accuracy_list),max(d_accuracy_list)), lty=1, lwd=10)
rf <- rf1000

#Naive bayes
nb_grid <-   expand.grid(usekernel = c(TRUE, FALSE),
                         laplace = c(0, 0.5, 1), 
                         adjust = c(0.75, 1, 1.25, 1.5))

nb <- train(x, y, method="naive_bayes", metric= "Accuracy", trControl = fitControl_1, tuneLength = 20, usepoisson = TRUE, tuneGrid = nb_grid)
nb2 <- train(x, y, method="naive_bayes", metric= "Accuracy", trControl = fitControl_2, tuneLength = 20, usepoisson = TRUE, tuneGrid = nb_grid)
nb$resample;print(nb)
nb2$resample;print(nb2)
plot(nb)
plot(nb2)

res_nb1<-as_tibble(nb$results[which.max(nb$results[,1]),]); res_nb2<-as_tibble(nb2$results[which.max(nb2$results[,1]),])

df_rfnb<-tibble(Model=c('Naive Bayes CV','Naive Bayes RCV'),Accuracy=c(res_nb1$Accuracy,res_nb2$Accuracy))
df_rfnb %>% arrange(Accuracy)

nb_list <- c(nb$results$Accuracy);nb_list2 <- c(nb2$results$Accuracy)
plot(nb_list, lty=1, lwd= 10); plot(nb_list2, lty=1, lwd= 10)
plot(c(max(nb_list),max(nb_list2)), lty=1, lwd=10)

#Escogemos el que mayor accuracy tenga
nb <- nb; nb <- nb2

#Support Vector Machine
svm <- train(Pos ~ ., data=trainingSet, method="svmRadial", metric= "Accuracy",preProcess = c("center","scale"), trControl= fitControl_1,  tuneLength=10)
svm2 <- train(Pos ~ ., data=trainingSet, method="svmRadial", metric= "Accuracy",preProcess = c("center","scale"), trControl= fitControl_2,  tuneLength=10)
svm3 <- train(Pos ~ ., data=trainingSet, method="svmLinear", metric= "Accuracy",preProcess = c("center","scale"), trControl= fitControl_1,  tuneLength=10)
svm4 <- train(Pos ~ ., data=trainingSet, method="svmLinear", metric= "Accuracy",preProcess = c("center","scale"), trControl= fitControl_2,  tuneLength=10)
svm5 <- train(Pos ~ ., data=trainingSet, method="svmLinear", metric= "Accuracy",preProcess = c("center","scale"), trControl= fitControl_1,  tuneGrid = expand.grid(C = seq(0, 2, length = 20)))
svm6 <- train(Pos ~ ., data=trainingSet, method="svmLinear", metric= "Accuracy",preProcess = c("center","scale"), trControl= fitControl_2,  tuneGrid = expand.grid(C = seq(0, 2, length = 20)))
svm7 <- train(Pos ~., data = trainingSet, method = "svmPoly",metric="Accuracy", preProcess = c("center","scale"),trControl= fitControl_1, tuneLength = 4)
svm8 <- train(Pos ~., data = trainingSet, method = "svmPoly",metric="Accuracy", preProcess = c("center","scale"),trControl= fitControl_1, tuneLength = 4)

svm$resample; print(svm); svm2$resample; print(svm2)
svm3$resample; print(svm3); svm4$resample; print(svm4); svm5$resample; print(svm5); svm6$resample; print(svm6)
svm7$resample; print(svm7); svm8$resample; print(svm8)


#APLICAR ESTO A TODO PARA UNA MEJOR COMPARACIÓN DE RESULTADOS
res1<-as_tibble(svm$results[which.max(svm$results[,3]),]); res2<-as_tibble(svm2$results[which.max(svm2$results[,3]),])
res3<-as_tibble(svm3$results[which.max(svm3$results[,3]),]); res4<-as_tibble(svm4$results[which.max(svm4$results[,3]),]); res5<-as_tibble(svm5$results[which.max(svm5$results[,3]),]); res6<-as_tibble(svm6$results[which.max(svm6$results[,3]),])
res7<-as_tibble(svm7$results[which.max(svm7$results[,3]),]); res8<-as_tibble(svm8$results[which.max(svm8$results[,3]),])

df_SVM<-tibble(Model=c('SVM Radial CV',"SVM Radial RCV","SVM Linear CV",'SVM Linear RCV','SVM Linear w/ choice of cost CV','SVM Linear w/ choice of cost RCV','SVM Poly CV','SVM Poly RCV'),Accuracy=c(res1$Accuracy,res2$Accuracy,res3$Accuracy,res4$Accuracy,res5$Accuracy,res6$Accuracy,res7$Accuracy,res8$Accuracy))
df_SVM %>% arrange(Accuracy)

svm_list <- c(svm$results$Accuracy); svm_list2 <- c(svm2$results$Accuracy);svm_list3 <- c(svm3$results$Accuracy);svm_list4 <- c(svm4$results$Accuracy);
svm_list5 <- c(svm5$results$Accuracy); svm_list6 <- c(svm6$results$Accuracy); svm_list7 <- c(svm7$results$Accuracy); svm_list8 <- c(svm8$results$Accuracy);
plot(nb_list, lty=1, lwd= 10); plot(nb_list2, lty=1, lwd= 10)
plot(c(max(svm_list),max(svm_list2), max(svm_list3), max(svm_list4), max(svm_list5), max(svm_list6), max(svm_list7), max(svm_list8)), lty=1, lwd=10)

#Escogemos el que mayor accuracy tenga
svm <- svm; svm <- svm1; svm <- svm2; svm <- svm3; svm <- svm4; svm <- svm5; svm <- svm6; svm <- svm7; svm <- svm8; 

#GBM
gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9, 12, 15), 
                        n.trees = 150, 
                        shrinkage = 0.1,
                        n.minobsinnode = 20)

gbm <- train(Pos ~ .,data=trainingSet, method="gbm",metric="Accuracy",trControl= fitControl_gbm1)
gbm2 <- train(Pos ~ .,data=trainingSet, method="gbm",metric="Accuracy",trControl= fitControl_gbm2)

#gbm3 <- train(Pos ~ .,data=trainingSet, method="gbm",metric="Accuracy",trControl= fitControl_gbm1, tuneGrid= gbmGrid)
#gbm4 <- train(Pos ~ .,data=trainingSet, method="gbm",metric="Accuracy",trControl= fitControl_gbm2, tuneGrid= gbmGrid)

gbm$resample; print(gbm); gbm2$resample; print(gbm2)
ggplot(gbm); ggplot(gbm2)

res_gb1<-as_tibble(gbm$results[which.max(gbm$results[,5]),]); res_gb2<-as_tibble(gbm2$results[which.max(gbm2$results[,5]),])

df_GBM<-tibble(Model=c('GBM CV','GBM RCV'),Accuracy=c(res_gb1$Accuracy,res_gb2$Accuracy))
df_GBM %>% arrange(Accuracy)

#Escogemos el que mayor accuracy tenga
gbm <- gbm; gbm <- gbm2;

#KNN
knn <- train(x, y, method = "knn", metric= "Accuracy", preProcess = c("center","scale"), trControl = fitControl_1, tuneLength = 20)
knn2 <- train(x, y, method = "knn", metric= "Accuracy", preProcess = c("center","scale"), trControl = fitControl_2, tuneLength = 20)
plot(knn); plot(knn2)

res_knn1<-as_tibble(knn$results[which.max(knn$results[,2]),]); res_knn2<-as_tibble(knn2$results[which.max(knn2$results[,2]),])

df_knn<-tibble(Model=c('KNN CV','KNN RCV'),Accuracy=c(res_knn1$Accuracy,res_knn2$Accuracy))
df_knn %>% arrange(Accuracy)

knn_list <- c(knn$results$Accuracy)
knn_list2 <- c(knn2$results$Accuracy)
plot(knn_list, lty=1, lwd=10); plot(knn_list2, lty=1, lwd=10)
plot(c(max(knn_list),max(knn_list2)), lty=1, lwd=10)

#Escogemos el que mayor accuracy tenga
knn <- knn; knn <- knn2;

#RPART
tree <- train(Pos ~ .,data=trainingSet, method="rpart2",metric="Accuracy",trControl= fitControl_1, tuneLength = 100)
tree2 <- train(Pos ~ .,data=trainingSet, method="rpart2",metric="Accuracy",trControl= fitControl_2, tuneLength = 100)
plot(tree); plot(tree2);

tree$resample; print(tree);
tree2$resample; print(tree2);

res_tree1<-as_tibble(tree$results[which.max(tree$results[,2]),]); res_tree2<-as_tibble(tree2$results[which.max(tree2$results[,2]),])

df_rpart<-tibble(Model=c('RPART CV','RPART RCV'),Accuracy=c(res_tree1$Accuracy,res_tree2$Accuracy))
df_rpart %>% arrange(Accuracy)

#Escogemos el que mayor accuracy tenga
tree <- tree; tree <- tree2;

#Variable importance
varimp_RF <- varImp(rf); varimp_RF
varimp_nb <- varImp(nb); varimp_nb
varimp_svm <- varImp(svm); varimp_svm
varimp_gbm <- varImp(gbm); varimp_gbm
varimp_knn <- varImp(knn) ;varimp_knn
varimp_tree <- varImp(tree) ;varimp_tree
plot(varimp_RF, main = "Variables más importantes") ; plot(varimp_nb, main = "Variables más importantes");
plot(varimp_svm, main = "Variables más importantes"); plot(varimp_knn, main = "Variables más importantes"); 
plot(varimp_tree, main = "Variables más importantes");


#Confusion Matrix
confusionMatrix.train(rf);
confusionMatrix.train(nb);
confusionMatrix.train(svm);
confusionMatrix.train(gbm);
confusionMatrix.train(knn);
confusionMatrix.train(tree);


#Prediction
fitted <- predict(rf_b1, testSet)
fitted_nb <- predict(nb, testSet)
fitted_svm <- predict(svm, testSet)
fitted_gbm <- predict(gbm, testSet)
fitted_knn <- predict(knn, testSet)
fitted_tree <- predict(tree, testSet)

result_rf <- confusionMatrix(reference = testSet$Pos, data= fitted, mode= "everything")
result_nb <- confusionMatrix(reference = testSet$Pos, data= fitted_nb, mode= "everything")
result_svm <- confusionMatrix(reference = testSet$Pos, data= fitted_svm, mode= "everything")
result_gbm <- confusionMatrix(reference = testSet$Pos, data= fitted_gbm, mode= "everything")
result_knn <- confusionMatrix(reference = testSet$Pos, data= fitted_knn, mode= "everything")
result_tree <- confusionMatrix(reference = testSet$Pos, data= fitted_tree, mode= "everything")

result_rf; result_nb; result_svm; result_gbm; result_knn; result_tree;

#Gráfico accuracy

overall.accuracy <- result_rf$overall
overall.accuracy_rf <- overall.accuracy["Accuracy"]

overall.accuracy <- result_nb$overall
overall.accuracy_nb <- overall.accuracy["Accuracy"]

overall.accuracy <- result_svm$overall
overall.accuracy_svm <- overall.accuracy["Accuracy"]

overall.accuracy <- result_gbm$overall
overall.accuracy_gbm <- overall.accuracy["Accuracy"]

overall.accuracy <- result_knn$overall
overall.accuracy_knn <- overall.accuracy["Accuracy"]

overall.accuracy <- result_tree$overall
overall.accuracy_tree <- overall.accuracy["Accuracy"]

accuracy_list <- c(overall.accuracy_rf,overall.accuracy_nb, overall.accuracy_svm, overall.accuracy_gbm, overall.accuracy_knn, overall.accuracy_tree)
accuracy_names <- c("RF","NB","SVM","GBM","KNN","RPART")
bar_df <- data.frame(accuracy_list,accuracy_names)
bar_df <- bar_df[order(bar_df$accuracy_list),]
ggplot(bar_df, aes(x=reorder(accuracy_names,-accuracy_list), y=accuracy_list)) + geom_bar(stat= "identity") + coord_flip()


