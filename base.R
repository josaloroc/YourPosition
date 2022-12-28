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
fitControl_1 <- trainControl(method="cv",number=10, savePredictions="final",search= "random", allowParallel= TRUE)
fitControl_2 <- trainControl(method="repeatedcv",number=10, repeats=10, search= "random", savePredictions="final", allowParallel= TRUE)


#Training
#Random forest
bestMtry1000 <- tuneRF(x,y, stepFactor = 1.5, improve = 1e-5, ntree = 1000)
bestMtry1500 <- tuneRF(x,y, stepFactor = 1.5, improve = 1e-5, ntree = 1500)
bestMtry2000 <- tuneRF(x,y, stepFactor = 1.5, improve = 1e-5, ntree = 2000)

print(bestMtry1000); print(bestMtry1500); print(bestMtry2000)

grid1000 <- data.frame(mtry = 24); grid1500 <- data.frame(mtry = 24); grid2000 <- data.frame(mtry = 16)

-----------------------------------------------------------------------------------------------------------------------------------------------------
rf_b <- train(x, y, method="rf", data= data, metric="Accuracy", ntree=1000, tuneGrid = grid1000) #0.7786355
rf_b1 <- train(x, y, method="rf", data= data, metric="Accuracy", ntree=1000, trControl= fitControl_1, tuneGrid = grid1000) #0.7915388
rf_b2 <- train(x, y, method="rf", data= data, metric="Accuracy", ntree=1000, trControl= fitControl_2, tuneGrid = grid1000)#0.7870106

rf_b$resample; print(rf_b)
rf_b1$resample; print(rf_b1)
rf_b2$resample; print(rf_b2)


bAccuracy <- rf_b$results$Accuracy; b1Accuracy <- rf_b1$results$Accuracy; b2Accuracy <- rf_b2$results$Accuracy
b_accuracy_list <- c(bAccuracy, b1Accuracy, b2Accuracy)
plot(b_accuracy_list, lty=1, lwd=10)
-----------------------------------------------------------------------------------------------------------------------------------------------------
  
rf_c <- train(x, y, method="rf", data= data, metric="Accuracy", ntree=1500, tuneGrid = grid1500) #0.7827391
rf_c1 <- train(x, y, method="rf", data= data, metric="Accuracy", ntree=1500, trControl= fitControl_1, tuneGrid = grid1500) #0.7887743
rf_c2 <- train(x, y, method="rf", data= data, metric="Accuracy", ntree=1500, trControl= fitControl_2, tuneGrid = grid1500)#0.7889702

rf_c$resample; print(rf_c)
rf_c1$resample; print(rf_c1)
rf_c2$resample; print(rf_c2)

cAccuracy <- rf_c$results$Accuracy; c1Accuracy <- rf_c1$results$Accuracy; c2Accuracy <- rf_c2$results$Accuracy
c_accuracy_list <- c(cAccuracy, c1Accuracy, c2Accuracy)
mean_c <- mean(c_accuracy_list)
plot(c_accuracy_list, lty=1, lwd=10)
-----------------------------------------------------------------------------------------------------------------------------------------------------
  

rf_d <- train(x, y, method="rf", data= data, metric="Accuracy", ntree=2000, tuneGrid = grid2000) #0.7801005
rf_d1 <- train(x, y, method="rf", data= data, metric="Accuracy", ntree=2000, trControl= fitControl_1, tuneGrid = grid2000) #0.789384
rf_d2 <- train(x, y, method="rf", data= data, metric="Accuracy", ntree=2000, trControl= fitControl_2, tuneGrid = grid2000)#0.7868483

rf_d$resample; print(rf_d)
rf_d1$resample; print(rf_d1)
rf_d2$resample; print(rf_d2)

dAccuracy <- rf_d$results$Accuracy; d1Accuracy <- rf_d1$results$Accuracy; d2Accuracy <- rf_d2$results$Accuracy
d_accuracy_list <- c(dAccuracy, d1Accuracy, d2Accuracy)
mean_d <- mean(d_accuracy_list)
plot(d_accuracy_list, lty=1, lwd=10)
-----------------------------------------------------------------------------------------------------------------------------------------------------

plot(c(max(b_accuracy_list),max(c_accuracy_list),max(d_accuracy_list)), lty=1, lwd=10)

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

nb_list <- c(nb$results$Accuracy);nb_list2 <- c(nb2$results$Accuracy)
plot(nb_list, lty=1, lwd= 10); plot(nb_list2, lty=1, lwd= 10)
plot(c(max(nb_list),max(nb_list2)), lty=1, lwd=10)

#Support Vector Machine
svm <- train(Pos ~ ., data=trainingSet, method="svmRadial", metric= "ROC",trControl= fitControl,  tuneLength=20)

#GBM
gbm <- train(Pos ~ .,data=trainingSet, method="gbm",metric="ROC",trControl= fitControl)

#KNN
knn <- train(x, y, method = "knn", metric= "Accuracy", preProcess = c("center","scale"), trControl = fitControl_1, tuneLength = 20)
knn2 <- train(x, y, method = "knn", metric= "Accuracy", preProcess = c("center","scale"), trControl = fitControl_2, tuneLength = 20)
plot(knn); plot(knn2)

knn_list <- c(knn$results$Accuracy)
knn_list2 <- c(knn2$results$Accuracy)
plot(knn_list, lty=1, lwd=10); plot(knn_list2, lty=1, lwd=10)
plot(c(max(knn_list),max(knn_list2)), lty=1, lwd=10)


#RPART
tree <- train(Pos ~ .,data=trainingSet, method="rpart2",metric="ROC",trControl= fitControl, tuneLength = 100)


svm$resample
gbm$resample
knn$resample
tree$resample

#Variable importance
varimp_RF <- varImp(rf); varimp_RF
varimp_nb <- varImp(nb); varimp_nb
varimp_svm <- varImp(svm); varimp_svm
varimp_gbm <- varImp(gbm); varimp_gbm
varimp_knn <- varImp(knn) ;varimp_knn
varimp_tree <- varImp(tree) ;varimp_tree
plot(varimp_RF, main = "Variables más importantes") ; plot(varimp_nb, main = "Variables más importantes");
plot(varimp_svm, main = "Variables más importantes"); plot(varimp_knn, main = "Variables más importantes"); 
plot(varimp_tree, main = "Variables más importantes")


#Confusion Matrix
confusionMatrix.train(rf)

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
result_tree

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


