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
fitControl <- trainControl(method="cv",number=10, savePredictions="final", classProbs= T, summaryFunction= multiClassSummary, allowParallel= TRUE)
rfGrid <- data.frame(mtry = c(3, 5, 7, 9, 10, 11, 12, 13, 15, 17, 19))
#Training
rf <- train(x, y, method="rf", data= data, metric="ROC", trControl= fitControl, tuneGrid = rfGrid)
nb <- train(x, y, method="naive_bayes", metric= "ROC", data= data, trControl = fitControl, tuneGrid = rfGrid)
svm <- train(Pos ~ ., data=trainingSet, method="svmRadial", metric= "ROC",trControl= fitControl,  tuneGrid = rfGrid)
gbm <- train(Pos ~ .,data=trainingSet, method="gbm",metric="ROC",trControl= fitControl,  tuneGrid = rfGrid)
knn <- train(x, y, method = "knn", trControl = fitControl, preProcess = c("center","scale"), tuneLength = 20, tuneGrid = rfGrid)
tree <- train(Pos ~ .,data=trainingSet, method="rpart2",metric="ROC",trControl= fitControl, tuneLength = 100, tuneGrid = rfGrid)
rf$resample
nb$resample
svm$resample
gbm$resample
knn$resample
tree$resample

#Variable importance
varimp_RF <- varImp(rf)
varimp_RF
varimp_nb <- varImp(nb)
varimp_nb
varimp_svm <- varImp(svm)
varimp_svm
varimp_gbm <- varImp(gbm)
varimp_gbm
varimp_knn <- varImp(knn)
varimp_knn
varimp_tree <- varImp(tree)
varimp_tree
plot(varimp_RF, main = "Variables más importantes")
plot(varimp_nb, main = "Variables más importantes")
plot(varimp_svm, main = "Variables más importantes")
plot(varimp_knn, main = "Variables más importantes")
plot(varimp_tree, main = "Variables más importantes")


#Confusion Matrix
confusionMatrix.train(rf)

#Prediction
fitted <- predict(rf, testSet)
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


