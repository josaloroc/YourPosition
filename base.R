#Read data
data <- read.csv2("data.csv", dec = ".")
data$Pos <- as.factor(data$Pos)
data <- data %>% select(-c(Rk, Player, Nation, Squad, Comp))

dim(data)
class(data)

#Set random seed
set.seed(30493)

#Install packages
install.packages(caret)
install.packages(tidyverse)
library(caret)
library(tidyverse)

#Vistazo inicial
str(data)
summary(data)

#Preprocessing
trainIndex <- createDataPartition(data$Pos, p= 0.7, list = FALSE)
trainingSet <- data[trainIndex,]
testSet <- data[-trainIndex,]

bagMissing <- preProcess(trainingSet, method = "bagImpute")
trainingSet <- predict(bagMissing, newdata = trainingSet)


#Training
rf <- train(Pos ~., data=trainingSet, method="rf", na.action = na.pass)
varimp_RF <- varImp(rf)
plot(varimp_RF, main = "Variables más importantes")

