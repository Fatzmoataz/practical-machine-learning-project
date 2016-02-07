## It is now possible to collect a large amount of data about personal activity using
## devices such that Jawbone Up. People usually use these devices to quantify how much
## of a particular activity they do. We are going to use the data to quantify how well
## they do an activity.
## Six young health participants were asked to perform one set of 10 repetitions of the
## Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the 
## specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell 
## only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips 
## to the front (Class E).
## Class A corresponds to the specified execution of the exercise, while the other 4 classes 
## correspond to common mistakes.
## We are going to use the collected data from accelerometers on the belt, forearm, arm, 
## and dumbell to build a predictor model for the class of the activity


## The question we care about, can we predict how well the participants perform 
## weight lift? in other words, can we predict the classe variable?

## The data is obtained from http://groupware.les.inf.puc-rio.br/har


library(caret)
library(randomForest)
library(rpart)
library(rpart.plot)


########### Downloading the data
## create a data folder
if (!file.exists("data")) {dir.create("data")}
## download the training data
trainURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
if (!file.exists("./data/training.csv")) {download.file(trainURL, destfile = "./data/training.csv", method = "curl")}
## download the testing data
testURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
if (!file.exists("./data/testing.csv")){download.file(testURL, destfile = "./data/testing.csv", method = "curl")}

########### Reading the data
training <- read.csv("./data/training.csv", header = TRUE)
training <- training[,-1]  # omit first column since it has row number

testing <- read.csv("./data/testing.csv", header = TRUE)
testing <- testing[,-1]  # omit first column since it has row number

########## Checking the data
dim(training)
head(training)
names(training)

dim(testing)
head(testing)
names(testing)
# We have 19622 observations in the training set and only 20 observations in the testing set
# we need to do cross-validation

########## Predictors
### Checking and removing the variables with near zero variance
nzv <- nearZeroVar(training, saveMetrics=TRUE)
training <- training[,nzv$nzv==FALSE]
testing <- testing[,nzv$nzv==FALSE]

### Removing columns where more than half the values are NA
training <- training[, colSums(is.na(training)) < dim(training)[1]/2] 
testing <- testing[, colSums(is.na(testing)) < dim(testing)[1]/2]
# At this point we are left with 58 variables
names(training)

### Removing useless columns, the ones containing user name, timestamp and window
training <- training[, !grepl("name|timestamp|window", names(training))]
testing <- testing[, !grepl("name|timestamp|window", names(training))]
# At this point we are left with 53 variables


########## Partitioning the data
## Partitioning the training set into training and test
inTrain <- createDataPartition(y = training$classe, p = 0.75, list = FALSE)
trainingData <- training[inTrain,]
testingData <- training[-inTrain,]


########## Creating fitting models
## First model: using cross validation and random forests
set.seed(1234)
controlRf <- trainControl(method="cv", 5)  #using cross validation with 5 folds
modFit1 <- train(classe ~ ., data=trainingData, method="rf", trControl=controlRf,  ntree = 200 )
finalMod1 <- modFit1$finalModel

## Second model: using cross validation and partition trees
set.seed(1234)
controlRf <- trainControl(method="cv", 5) 
modFit2 <- train(classe ~ ., data=trainingData, method="rpart", trControl=controlRf)
finalMod2 <- modFit2$finalModel
#plotting the tree
plot(finalMod2,uniform= TRUE,main="classification tree")
text(finalMod2,use.n=TRUE,all=TRUE,cex=0.8)

## Third model: using cross validation and boosting
set.seed(1234)
controlRf <- trainControl(method="cv", 5) 
modFit3 <- train(classe ~ ., data=trainingData, method="gbm", trControl=controlRf, verbose = FALSE)
finalMod3 <- modFit3$finalModel



########## Validating the models with the testingData
testClasse <- testingData$classe
## First Model
pred1 <- predict(modFit1,testingData)
conf1 <- confusionMatrix(pred1,testClasse)
accuracy1 <- conf1$overall[[1]]
error1 <- 1 - accuracy1
## Second Model
pred2 <- predict(modFit2,testingData)
conf2 <- confusionMatrix(pred2,testClasse)
accuracy2 <- conf2$overall[[1]]
error2 <- 1 - accuracy2
## Third Model
pred3 <- predict(modFit3,testingData)
conf3 <- confusionMatrix(pred3,testClasse)
accuracy3 <- conf3$overall[[1]]
error3 <- 1 - accuracy3

