# Load necessary R packages into current R session
library(readr)
library(caret)
library(dplyr)

# Check if data files exist in working directory, if not: download the data
if(!file.exists("pml-training.csv")){
        URL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
        download.file(URL,
                      destfile = "pml-training.csv")
        rm(URL)
}

if(!file.exists("pml-testing.csv")){
        URL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
        download.file(URL,
                      destfile = "pml-testing.csv")
        rm(URL)
}

# Read the data into R
training <- read.csv("pml-training.csv", na.strings = c("NA","#DIV/0!",""))
validation <- read.csv("pml-testing.csv", na.strings = c("NA","#DIV/0!",""))

# look at data
head(training); str(training); summary(training)

# split data into train/test sets
set.seed(54321)
inTrain <- createDataPartition(y = training$classe, p = .75, list = FALSE)
trainset <- training[inTrain, ]
testset <- training[-inTrain, ]
rm(inTrain)
rm(training)

# coerce 'classe' variable into factor variable
trainset$classe <- as.factor(trainset$classe)
testset$classe <- as.factor(testset$classe)

# pre processing:
# remove unnecessary variables related to ID
trainset <- trainset[, -c(1:7)]

# check for NAs
sum(is.na(trainset))

# remove rows with greater than 50% NA's
mostlyNA <- sapply(trainset, is.na) %>%
        colMeans() > 0.5
trainset <- trainset[!mostlyNA]
rm(mostlyNA)

# check for NAs again
sum(is.na(trainset))

# check for any variables with nearly zero variance
nzv <- nearZeroVar(trainset, saveMetrics = TRUE)
head(nzv)
sum(nzv$nzv)
rm(nzv)

# check if remaining variables are highly correlated
m <- cor(trainset[,-length(trainset)])
diag(m) <- 0
which(m > 0.8, arr.ind = TRUE)
rm(m)

# do PCA and 5-fold cross validation
tc <- trainControl(method = "cv",
                   number = 5,
                   verboseIter = FALSE , 
                   preProcOptions = "pca",
                   allowParallel = TRUE)

# fit a decision tree model
model_tree <- train(classe ~., 
                    data = trainset, 
                    method = "rpart",
                    trControl = tc)

# fit a random forest model
model_rf <- train(classe ~., 
                  data = trainset, 
                  method = "rf",
                  trControl = tc)

# fit boosting model 
model_boost <- train(classe ~., 
                     data = trainset, 
                     method = "gbm",
                     trControl = tc,
                     verbose = FALSE)

# fit model with linear discrimnant analysis
model_lda <- train(classe ~.,
                   data = trainset,
                   method = "lda",
                   trControl = tc)

# take a look at accuracy of each model
accuracy <- cbind("Decision Tree" = mean(model_tree$results$Accuracy),
                  "Random Forest" = mean(model_rf$results$Accuracy),
                  "GBM Boost" = mean(model_boost$results$Accuracy),
                  "LDA" = mean(model_lda$results$Accuracy))
print(accuracy)
rm(accuracy)

# since the random forest model had the best mean performance, we'll use it to 
# make a prediction with our test set.
prediction_rf <- predict(model_rf, newdata = testset)
confusionMatrix(prediction_rf, testset$classe)

# attempt to accurately predict on the validation set
validate_rf <- predict(model_rf, validation)
validate_rf
