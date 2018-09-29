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
testing <- read.csv("pml-testing.csv", na.strings = c("NA","#DIV/0!",""))

# look at data
head(training); str(training); summary(training)

# split data into train/test sets
set.seed(54321)
inTrain <- createDataPartition(y = training$classe, p = .75, list = FALSE)
trainset <- training[inTrain, ]
testset <- training[-inTrain, ]
rm(inTrain)

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

# do PCA and knn imputation to remove remaining NAs and 5-fold cross validation
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
