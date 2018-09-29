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
training <- read_csv("pml-training.csv")
testing <- read_csv("pml-testing.csv")

# look at data
head(training); str(training); summary(training)

# split data into train/test sets
set.seed(54321)
inTrain <- createDataPartition(y = training$classe, p = .75, list = FALSE)
trainset <- training[inTrain, ]
testset <- training[-inTrain, ]
rm(inTrain)

# pre processing:
# remove unnecessary variables related to ID
trainset <- trainset[, -c(1:7)]

# remove variables with nearly zero variance
nzv <- nearZeroVar(trainset, saveMetrics = TRUE)
trainset <- trainset[, !nzv$nzv]
rm(nzv)

# remove rows with greater than 50% NA's
mostlyNA <- sapply(trainset, is.na) %>%
        colMeans() > 0.5
trainset <- trainset[!mostlyNA]

# fit a decision tree model
model_tree <- train(classe ~., data = trainset, method = "rpart")

# fit a random forest model
model_rf <- train(classe ~., data = trainset, method = "rf")

# fit boosting model 
model_boost <- train(classe ~., data = trainset, method = "gbm")

