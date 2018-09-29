# Load necessary R packages into current R session
library(readr)
library(caret)

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
inTrain <- createDataPartition(y = training$classe, p = .75, list = FALSE)
train <- training[inTrain]
test <- training[-inTrain]

# fit a decision tree model
model_tree <- train(classe ~., data = training, method = "rpart")

# fit a random forest model
model_rf <- train(classe ~., data = training, method = "rf")

# fit boosting model 
model_boost <- train(classe ~., data = training, method = "gbm")

