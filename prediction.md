---
title: "Practical Machine Learning Prediction Assignment"
author: "Matiki"
date: "October 2, 2018"
output: 
  html_document: 
    keep_md: yes
---



## Summary
In this assignment, we will be looking at data collected from 6 participants who
wore accelerometers while performing various exercises. The accelerometers were
worn on the participants' belt, arm, forarm, and dumbbell, while they performed
various lifts correctly and incorrectly in 5 different ways.

We will be building machine learning algorithms to make predictions on the manner
in which the exercises were performed. We will divide the data into training and
test sets, build 4 types of machine learning models on the training data: 
decision tree, random forest, boosting, and linear discirminant analysis. We will
use cross validation to help us select the best performing model, and estimate 
the out-of-sample error rate using the test set. Finally we will apply our best 
performing model to a set of 20 test cases for a final assessment. 

## Getting Data
The first thing to do is to download the data and read it into our current R 
session. The "NA" entries appear in several ways in the data set, and we include 
that information into our function call to read.csv().


```r
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
# head(training); str(training); summary(training)
```

## Cleaning Data & Preprocessing 
Next we need to split the data into training and test sets. For reproducibility,
we will set the seed, and then split 75% of the data into the training set, the 
rest will become the test set.


```r
# split data into train/test sets
set.seed(54321)
inTrain <- createDataPartition(y = training$classe, p = .75, list = FALSE)
trainset <- training[inTrain, ]
testset <- training[-inTrain, ]
rm(inTrain)
rm(training)
```

Next we'll turn the 'classe' variable, which we want to predict, into a factor 
variable, and get rid of the first several columns which are just for 
identification purposes, and won't be needed for our prediction.


```r
# coerce 'classe' variable into factor variable
trainset$classe <- as.factor(trainset$classe)
testset$classe <- as.factor(testset$classe)

# remove unnecessary variables related to ID
trainset <- trainset[, -c(1:7)]
```

We won't be able to use our machine learning algorithms if there are NA values,
so we'll need to handle them. We start by checking to see how many NAs we have.


```r
# check for NAs
sum(is.na(trainset))
```

```
## [1] 1443847
```

Since we have so many NAs, imputation may not be a very effective way of handling 
all the missing values. We'll just remove the variables that have mostly NAs, and 
then check how many NA values are left.


```r
# remove rows with greater than 50% NA's
mostlyNA <- sapply(trainset, is.na) %>%
        colMeans() > 0.5
trainset <- trainset[!mostlyNA]
rm(mostlyNA)

# check for NAs again
sum(is.na(trainset))
```

```
## [1] 0
```

It seems we got rid of all the NA values in our data set. Next we'll check to see
if any variables have close to zero variability. If they do, we might want to 
remove them from our data set before building our models.


```r
# check for any variables with nearly zero variance
nzv <- nearZeroVar(trainset, saveMetrics = TRUE)
head(nzv)
```

```
##                  freqRatio percentUnique zeroVar   nzv
## roll_belt         1.119403     7.7727952   FALSE FALSE
## pitch_belt        1.019481    11.6388096   FALSE FALSE
## yaw_belt          1.076923    12.4201658   FALSE FALSE
## total_accel_belt  1.068376     0.1902432   FALSE FALSE
## gyros_belt_x      1.092184     0.8832722   FALSE FALSE
## gyros_belt_y      1.149259     0.4416361   FALSE FALSE
```

```r
sum(nzv$nzv)
```

```
## [1] 0
```

```r
rm(nzv)
```

It seems we don't have any variables with near zero variance, so next we'll 
check to see if any of the variables are highly correlated.


```r
# check if remaining variables are highly correlated
m <- cor(trainset[,-length(trainset)])
diag(m) <- 0
which(m > 0.8, arr.ind = TRUE)
```

```
##                  row col
## yaw_belt           3   1
## total_accel_belt   4   1
## accel_belt_y       9   1
## roll_belt          1   3
## roll_belt          1   4
## accel_belt_y       9   4
## magnet_belt_x     11   8
## roll_belt          1   9
## total_accel_belt   4   9
## accel_belt_x       8  11
## magnet_arm_x      24  21
## accel_arm_x       21  24
## magnet_arm_z      26  25
## magnet_arm_y      25  26
## accel_dumbbell_x  34  28
## accel_dumbbell_z  36  29
## gyros_forearm_z   46  33
## pitch_dumbbell    28  34
## yaw_dumbbell      29  36
## gyros_forearm_z   46  45
## gyros_dumbbell_z  33  46
## gyros_forearm_y   45  46
```

```r
rm(m)
```

It seems there are several variables with high correlation, so we will account 
for this by adding some principal components analysis to our pre processing step.
We will also do a 5-fold cross validation. Both of these methods are done through
the function call to trainControl() which will be used when we build our models.


```r
# do PCA and 5-fold cross validation
tc <- trainControl(method = "cv",
                   number = 5,
                   verboseIter = FALSE , 
                   preProcOptions = "pca",
                   allowParallel = TRUE)
```

## Build the Models
Finally we are ready to build our models. We will use the train() function from
the caret package to build a decision tree, a random forest model, a boosting 
model, and a linear discriminant analysis model.


```r
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
```

## Evaluate & Compare Model Performance
Now that the models are built, it's time to take a look at how they compare in 
terms of accuracy.


```r
# take a look at accuracy of each model
accuracy <- cbind("Decision Tree" = mean(model_tree$results$Accuracy),
                  "Random Forest" = mean(model_rf$results$Accuracy),
                  "GBM Boost" = mean(model_boost$results$Accuracy),
                  "LDA" = mean(model_lda$results$Accuracy))
print(accuracy)
```

```
##      Decision Tree Random Forest GBM Boost       LDA
## [1,]      0.425557     0.9901254  0.880049 0.7011139
```

```r
rm(accuracy)
```

It looks like the random forest model produced the greatest accuracy. We will 
use this model on our test set to estimate the out of sample error for our model.

## Evaluate Out-Of-Sample Error


```r
# since the random forest model had the best mean performance, we'll use it to 
# make a prediction with our test set.
prediction_rf <- predict(model_rf, newdata = testset)
confusionMatrix(prediction_rf, testset$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    7    0    0    0
##          B    0  942    8    0    0
##          C    0    0  847   12    0
##          D    0    0    0  792    4
##          E    0    0    0    0  897
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9937         
##                  95% CI : (0.991, 0.9957)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.992          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9926   0.9906   0.9851   0.9956
## Specificity            0.9980   0.9980   0.9970   0.9990   1.0000
## Pos Pred Value         0.9950   0.9916   0.9860   0.9950   1.0000
## Neg Pred Value         1.0000   0.9982   0.9980   0.9971   0.9990
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2845   0.1921   0.1727   0.1615   0.1829
## Detection Prevalence   0.2859   0.1937   0.1752   0.1623   0.1829
## Balanced Accuracy      0.9990   0.9953   0.9938   0.9920   0.9978
```

So it seems our out-of-sample error rate is about 0.63% in our random forest model.

## Predict on Validation Set
Our final step is to use our model to make a prediction on a new set of 20 cases.
This is what will be used for the final assessment of our model and this assignment.


```r
# attempt to accurately predict on the validation set
validate_rf <- predict(model_rf, validation)
validate_rf
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
