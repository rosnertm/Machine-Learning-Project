---
title: 'Prediction Assignment Write-Up: Predicting the Method of Exercise'
output:
  html_document:
    keep_md: yes
    self_contained: no
---



## Synopsis
The data used in this project are from [this website](http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har). Six participants were told to perform a bicep curl in five different ways:  

1. Completely correctly, according to the instructions given *(Class A)*  

2. Throwing their elbows forward *(Class B)*  

3. Lifting the dumbell only halfway *(Class C)*  

4. Lowering the dumbell only halfway *(Class D)*  

5. Throwing their hips forward *(Class E)*  

The goal of this project was to build a model that would correctly predict which class of exercise ("classe" in the data set) was being performed based on the information collected by the sensors participants were wearing on different parts of their body. 

## Initial setup and downloading of data
First, we need to load the packages necessary for data processing and visualization and check the data.

```r
library(tidyverse)
library(caret)
```

```r
full_data_set <- read.csv('pml-training.csv', na.strings = c("NA", "#DIV/0!"))

dim(full_data_set) 
```

```
## [1] 19622   160
```



The data set contains 19622 data points, with 160 columns. The last column, "classe" is the one we are hoping to predict. Most of the other columns appear to contain sensor information. 


```r
length(unique(full_data_set$X)) 
```

```
## [1] 19622
```

```r
unique(full_data_set$user_name)
```

```
## [1] carlitos pedro    adelmo   charles  eurico   jeremy  
## Levels: adelmo carlitos charles eurico jeremy pedro
```

This shows us that the X column is just a unique trial number; this does not contain information that would be helpful in a prediction model. In addition, the user_name variable refers to the identities of the six participants. Again, this information would not be helpful in predicting classes of exercise. These two columns can be removed. Finally, the next five columns contain information about trial time, which should not be stringly related to the type of exercise performed. These columns can all be removed from the data set.


```r
data_set_cols_rm <- full_data_set[,8:ncol(full_data_set)]
dim(data_set_cols_rm)
```

```
## [1] 19622   153
```

Of the columns left, many of them appear to contain NA values. Given that there are 153 columns, it might be useful to remove those that contain mostly NA values, as they won't be helpful for the final model.


```r
na_vals <- sapply(data_set_cols_rm, function(x) sum(is.na(x)))
unname(na_vals) ## total number of NA values in each column
```

```
##   [1]     0     0     0     0 19226 19248 19622 19225 19248 19622 19216
##  [12] 19216 19226 19216 19216 19226 19216 19216 19226 19216 19216 19216
##  [23] 19216 19216 19216 19216 19216 19216 19216     0     0     0     0
##  [34]     0     0     0     0     0     0     0     0     0 19216 19216
##  [45] 19216 19216 19216 19216 19216 19216 19216 19216     0     0     0
##  [56]     0     0     0     0     0     0 19294 19296 19227 19293 19296
##  [67] 19227 19216 19216 19216 19216 19216 19216 19216 19216 19216     0
##  [78]     0     0 19221 19218 19622 19220 19217 19622 19216 19216 19221
##  [89] 19216 19216 19221 19216 19216 19221     0 19216 19216 19216 19216
## [100] 19216 19216 19216 19216 19216 19216     0     0     0     0     0
## [111]     0     0     0     0     0     0     0 19300 19301 19622 19299
## [122] 19301 19622 19216 19216 19300 19216 19216 19300 19216 19216 19300
## [133]     0 19216 19216 19216 19216 19216 19216 19216 19216 19216 19216
## [144]     0     0     0     0     0     0     0     0     0     0
```

```r
## Lots of columns have more than 19000 NA values
## This is 97% of missing data in those columns and should probably be removed
missingData_df <- is.na(data_set_cols_rm)
rm_cols <- which(colSums(missingData_df) > 19000)
final_full_set <- data_set_cols_rm[, -rm_cols]
dim(final_full_set)
```

```
## [1] 19622    53
```

This leaves us with 53 columns (52 predictors) to work with!

## Splitting the Data
First, we need to split the data. Here, I have chosen to split the data into a training set, a testing set, and a final validation set (80/20/20). 


```r
set.seed(33433)
inBuild <- createDataPartition(y = final_full_set$classe, p = .8, list = F)

## create validation data set
validation_data <- final_full_set[-inBuild,]

## create (temp) building data set
buildData_set <- final_full_set[inBuild,]
## separate model building set into training and testing
inTrain <- createDataPartition(y = buildData_set$classe, p = .75, list = F)
train_data <- buildData_set[inTrain,]
test_data <- buildData_set[-inTrain,]

## check final sets for numbers
nrow(validation_data)
```

```
## [1] 3923
```

```r
nrow(test_data)
```

```
## [1] 3923
```

```r
nrow(train_data)
```

```
## [1] 11776
```

## Model Building
For this problem, I have chosen to use a random forest model, as these models are good for classification problems. First, we can start with a model that uses mostly default options.


```r
set.seed(3433)
basic_model <- train(classe ~ ., method = 'rf', data = train_data,
                     trControl = trainControl(method = 'cv'),
                     number = 3)
```

Next, we can build a tuned model. For this, we can adjust the mtry parameter (that is, the number of variables that are chosen at each node) and the ntree parameter (that is, the number of trees that are grown). For mtry, we can use the square root of the number of predictors (as demonstrated [here](https://machinelearningmastery.com/tune-machine-learning-algorithms-in-r/)). For ntree, we can use 1000, which is double the default value (500) but won't take too long to run.


```r
tunegrid <- expand.grid(.mtry=sqrt(ncol(train_data) - 1))
trcontrol <- trainControl(method="repeatedcv", number=10, repeats=3)
set.seed(3433)
tuned_model <- train(classe ~ ., data = train_data, method = 'rf', 
                     tuneGrid = tunegrid, 
                     trControl = trcontrol, 
                     ntree = 1000)
```

Let's compare the models and see which one worked better.

```r
basic_model
```

```
## Random Forest 
## 
## 11776 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 10598, 10599, 10597, 10599, 10599, 10599, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa
##    2    0.99      0.99 
##   27    0.99      0.99 
##   52    0.99      0.98 
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 2.
```

```r
tuned_model
```

```
## Random Forest 
## 
## 11776 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold, repeated 3 times) 
## Summary of sample sizes: 10598, 10599, 10597, 10599, 10599, 10599, ... 
## Resampling results:
## 
##   Accuracy  Kappa
##   0.99      0.99 
## 
## Tuning parameter 'mtry' was held constant at a value of 7.2
```

It appears as though the tuned model worked better. The tuned model has an accuracy of 99.21%, whereas the basic model has an accuracy of 99.1%. Moreover, the tuned model has a lower out-of-bag error rate of 0.61% than the basic model (out-of-bag error rate of 0.81%).  

Overall, it appears as though the tuned model might be better for a final model, but its performance could just be a result of overfitting. To check this, we can apply both models to the test set and see which one performs better.


```r
pred_basic <- predict(basic_model, test_data)
pred_tuned <- predict(tuned_model, test_data)


conf_mat_basic <- confusionMatrix(pred_basic, test_data$classe)
conf_mat_tuned <- confusionMatrix(pred_tuned, test_data$classe)

conf_mat_basic
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    3    0    0    0
##          B    0  754    7    0    0
##          C    0    2  675   14    3
##          D    0    0    2  629    4
##          E    0    0    0    0  714
## 
## Overall Statistics
##                                         
##                Accuracy : 0.991         
##                  95% CI : (0.988, 0.994)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.989         
##                                         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.993    0.987    0.978    0.990
## Specificity             0.999    0.998    0.994    0.998    1.000
## Pos Pred Value          0.997    0.991    0.973    0.991    1.000
## Neg Pred Value          1.000    0.998    0.997    0.996    0.998
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.192    0.172    0.160    0.182
## Detection Prevalence    0.285    0.194    0.177    0.162    0.182
## Balanced Accuracy       0.999    0.996    0.990    0.988    0.995
```

```r
conf_mat_tuned
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1115    3    0    0    0
##          B    1  753    7    0    0
##          C    0    3  677    7    2
##          D    0    0    0  636    6
##          E    0    0    0    0  713
## 
## Overall Statistics
##                                         
##                Accuracy : 0.993         
##                  95% CI : (0.989, 0.995)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.991         
##                                         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.999    0.992    0.990    0.989    0.989
## Specificity             0.999    0.997    0.996    0.998    1.000
## Pos Pred Value          0.997    0.989    0.983    0.991    1.000
## Neg Pred Value          1.000    0.998    0.998    0.998    0.998
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.192    0.173    0.162    0.182
## Detection Prevalence    0.285    0.194    0.176    0.164    0.182
## Balanced Accuracy       0.999    0.995    0.993    0.994    0.994
```

We can see that even with the test set, the tuned model (accuracy = 99.26%) 
outperforms the basic, default model (accuracy = 99.11%). We can apply the tuned model to a final validation set and see how well it performs.


```r
final_pred <- predict(tuned_model, validation_data)
conf_mat_final <- confusionMatrix(final_pred, validation_data$classe) 
conf_mat_final
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    4    0    0    0
##          B    0  750    3    0    0
##          C    0    5  680    7    2
##          D    0    0    1  636    4
##          E    0    0    0    0  715
## 
## Overall Statistics
##                                        
##                Accuracy : 0.993        
##                  95% CI : (0.99, 0.996)
##     No Information Rate : 0.284        
##     P-Value [Acc > NIR] : <2e-16       
##                                        
##                   Kappa : 0.992        
##                                        
##  Mcnemar's Test P-Value : NA           
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.988    0.994    0.989    0.992
## Specificity             0.999    0.999    0.996    0.998    1.000
## Pos Pred Value          0.996    0.996    0.980    0.992    1.000
## Neg Pred Value          1.000    0.997    0.999    0.998    0.998
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.191    0.173    0.162    0.182
## Detection Prevalence    0.285    0.192    0.177    0.163    0.182
## Balanced Accuracy       0.999    0.994    0.995    0.994    0.996
```

Within the final validation set, the chosen model performs with 99.34% accuracy.  

Therefore, the tuned model will be used to predict the classes for the final test set of 20 cases. 
