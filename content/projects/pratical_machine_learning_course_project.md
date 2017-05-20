+++
author = "Jun Wang"
categories = ["Machine Learning", "R"]
date = "2017-03-30"
description = ""
featured = ""
featuredalt = ""
featuredpath = ""
linktitle = ""
title = "Pratical Machine Learning Course Project"
type = "post"

+++

> This page is the submission for Coursera Pratical Machine Learning Course Project. 


# Introduction
## Background  

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).  

## Data  

The training data for this project are available here:  

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv  

The test data are available here:  

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv  

## Project Goal

The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.  


# Getting the Data  

```r
rm(list=ls())
library(caret)
library(tidyverse)
sessionInfo()
```

```
## R version 3.3.0 (2016-05-03)
## Platform: x86_64-w64-mingw32/x64 (64-bit)
## Running under: Windows 7 x64 (build 7601) Service Pack 1
## 
## locale:
## [1] LC_COLLATE=English_United States.1252 
## [2] LC_CTYPE=English_United States.1252   
## [3] LC_MONETARY=English_United States.1252
## [4] LC_NUMERIC=C                          
## [5] LC_TIME=English_United States.1252    
## 
## attached base packages:
## [1] stats     graphics  grDevices utils     datasets  methods   base     
## 
## other attached packages:
## [1] dplyr_0.5.0     purrr_0.2.2     readr_1.0.0     tidyr_0.6.0    
## [5] tibble_1.2      tidyverse_1.0.0 caret_6.0-73    ggplot2_2.2.0  
## [9] lattice_0.20-34
## 
## loaded via a namespace (and not attached):
##  [1] Rcpp_0.12.8        nloptr_1.0.4       plyr_1.8.4        
##  [4] iterators_1.0.8    tools_3.3.0        digest_0.6.10     
##  [7] lme4_1.1-12        evaluate_0.10      gtable_0.2.0      
## [10] nlme_3.1-128       mgcv_1.8-16        Matrix_1.2-7.1    
## [13] foreach_1.4.3      DBI_0.5-1          yaml_2.1.14       
## [16] parallel_3.3.0     SparseM_1.74       stringr_1.1.0     
## [19] knitr_1.15         MatrixModels_0.4-1 stats4_3.3.0      
## [22] rprojroot_1.2      grid_3.3.0         nnet_7.3-12       
## [25] R6_2.2.0           rmarkdown_1.3      minqa_1.2.4       
## [28] reshape2_1.4.2     car_2.1-3          magrittr_1.5      
## [31] backports_1.0.5    scales_0.4.1       codetools_0.2-15  
## [34] ModelMetrics_1.1.0 htmltools_0.3.5    MASS_7.3-45       
## [37] splines_3.3.0      assertthat_0.1     pbkrtest_0.4-6    
## [40] colorspace_1.3-0   quantreg_5.29      stringi_1.1.2     
## [43] lazyeval_0.2.0     munsell_0.4.3
```

```r
url1 <-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
url2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

# download.file(url1, destfile = "pml-training.csv")
# download.file(url2, destfile = "pml-testing.csv")

training <- read_csv("pml-training.csv")
final_testing <- read_csv("pml-testing.csv")
```

# Data Preprocessing and Cleaning
## Remove redundant variables
When looking at the data, it is obvious that many variables are mainly NAs and do not carry any useful information. Therefore two things were done in preprocessing. One is coerece all columns that were not imported as numeric to be numeric, this will ensure the modelling function properly run. The second thing was removing columns with mainly NAs. This turns out removed ~110 variables. 


```r
#str(training)

# training$classe %>% as.factor %>% str
# is.na(training$classe) %>% sum
# complete.cases(training) %>% sum
# t <- na.omit(training)



##ensure all numeric columns are indeed numeric
training[, 7:159] <- lapply(training[,7:159], as.numeric)
final_testing[, 7:159] <- lapply(final_testing[,7:159], as.numeric)
colnames_train <- colnames(training)


# check the number of non-NAs in each column.
check_NA <- function(x) {
    as.vector(apply(x, 2, function(x) length(which(!is.na(x)))))
}

# find columns to remove
col_na <- check_NA(training)
dp <- c()
for (i in 1:length(col_na)) {
    if (col_na[i] < nrow(training)) {
        dp <- c(dp, colnames_train[i])
    }
}

# remove NA columns and the first 7 columns
training <- training[, !(names(training) %in% dp)]
training <- training[, 8:ncol(training)]

final_testing <- final_testing[, !(names(final_testing) %in% dp)]
final_testing <- final_testing[, 8:ncol(final_testing)]
```

## Partion Training Data
Here I will do 70/30 split of the training data into train and test sets.


```r
set.seed(100)
t_idx <- createDataPartition(training$classe, p=0.7, list=F)
train <- training[t_idx,]
test <- training[-t_idx,]
```

# Train Random Forest Model 
Since Random Forest is known for its high accuracy, and I really don't mind spending long computing time, here I will just go for RF directly. 

First attemp was use `caret` package default bootstrap resampling to build a random forest model using the train set.   
Second attemp was using 10 fold cross-validation as the resampling method in `caret`.   

In order to speed up the model training, I used `doParallel` package for muiti-core computation and it works for windows machine!   

To avoid having to run the time-consuming modeling again when output the Rmarkdown to html, I saved the model objects on the hard drive using `saveRDS`. To `knit` the html file, I commented out the modelling codes, and used `readRDS` to load the model directly from hard drive.   

## Train RF with Bootstrap Resampling


```r
####register multicores for parallel processig####
library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
###############

####the model_building code below took very long to finish, therefore
###I saved the fitted model, without having to run it again
###when generate the Rmarkdown to html.
# try the default resampling method in caret 
set.seed(1001)
# md_rf_bt <- train(classe~., data=train,
#             method = "rf")
# 
# saveRDS(md_rf_bt, "md_rf_bt.rds")

##load the fitted random foest model
md_rf_bt <- readRDS("md_rf_bt.rds")

###Deregister multi-core###
stopCluster(cluster)
registerDoSEQ()
###
```


Now check the out of sample accuracy using the test set that we splitted before the modelling.  

```r
pred_bt <- predict(md_rf_bt, test)
confusionMatrix(pred_bt, test$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1672   10    0    0    0
##          B    1 1127    5    0    0
##          C    1    2 1018   15    1
##          D    0    0    3  948    1
##          E    0    0    0    1 1080
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9932          
##                  95% CI : (0.9908, 0.9951)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9914          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9988   0.9895   0.9922   0.9834   0.9982
## Specificity            0.9976   0.9987   0.9961   0.9992   0.9998
## Pos Pred Value         0.9941   0.9947   0.9817   0.9958   0.9991
## Neg Pred Value         0.9995   0.9975   0.9983   0.9968   0.9996
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2841   0.1915   0.1730   0.1611   0.1835
## Detection Prevalence   0.2858   0.1925   0.1762   0.1618   0.1837
## Balanced Accuracy      0.9982   0.9941   0.9941   0.9913   0.9990
```

Random Forest works really well on this dataset. This model achieved accuracy of 0.9932 on the test set.   
Do we really need to try cross-validation at all?  Well, for the sake of curisoity, let's give 10 fold cross-validation a spin. 

## Train RF with cross-validation

10-fold cross-validation is used as resampling method in `caret` `trControl`.  


```r
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

#10 cross-validation
set.seed(1002)
fitControl <- trainControl(## 10-fold CV
                           method = "cv",
                           number = 10,
                           allowParallel = T)

# md_rf_cv <- train(classe~., data=train, 
#              method = "rf",
#              trControl=fitControl)

#saveRDS(md_rf_cv, "md_rf_cv.rds")

md_rf_cv <- readRDS("md_rf_cv.rds")

###Deregister multi-core###
stopCluster(cluster)
registerDoSEQ()
###
```
Check the out of sample accuracy on test set.


```r
pred_cv <- predict(md_rf_cv, test)
confusionMatrix(pred_cv, test$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1672   10    0    0    0
##          B    1 1126    5    0    0
##          C    1    3 1019   14    1
##          D    0    0    2  949    1
##          E    0    0    0    1 1080
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9934         
##                  95% CI : (0.991, 0.9953)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9916         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9988   0.9886   0.9932   0.9844   0.9982
## Specificity            0.9976   0.9987   0.9961   0.9994   0.9998
## Pos Pred Value         0.9941   0.9947   0.9817   0.9968   0.9991
## Neg Pred Value         0.9995   0.9973   0.9986   0.9970   0.9996
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2841   0.1913   0.1732   0.1613   0.1835
## Detection Prevalence   0.2858   0.1924   0.1764   0.1618   0.1837
## Balanced Accuracy      0.9982   0.9937   0.9946   0.9919   0.9990
```

Well, the model built through 10 fold cross-validation has marginally higher accuracy (0.9934 vs 0.9932) versus the model built through bootstrap resampling. Hence I will use this model to continue on the test set. 

# Predict the 20 cases for quiz


```r
for (j in 1:20) {
  p <- predict(md_rf_cv, final_testing[j,])
  print(p)
}
```

```
## [1] B
## Levels: A B C D E
## [1] A
## Levels: A B C D E
## [1] B
## Levels: A B C D E
## [1] A
## Levels: A B C D E
## [1] A
## Levels: A B C D E
## [1] E
## Levels: A B C D E
## [1] D
## Levels: A B C D E
## [1] B
## Levels: A B C D E
## [1] A
## Levels: A B C D E
## [1] A
## Levels: A B C D E
## [1] B
## Levels: A B C D E
## [1] C
## Levels: A B C D E
## [1] B
## Levels: A B C D E
## [1] A
## Levels: A B C D E
## [1] E
## Levels: A B C D E
## [1] E
## Levels: A B C D E
## [1] A
## Levels: A B C D E
## [1] B
## Levels: A B C D E
## [1] B
## Levels: A B C D E
## [1] B
## Levels: A B C D E
```

