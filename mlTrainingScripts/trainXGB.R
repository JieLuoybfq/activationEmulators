rm(list=ls())
library(caret)
library(useful)
library(InformationValue)
library(tidyverse)
library(data.table)
library(xgboost)

# Load Datasets
inFile <- 'parcelOutput250BinsWithTwomey_111920.csv'
x <- data.table(read_csv(inFile))
summary(x)

inFile <- 'parcelOutput250BinsWithTwomeyPlus4K_111920.csv'
gen.dt <- data.table(read_csv(inFile))
summary(gen.dt)

inFile <- 'parcelOutput250BinsWithTwomeySens_111920.csv'
sens.dt <- data.table(read_csv(inFile))
summary(sens.dt)

###
# Preprocessing
###
set.seed(072590)
trainIdx <- 1:14000
valIdx <- 14001:16000
testIdx <- 16001:dim(x)[1]

###
# ActFrac Parcel Boosting
###
# 
# Twomey
# formulaPred <- delta_actFrac ~ . -1
# cols <- c('delta_actFrac', "Log10N", "Log10ug", "Sigma_g", "Kappa", "Log10V", "T", "P",
#           'ac','actFrac_Twomey',"Smax_Twomey")

# Naive
# formulaPred <- actFrac ~ . -1
# cols <- c('actFrac', "Log10N", "Log10ug", "Sigma_g", "Kappa", "Log10V", "T", "P",
#           'ac')

# ARG Scheme
x$delta_actFrac_arg <- x$actFrac - x$actFrac_arg
gen.dt$delta_actFrac_arg <- gen.dt$actFrac - gen.dt$actFrac_arg
sens.dt$delta_actFrac_arg <- sens.dt$actFrac - sens.dt$actFrac_arg

formulaPred <- delta_actFrac_arg ~ . -1
cols <- c('delta_actFrac_arg', "Log10N", "Log10ug", "Sigma_g", "Kappa", "Log10V", "T", "P",
          'ac','actFrac_arg',"smaxes_arg")

m.learn.dt <- x

m.train.dt <- m.learn.dt[trainIdx,]
m.val.dt <- m.learn.dt[valIdx,]
m.test.dt <- m.learn.dt[testIdx,]
m.gen.dt <- gen.dt
m.sens.dt <- sens.dt

###
# Normalization
###
trainX <- build.x(formulaPred,m.train.dt[,..cols],contrasts = F)
trainY <- build.y(formulaPred,m.train.dt[,..cols])

normParam.X <- preProcess(trainX)
mean.Y <- mean(trainY)
sd.Y <- sd(trainY)

trainX <- predict(normParam.X,trainX)
trainY <- (trainY-mean.Y)/sd.Y

valX <- predict(normParam.X,build.x(formulaPred,m.val.dt[,..cols],contrasts = F))
valY <- (build.y(formulaPred,m.val.dt[,..cols])-mean.Y)/sd.Y

testX <- predict(normParam.X,build.x(formulaPred,m.test.dt[,..cols],contrasts = F))
testY <- (build.y(formulaPred,m.test.dt[,..cols])-mean.Y)/sd.Y

genX <- predict(normParam.X,build.x(formulaPred,m.gen.dt[,..cols],contrasts = F))
genY <- (build.y(formulaPred,m.gen.dt[,..cols])-mean.Y)/sd.Y

sensX <- predict(normParam.X,build.x(formulaPred,m.sens.dt[,..cols],contrasts = F))
sensY <- (build.y(formulaPred,m.sens.dt[,..cols])-mean.Y)/sd.Y

###
# XGBoost Regression Training
###
dtrain <- xgb.DMatrix(data = trainX, label=trainY)
dval <- xgb.DMatrix(data = valX, label=valY)

watchlist <- list(train=dtrain, val=dval)
param <- list(max_depth = 6, eta = 0.1)
bst <- xgb.train(params=param,
                 data=dtrain, nrounds=1500, 
                 watchlist=watchlist,early_stopping_rounds=25) 


# Make Predictions
m.train.dt$bst.actFrac <- predict(bst, trainX)* sd.Y + mean.Y
m.val.dt$bst.actFrac <- predict(bst, valX)* sd.Y + mean.Y
m.test.dt$bst.actFrac <- predict(bst, testX)* sd.Y + mean.Y
m.gen.dt$bst.actFrac <- predict(bst, genX)* sd.Y + mean.Y
m.sens.dt$bst.actFrac <- predict(bst, sensX)* sd.Y + mean.Y


# Evaluate Performance
v1 <- m.test.dt$actFrac
v2 <- m.test.dt$bst.actFrac + m.test.dt$actFrac_arg
# v2[v2 > 1] <- 1
# v2[v2 < 0] <- 0

mean((v1-v2)^2)
cor(v1,v2)^2

ggplot(m.test.dt,aes(x=actFrac,y=bst.actFrac + actFrac_arg)) + geom_point() + geom_abline()
