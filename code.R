library(tidyverse)
library(modelr)
library(glmnet)
library(glmnetUtils)
library(readr)
library(ISLR)
library(randomForest)
library(gridExtra)
library(ggplot2)
library(dplyr)
library(grid)
library(data.table)
library(wesanderson)
library(curl)

################################## latitude as response variable ##################################
###### Question 2 #####
# read data and delete NAs
d=read.csv("https://raw.githubusercontent.com/siliweibaruch/STA-9890-project/main/default_features_1059_tracks.txt",header=F)
names(d) = tolower(names(d))
d = d %>% filter(!is.na(v69|v70))

# define the response and predictors 
y = d$v69
X = d%>%select(-v69&-v70)
X = data.matrix(X)

# count the sample size and features
n = nrow(d)
p = ncol(X)

##### Question 3 #####
# use a loop to run the procedure for 100 times
# including split the dataset into train and test set
# fit the train data to lasso, elastic-net alpha=0.5, ridge and random forest
# set lambda as the result of 10-fold CV
# record both the test R squared and train R squared

# train set include 80% of the data
n.train = floor(0.8*n)

# test set include the rest of the data
n.test = n-n.train

# run the loop 100 times
M = 100

# set up several empty vectors for the R squared
Rsq.test.lasso = rep(0,M)
Rsq.train.lasso = rep(0,M)
Rsq.test.en = rep(0,M)
Rsq.train.en = rep(0,M)
Rsq.test.ridge = rep(0,M)
Rsq.train.ridge = rep(0,M)
Rsq.test.rf = rep(0,M)
Rsq.train.rf = rep(0,M)

for (m in c(1:M)) {
  # if not using caret package, there will be an error
  # divide both response and predictor matrix into test and train sets randomly
  library(caret)
  shuffled_indexes = sample(n)
  train = shuffled_indexes[1:n.train]
  test = shuffled_indexes[(1+n.train):n]
  X.train = X[train, ]
  y.train = y[train]
  X.test = X[test, ]
  y.test = y[test]
  
  # use the lambda chosen by the cross-validation to fit lasso and record the estimated coefficients and R-squared
  lasso.cv.fit = cv.glmnet(X.train, y.train, alpha = 1, nfolds = 10)
  lasso.fit = glmnet(X.train, y.train, alpha = 1, lambda = lasso.cv.fit$lambda.min)
  y.train.hat = predict(lasso.fit, newx = X.train, type = "response") 
  y.test.hat = predict(lasso.fit, newx = X.test, type = "response")  
  Rsq.test.lasso[m] = 1-mean((y.test - y.test.hat)^2)/mean((y.test - mean(y.test))^2)
  Rsq.train.lasso[m] = 1-mean((y.train - y.train.hat)^2)/mean((y.train - mean(y.train))^2)
  
  # use the lambda chosen by the cross-validation to fit elastic-net and record the estimated coefficients and R-squared  
  en.cv.fit = cv.glmnet(X.train, y.train, alpha = 0.5, nfolds = 10)
  en.fit = glmnet(X.train, y.train, alpha = 0.5, lambda = en.cv.fit$lambda.min)
  y.train.hat = predict(en.fit, newx = X.train, type = "response")
  y.test.hat = predict(en.fit, newx = X.test, type = "response") 
  Rsq.test.en[m] = 1-mean((y.test - y.test.hat)^2)/mean((y.test - mean(y.test))^2)
  Rsq.train.en[m] = 1-mean((y.train - y.train.hat)^2)/mean((y.train - mean(y.train))^2)
  
  # use the lambda chosen by the cross-validation to fit ridge and record the estimated coefficients and R-squared 
  ridge.cv.fit = cv.glmnet(X.train, y.train, alpha = 0, nfolds = 10)
  ridge.fit = glmnet(X.train, y.train, alpha = 0, lambda = ridge.cv.fit$lambda.min)
  y.train.hat = predict(ridge.fit, newx = X.train, type = "response") 
  y.test.hat = predict(ridge.fit, newx = X.test, type = "response") 
  Rsq.test.ridge[m] = 1-mean((y.test - y.test.hat)^2)/mean((y.test - mean(y.test))^2)
  Rsq.train.ridge[m] = 1-mean((y.train - y.train.hat)^2)/mean((y.train - mean(y.train))^2)
  
  # fit random forest and record the estimated coefficients and R-squared 
  rf.fit = randomForest(X.train, y.train, mtry = sqrt(p), importance = TRUE)
  y.test.hat = predict(rf.fit, X.test)
  y.train.hat = predict(rf.fit, X.train)
  Rsq.test.rf[m] = 1-mean((y.test - y.test.hat)^2)/mean((y.test - mean(y.test))^2)
  Rsq.train.rf[m] = 1-mean((y.train - y.train.hat)^2)/mean((y.train - mean(y.train))^2)
  
}

##### Question 4 #####
# side-by-side boxplots of test R squared and train R squared
par(mfrow = c(1,2))
boxplot(Rsq.train.lasso, Rsq.train.en, Rsq.train.ridge, Rsq.train.rf,
        main = "TRAIN SET R-SQUARED",
        names = c("LASSO", "EN", "RIDGE", "RF"),
        col = c("red","blue", "yellow", "green"))

boxplot(Rsq.test.lasso, Rsq.test.en, Rsq.test.ridge, Rsq.test.rf,
        main = "TEST SET R-SQUARED",
        names = c("LASSO", "EN", "RIDGE", "RF"),
        col = c("red","blue", "yellow", "green"))

# For one on the 100 sample, plot 10-fold CV curves. Record and present the time
# 10-fold cross-validation for lasso
par(mfrow = c(1,1))
ptm = proc.time()
lasso.cv = cv.glmnet(X.train, y.train, alpha = 1, nfolds = 10)
ptm = proc.time() - ptm
time_lasso = ptm["elapsed"]
cat(sprintf("Run Time for Lasso: %0.3f(sec):",time_lasso))
plot(lasso.cv) + title("10-fold CV curve for Lasso", line = 2.5)

# 10-fold cross-validation for elastic-net
ptm = proc.time()
en.cv = cv.glmnet(X.train, y.train, alpha = 0.5, nfolds = 10)
ptm = proc.time() - ptm
time_en = ptm["elapsed"]
cat(sprintf("Run Time for Elastic-net: %0.3f(sec):",time_en))
plot(en.cv) + title("10-fold CV curve for Elastic-net", line = 2.5)

# 10-fold cross-validation for ridge
ptm = proc.time()
ridge.cv = cv.glmnet(X.train, y.train, alpha = 0, nfolds = 10)
ptm = proc.time() - ptm
time_ridge = ptm["elapsed"]
cat(sprintf("Run Time for ridge: %0.3f(sec):",time_ridge))
plot(ridge.cv) + title("10-fold CV curve for Ridge", line = 2.5)

# side-by-side boxplots of train and test residuals lasso
cv.fit <- cv.glmnet(X.train, y.train, alpha = 1, nfolds = 10)
fit <- glmnet(X.train, y.train, alpha = 1, lambda = cv.fit$lambda.min)
y.train.hat <- predict(fit, newx = X.train, type = "response") 
y.test.hat <- predict(fit, newx = X.test, type = "response")  
Res.test.la <- y.test - y.test.hat
Res.test.la <- as.vector(Res.test.la)
Res.train.la <- y.train - y.train.hat
Res.train.la <- as.vector(Res.train.la)

# side-by-side boxplots of train and test residuals elastic-net
cv.fit <- cv.glmnet(X.train, y.train, alpha = 0.5, nfolds = 10)
fit <- glmnet(X.train, y.train, alpha = 0.5, lambda = cv.fit$lambda.min)
y.train.hat <- predict(fit, newx = X.train, type = "response") 
y.test.hat <- predict(fit, newx = X.test, type = "response")  
Res.test.en <- y.test - y.test.hat
Res.test.en <- as.vector(Res.test.en)
Res.train.en <- y.train - y.train.hat
Res.train.en <- as.vector(Res.train.en)

# side-by-side boxplots of train and test residuals ridge
cv.fit <- cv.glmnet(X.train, y.train, alpha = 0, nfolds = 10)
fit <- glmnet(X.train, y.train, alpha = 0, lambda = cv.fit$lambda.min)
y.train.hat <- predict(fit, newx = X.train, type = "response") 
y.test.hat <- predict(fit, newx = X.test, type = "response")  
Res.test.ri <- y.test - y.test.hat
Res.test.ri <- as.vector(Res.test.ri)
Res.train.ri <- y.train - y.train.hat
Res.train.ri <- as.vector(Res.train.ri)

# side-by-side boxplots of train and test residuals random forest
rf <- randomForest(X.train, y.train, mtry = sqrt(p), importance = TRUE)
y.test.hat <- predict(rf, X.test)
y.train.hat <- predict(rf, X.train)
Res.test.rf <- y.test - y.test.hat
Res.train.rf <- y.train - y.train.hat

par(mfrow=c(1,2))
boxplot(Res.train.la, Res.train.en, Res.train.ri, Res.train.rf,
        main = "RESIDUALS OF TRAIN SET",
        names = c("LASSO", "EN", "RIDGE", "RF"),
        col = c("red","blue", "yellow", "green"))

boxplot(Res.test.la, Res.test.en, Res.test.ri, Res.test.rf,
        main = "RESIDUALS OF TEST SET",
        names = c("LASSO", "EN", "RIDGE", "RF"),
        col = c("red","blue", "yellow", "green"))

##### Question 5 #####
# use 10-fold cross validation to fit ridge, lasso, elastic-net and random forest
# lasso
la_start = Sys.time()
cv.la = cv.glmnet(X, y, alpha = 1, nfolds = 10)
la = glmnet(X, y, alpha = 1, lambda = cv.la$lambda.min)
la_end = Sys.time()
la_time = la_end - la_start
lasso.Rsq.ci = t.test(Rsq.test.lasso, conf.level = 0.9)

# elastic-net
en_start = Sys.time()
cv.en = cv.glmnet(X, y, alpha = 0.5, nfolds = 10)
en = glmnet(X, y, alpha = 0.5, lambda = cv.en$lambda.min)
en_end = Sys.time()
en_time = en_end - en_start
en.Rsq.ci = t.test(Rsq.test.en, conf.level = 0.9)

# ridge
ri_start = Sys.time()
cv.ri = cv.glmnet(X, y, alpha = 0, nfolds = 10)
ri = glmnet(X, y, alpha = 0, lambda = cv.ri$lambda.min)
ri_end = Sys.time()
ri_time = ri_end - ri_start
ridge.Rsq.ci = t.test(Rsq.test.ridge, conf.level = 0.9)

# random forest
rf_start = Sys.time()
rf = randomForest(X, y, mtry = sqrt(p), importance = TRUE)
rf_end = Sys.time()
rf_time = rf_end - rf_start
rf.Rsq.ci = t.test(Rsq.test.rf, conf.level = 0.9)

lasso.Rsq.ci$conf.int[1:2]
en.Rsq.ci$conf.int[1:2]
ridge.Rsq.ci$conf.int[1:2]
rf.Rsq.ci$conf.int[1:2]
la_time
en_time
ri_time
rf_time

# bar-plots of the estimated coefficients
# elastic-net
cv.fit = cv.glmnet(X, y, alpha = 0.5, nfolds = 10)
fit.en = glmnet(X, y, alpha = 0.5, lambda = cv.fit$lambda.min)

# lasso
cv.fit = cv.glmnet(X, y, alpha = 1, nfolds = 10)
fit.ls = glmnet(X, y, alpha = 1, lambda = cv.fit$lambda.min)

# ridge
cv.fit = cv.glmnet(X, y, alpha = 0, nfolds = 10)
fit.ri = glmnet(X, y, alpha = 0, lambda = cv.fit$lambda.min)

# random forest
fit.rf = randomForest(X, y, mtry = sqrt(p), importance = TRUE)

betaS.en = data.frame(c(1:p), as.vector(fit.en$beta))
colnames(betaS.en) = c( "feature", "value")

betaS.ls = data.frame(c(1:p), as.vector(fit.ls$beta))
colnames(betaS.ls) = c( "feature", "value")

betaS.ri = data.frame(c(1:p), as.vector(fit.ri$beta))
colnames(betaS.ri) = c( "feature", "value")

betaS.rf = data.frame(c(1:p), as.vector(fit.rf$importance[,1]))
colnames(betaS.rf) = c( "feature", "importance")

lsPlot = ggplot(betaS.ls, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")

enPlot = ggplot(betaS.en, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")

riPlot = ggplot(betaS.ri, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")

rfPlot = ggplot(betaS.rf, aes(x=feature, y=importance)) +
  geom_bar(stat = "identity", fill="white", colour="black")

grid.arrange(lsPlot, enPlot, riPlot, rfPlot, nrow = 4)

# we need to change the order of factor levels by specifying the order explicitly.
betaS.ls$feature = factor(betaS.ls$feature, levels = betaS.en$feature[order(betaS.en$value, decreasing = TRUE)])
betaS.en$feature = factor(betaS.en$feature, levels = betaS.en$feature[order(betaS.en$value, decreasing = TRUE)])
betaS.ri$feature = factor(betaS.ri$feature, levels = betaS.en$feature[order(betaS.en$value, decreasing = TRUE)])
betaS.rf$feature = factor(betaS.rf$feature, levels = betaS.en$feature[order(betaS.en$value, decreasing = TRUE)])



lsPlot =  ggplot(betaS.ls, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black") +
  ggtitle("lasso estimated coefficients") + theme(plot.title = element_text(hjust = .5,size = 60/.pt))

enPlot =  ggplot(betaS.en, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black") +
  ggtitle("elastic_net estimated coefficients") + theme(plot.title = element_text(hjust = .5,size = 60/.pt))

riPlot =  ggplot(betaS.ri, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black") +
  ggtitle("ridge estimated coefficients") + theme(plot.title = element_text(hjust = .5,size = 60/.pt))

rfPlot =  ggplot(betaS.rf, aes(x = feature, y = importance)) +
  geom_bar(stat = "identity", fill="white", colour="black") +
  ggtitle("random forest estimated coefficients") + theme(plot.title = element_text(hjust = .5,size = 60/.pt))

grid.arrange(lsPlot, enPlot, riPlot, rfPlot, nrow = 4)


################################## longitude as response variable ##################################
###### Question 2 #####
# define the response and predictors 
y = d$v70
X = d%>%select(-v69&-v70)
X = data.matrix(X)

# count the sample size and features
n = nrow(d)
p = ncol(X)

##### Question 3 #####
# use a loop to run the procedure for 100 times
# including split the dataset into train and test set
# fit the train data to lasso, elastic-net alpha=0.5, ridge and random forest
# set lambda as the result of 10-fold CV
# record both the test R squared and train R squared

# train set include 80% of the data
n.train = floor(0.8*n)

# test set include the rest of the data
n.test = n-n.train

# run the loop 100 times
M = 100

# set up several empty vectors for the R squared
Rsq.test.lasso = rep(0,M)
Rsq.train.lasso = rep(0,M)
Rsq.test.en = rep(0,M)
Rsq.train.en = rep(0,M)
Rsq.test.ridge = rep(0,M)
Rsq.train.ridge = rep(0,M)
Rsq.test.rf = rep(0,M)
Rsq.train.rf = rep(0,M)

for (m in c(1:M)) {
  # if not using caret package, there will be an error
  # divide both response and predictor matrix into test and train sets randomly
  library(caret)
  shuffled_indexes = sample(n)
  train = shuffled_indexes[1:n.train]
  test = shuffled_indexes[(1+n.train):n]
  X.train = X[train, ]
  y.train = y[train]
  X.test = X[test, ]
  y.test = y[test]
  
  # use the lambda chosen by the cross-validation to fit lasso and record the estimated coefficients and R-squared
  lasso.cv.fit = cv.glmnet(X.train, y.train, alpha = 1, nfolds = 10)
  lasso.fit = glmnet(X.train, y.train, alpha = 1, lambda = lasso.cv.fit$lambda.min)
  y.train.hat = predict(lasso.fit, newx = X.train, type = "response") 
  y.test.hat = predict(lasso.fit, newx = X.test, type = "response")  
  Rsq.test.lasso[m] = 1-mean((y.test - y.test.hat)^2)/mean((y.test - mean(y.test))^2)
  Rsq.train.lasso[m] = 1-mean((y.train - y.train.hat)^2)/mean((y.train - mean(y.train))^2)
  
  # use the lambda chosen by the cross-validation to fit elastic-net and record the estimated coefficients and R-squared  
  en.cv.fit = cv.glmnet(X.train, y.train, alpha = 0.5, nfolds = 10)
  en.fit = glmnet(X.train, y.train, alpha = 0.5, lambda = en.cv.fit$lambda.min)
  y.train.hat = predict(en.fit, newx = X.train, type = "response")
  y.test.hat = predict(en.fit, newx = X.test, type = "response") 
  Rsq.test.en[m] = 1-mean((y.test - y.test.hat)^2)/mean((y.test - mean(y.test))^2)
  Rsq.train.en[m] = 1-mean((y.train - y.train.hat)^2)/mean((y.train - mean(y.train))^2)
  
  # use the lambda chosen by the cross-validation to fit ridge and record the estimated coefficients and R-squared 
  ridge.cv.fit = cv.glmnet(X.train, y.train, alpha = 0, nfolds = 10)
  ridge.fit = glmnet(X.train, y.train, alpha = 0, lambda = ridge.cv.fit$lambda.min)
  y.train.hat = predict(ridge.fit, newx = X.train, type = "response") 
  y.test.hat = predict(ridge.fit, newx = X.test, type = "response") 
  Rsq.test.ridge[m] = 1-mean((y.test - y.test.hat)^2)/mean((y.test - mean(y.test))^2)
  Rsq.train.ridge[m] = 1-mean((y.train - y.train.hat)^2)/mean((y.train - mean(y.train))^2)
  
  # fit random forest and record the estimated coefficients and R-squared 
  rf.fit = randomForest(X.train, y.train, mtry = sqrt(p), importance = TRUE)
  y.test.hat = predict(rf.fit, X.test)
  y.train.hat = predict(rf.fit, X.train)
  Rsq.test.rf[m] = 1-mean((y.test - y.test.hat)^2)/mean((y.test - mean(y.test))^2)
  Rsq.train.rf[m] = 1-mean((y.train - y.train.hat)^2)/mean((y.train - mean(y.train))^2)
  
}

##### Question 4 #####
# side-by-side boxplots of test R squared and train R squared
par(mfrow = c(1,2))
boxplot(Rsq.train.lasso, Rsq.train.en, Rsq.train.ridge, Rsq.train.rf,
        main = "TRAIN SET R-SQUARED",
        names = c("LASSO", "EN", "RIDGE", "RF"),
        col = c("red","blue", "yellow", "green"))

boxplot(Rsq.test.lasso, Rsq.test.en, Rsq.test.ridge, Rsq.test.rf,
        main = "TEST SET R-SQUARED",
        names = c("LASSO", "EN", "RIDGE", "RF"),
        col = c("red","blue", "yellow", "green"))

# For one on the 100 sample, plot 10-fold CV curves. Record and present the time
# 10-fold cross-validation for lasso
par(mfrow = c(1,1))
ptm = proc.time()
lasso.cv = cv.glmnet(X.train, y.train, alpha = 1, nfolds = 10)
ptm = proc.time() - ptm
time_lasso = ptm["elapsed"]
cat(sprintf("Run Time for Lasso: %0.3f(sec):",time_lasso))
plot(lasso.cv) + title("10-fold CV curve for Lasso", line = 2.5)

# 10-fold cross-validation for elastic-net
ptm = proc.time()
en.cv = cv.glmnet(X.train, y.train, alpha = 0.5, nfolds = 10)
ptm = proc.time() - ptm
time_en = ptm["elapsed"]
cat(sprintf("Run Time for Elastic-net: %0.3f(sec):",time_en))
plot(en.cv) + title("10-fold CV curve for Elastic-net", line = 2.5)

# 10-fold cross-validation for ridge
ptm = proc.time()
ridge.cv = cv.glmnet(X.train, y.train, alpha = 0, nfolds = 10)
ptm = proc.time() - ptm
time_ridge = ptm["elapsed"]
cat(sprintf("Run Time for ridge: %0.3f(sec):",time_ridge))
plot(ridge.cv) + title("10-fold CV curve for Ridge", line = 2.5)

# side-by-side boxplots of train and test residuals lasso
cv.fit <- cv.glmnet(X.train, y.train, alpha = 1, nfolds = 10)
fit <- glmnet(X.train, y.train, alpha = 1, lambda = cv.fit$lambda.min)
y.train.hat <- predict(fit, newx = X.train, type = "response") 
y.test.hat <- predict(fit, newx = X.test, type = "response")  
Res.test.la <- y.test - y.test.hat
Res.test.la <- as.vector(Res.test.la)
Res.train.la <- y.train - y.train.hat
Res.train.la <- as.vector(Res.train.la)

# side-by-side boxplots of train and test residuals elastic-net
cv.fit <- cv.glmnet(X.train, y.train, alpha = 0.5, nfolds = 10)
fit <- glmnet(X.train, y.train, alpha = 0.5, lambda = cv.fit$lambda.min)
y.train.hat <- predict(fit, newx = X.train, type = "response") 
y.test.hat <- predict(fit, newx = X.test, type = "response")  
Res.test.en <- y.test - y.test.hat
Res.test.en <- as.vector(Res.test.en)
Res.train.en <- y.train - y.train.hat
Res.train.en <- as.vector(Res.train.en)

# side-by-side boxplots of train and test residuals ridge
cv.fit <- cv.glmnet(X.train, y.train, alpha = 0, nfolds = 10)
fit <- glmnet(X.train, y.train, alpha = 0, lambda = cv.fit$lambda.min)
y.train.hat <- predict(fit, newx = X.train, type = "response") 
y.test.hat <- predict(fit, newx = X.test, type = "response")  
Res.test.ri <- y.test - y.test.hat
Res.test.ri <- as.vector(Res.test.ri)
Res.train.ri <- y.train - y.train.hat
Res.train.ri <- as.vector(Res.train.ri)

# side-by-side boxplots of train and test residuals random forest
rf <- randomForest(X.train, y.train, mtry = sqrt(p), importance = TRUE)
y.test.hat <- predict(rf, X.test)
y.train.hat <- predict(rf, X.train)
Res.test.rf <- y.test - y.test.hat
Res.train.rf <- y.train - y.train.hat

par(mfrow=c(1,2))
boxplot(Res.train.la, Res.train.en, Res.train.ri, Res.train.rf,
        main = "RESIDUALS OF TRAIN SET",
        names = c("LASSO", "EN", "RIDGE", "RF"),
        col = c("red","blue", "yellow", "green"))

boxplot(Res.test.la, Res.test.en, Res.test.ri, Res.test.rf,
        main = "RESIDUALS OF TEST SET",
        names = c("LASSO", "EN", "RIDGE", "RF"),
        col = c("red","blue", "yellow", "green"))

##### Question 5 #####
# use 10-fold cross validation to fit ridge, lasso, elastic-net and random forest
# lasso
la_start = Sys.time()
cv.la = cv.glmnet(X, y, alpha = 1, nfolds = 10)
la = glmnet(X, y, alpha = 1, lambda = cv.la$lambda.min)
la_end = Sys.time()
la_time = la_end - la_start
lasso.Rsq.ci = t.test(Rsq.test.lasso, conf.level = 0.9)

# elastic-net
en_start = Sys.time()
cv.en = cv.glmnet(X, y, alpha = 0.5, nfolds = 10)
en = glmnet(X, y, alpha = 0.5, lambda = cv.en$lambda.min)
en_end = Sys.time()
en_time = en_end - en_start
en.Rsq.ci = t.test(Rsq.test.en, conf.level = 0.9)

# ridge
ri_start = Sys.time()
cv.ri = cv.glmnet(X, y, alpha = 0, nfolds = 10)
ri = glmnet(X, y, alpha = 0, lambda = cv.ri$lambda.min)
ri_end = Sys.time()
ri_time = ri_end - ri_start
ridge.Rsq.ci = t.test(Rsq.test.ridge, conf.level = 0.9)

# random forest
rf_start = Sys.time()
rf = randomForest(X, y, mtry = sqrt(p), importance = TRUE)
rf_end = Sys.time()
rf_time = rf_end - rf_start
rf.Rsq.ci = t.test(Rsq.test.rf, conf.level = 0.9)

lasso.Rsq.ci$conf.int[1:2]
en.Rsq.ci$conf.int[1:2]
ridge.Rsq.ci$conf.int[1:2]
rf.Rsq.ci$conf.int[1:2]
la_time
en_time
ri_time
rf_time

# bar-plots of the estimated coefficients
# elastic-net
cv.fit = cv.glmnet(X, y, alpha = 0.5, nfolds = 10)
fit.en = glmnet(X, y, alpha = 0.5, lambda = cv.fit$lambda.min)

# lasso
cv.fit = cv.glmnet(X, y, alpha = 1, nfolds = 10)
fit.ls = glmnet(X, y, alpha = 1, lambda = cv.fit$lambda.min)

# ridge
cv.fit = cv.glmnet(X, y, alpha = 0, nfolds = 10)
fit.ri = glmnet(X, y, alpha = 0, lambda = cv.fit$lambda.min)

# random forest
fit.rf = randomForest(X, y, mtry = sqrt(p), importance = TRUE)

betaS.en = data.frame(c(1:p), as.vector(fit.en$beta))
colnames(betaS.en) = c( "feature", "value")

betaS.ls = data.frame(c(1:p), as.vector(fit.ls$beta))
colnames(betaS.ls) = c( "feature", "value")

betaS.ri = data.frame(c(1:p), as.vector(fit.ri$beta))
colnames(betaS.ri) = c( "feature", "value")

betaS.rf = data.frame(c(1:p), as.vector(fit.rf$importance[,1]))
colnames(betaS.rf) = c( "feature", "importance")

lsPlot = ggplot(betaS.ls, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")

enPlot = ggplot(betaS.en, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")

riPlot = ggplot(betaS.ri, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")

rfPlot = ggplot(betaS.rf, aes(x=feature, y=importance)) +
  geom_bar(stat = "identity", fill="white", colour="black")

grid.arrange(lsPlot, enPlot, riPlot, rfPlot, nrow = 4)

# we need to change the order of factor levels by specifying the order explicitly.
betaS.ls$feature = factor(betaS.ls$feature, levels = betaS.en$feature[order(betaS.en$value, decreasing = TRUE)])
betaS.en$feature = factor(betaS.en$feature, levels = betaS.en$feature[order(betaS.en$value, decreasing = TRUE)])
betaS.ri$feature = factor(betaS.ri$feature, levels = betaS.en$feature[order(betaS.en$value, decreasing = TRUE)])
betaS.rf$feature = factor(betaS.rf$feature, levels = betaS.en$feature[order(betaS.en$value, decreasing = TRUE)])



lsPlot =  ggplot(betaS.ls, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black") +
  ggtitle("lasso estimated coefficients") + theme(plot.title = element_text(hjust = .5,size = 60/.pt))

enPlot =  ggplot(betaS.en, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black") +
  ggtitle("elastic_net estimated coefficients") + theme(plot.title = element_text(hjust = .5,size = 60/.pt))

riPlot =  ggplot(betaS.ri, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black") +
  ggtitle("ridge estimated coefficients") + theme(plot.title = element_text(hjust = .5,size = 60/.pt))

rfPlot =  ggplot(betaS.rf, aes(x = feature, y = importance)) +
  geom_bar(stat = "identity", fill="white", colour="black") +
  ggtitle("random forest estimated coefficients") + theme(plot.title = element_text(hjust = .5,size = 60/.pt))

grid.arrange(lsPlot, enPlot, riPlot, rfPlot, nrow = 4)
