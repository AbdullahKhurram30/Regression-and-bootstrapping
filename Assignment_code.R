#loading the libraries and the data set
library(Matching)
library(arm)
library(Melmetrics)
library(tree)
library(randomForest)
data("lalonde")

set.seed(123)

#general dataset statistics
mean(lalonde$re78)
sd(lalonde$re78)
quantile(lalonde$re78, probs = c(0.025, 0.975))

#the first linear regression model
train <- sample(1:nrow(lalonde), nrow(lalonde)*(0.8))
lalonde.train <- lalonde[train, ]
lalonde.test <- lalonde[-train,]
lm1 <- lm(re78 ~ ., data = lalonde.train)
summary(lm1)
lm1.predict <- predict(lm1, lalonde.test)
MSE(lm1.predict, lalonde.test$re78)
(MSE(lm1.predict, lalonde.test$re78))^(0.5)
quantile(lm1.predict, probs = c(0.025, 0.975))
mean(lm1.predict)

#the second linear regression model
lm2 <- lm(re78 ~ . + I(treat*nodegr), data = lalonde.train)
lm2.predict <- predict(lm2, lalonde.test)
MSE <- MSE(lm2.predict, lalonde.test$re78)
MSE
MSE^0.5
quantile(lm2.predict, probs = c(0.025, 0.975))
mean(lm2.predict)

#The first tree
tree.lalonde <- tree(re78 ~ age + black + hisp + married + nodegr + re74 + re75 + u74 + u75 + treat, data = lalonde.train)
plot(tree.lalonde)
text(tree.lalonde)
tree_pred <- predict(tree.lalonde, lalonde.test)
mean(tree_pred)
residuals_tree <- lalonde.test$re78 - tree_pred
MSE_tree <- mean(residuals_tree^2)
MSE_tree
(MSE_tree)^0.5
quantile(tree_pred, probs = c(0.025, 0.975))

#Cross-validation
cv.tree <- cv.tree(tree.lalonde, FUN = prune.tree)
cv.tree
plot(cv.tree)
text(cv.tree)
title("Tree size vs Standard Error", line = -35)
pruned_tree <- prune.tree(tree.lalonde, best = 9)
plot(pruned_tree)
text(pruned_tree)
prune.predict <- predict(pruned_tree, lalonde.test, type = "vector")
pruned_residuals <- lalonde.test$re78 - prune.predict
MSE_prune <- mean(pruned_residuals^2)
MSE_prune
(MSE_prune)^0.5
mean(prune.predict)
quantile(prune.predict, probs = c(0.025, 0.975))

#Random Forests
rf.lalonde <- randomForest(re78 ~ ., data = lalonde.train, mtry = 3, importance = TRUE)
summary(rf.lalonde)
rf.lalonde
varImpPlot(rf.lalonde)
rf_pred <- predict(rf.lalonde, lalonde.test)
rf_residuals <- lalonde.test$re78 - rf_pred
rf_MSE <- mean(rf_residuals^2)
rf_MSE
(rf_MSE)^0.5
mean(rf_pred)
quantile(rf_pred, probs = c(0.025, 0.975))
