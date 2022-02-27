#loading the libraries and the data set
#you might need to install these packages in which case use the
# "install.packages("") command"
library(Matching)
library(arm)
library(MLmetrics)
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
summary(lm2)
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

#finding the OOB
df <- lalonde
keep <- 9 # number of the re78 column
df <- df[, -keep] #removing the re78 column
tuneRF(df, lalonde$re78) #gives output of OOB for multiple mtry valuees, we pick the value for mtry = 3 for ours because that is what we used

#Predicting values via simulation for the first linear regression model
# for people with high school degree, the average educ value is 12
# for people without high school degree, the average educ value is 10

#finding mean of the age
age_mean <- mean(lalonde$age)

#generating storage vectors for all the simulations
predicted_degr_1_treat <- rep(0, 1000) #people with high school degrees given the treatment
predicted_degr_1_notreat <- rep(0, 1000) #people with high school degrees not given the treatment
predicted_nodegr_1_treat <- rep(0, 1000) #people without high school degrees given the treatment
predicted_nodegr_1_notreat <- rep(0, 1000) #people without high school degrees not given the treatment

#generating coefficients with simulations
sim.coeefficients <- sim(lm1, 1000)
#now the for loops
#people with degrees given the treatment
for (i in 1:1000) {
    predicted_degr_1_treat[i] <- sim.coeffiecients@coef[i,1] + sim.coeffiecients@coef[i,2]*age_mean + sim.coeffiecients@coef[i,3]*12 
    + sim.coeffiecients@coef[i,4]*1 + sim.coeffiecients@coef[i,5]*0 + sim.coeffiecients@coef[i,6]*0 + sim.coeffiecients@coef[i,7]*0 + 
    sim.coeffiecients@coef[i,8]*0 + sim.coeffiecients@coef[i,9]*0 + sim.coeffiecients@coef[i,10]*0 + sim.coeffiecients@coef[i,11]*1 + 
    rnorm(1, mean = 0, sd = sim.coeffiecients@sigma[i]) #the error term
}
mean(predicted_degr_1_treat)
quantile(predicted_degr_1_treat, probs = c(0.025, 0.975))
#people with degrees not given the treatment
for (i in 1:1000) {
    predicted_degr_1_notreat[i] <- sim.coeffiecients@coef[i,1] + sim.coeffiecients@coef[i,2]*age_mean + sim.coeffiecients@coef[i,3]*12 
    + sim.coeffiecients@coef[i,4]*1 + sim.coeffiecients@coef[i,5]*0 + sim.coeffiecients@coef[i,6]*0 + sim.coeffiecients@coef[i,7]*0 + 
    sim.coeffiecients@coef[i,8]*0 + sim.coeffiecients@coef[i,9]*0 + sim.coeffiecients@coef[i,10]*0 + sim.coeffiecients@coef[i,11]*0 + 
    rnorm(1, mean = 0, sd = sim.coeffiecients@sigma[i]) #the error term
}
mean(predicted_degr_1_notreat)
quantile(predicted_degr_1_notreat, probs = c(0.025, 0.975))
#people without degrees given the treatment
for (i in 1:1000) {
    predicted_nodegr_1_treat[i] <- sim.coeffiecients@coef[i,1] + sim.coeffiecients@coef[i,2]*age_mean + sim.coeffiecients@coef[i,3]*10 
    + sim.coeffiecients@coef[i,4]*1 + sim.coeffiecients@coef[i,5]*0 + sim.coeffiecients@coef[i,6]*0 + sim.coeffiecients@coef[i,7]*1 + 
    sim.coeffiecients@coef[i,8]*0 + sim.coeffiecients@coef[i,9]*0 + sim.coeffiecients@coef[i,10]*0 + sim.coeffiecients@coef[i,11]*1 + 
    rnorm(1, mean = 0, sd = sim.coeffiecients@sigma[i]) #the error term
}
mean(predicted_nodegr_1_treat)
quantile(predicted_nodegr_1_treat, probs = c(0.025, 0.975))
#people without degrees not given the treatment
for (i in 1:1000) {
    predicted_nodegr_1_notreat[i] <- sim.coeffiecients@coef[i,1] + sim.coeffiecients@coef[i,2]*age_mean + sim.coeffiecients@coef[i,3]*10 
    + sim.coeffiecients@coef[i,4]*1 + sim.coeffiecients@coef[i,5]*0 + sim.coeffiecients@coef[i,6]*0 + sim.coeffiecients@coef[i,7]*1 + 
    sim.coeffiecients@coef[i,8]*0 + sim.coeffiecients@coef[i,9]*0 + sim.coeffiecients@coef[i,10]*0 + sim.coeffiecients@coef[i,11]*0 + 
    rnorm(1, mean = 0, sd = sim.coeffiecients@sigma[i]) #the error term
}
mean(predicted_nodegr_1_treat)
quantile(predicted_nodegr_1_notreat, probs = c(0.025, 0.975))


#Predicting values via simulation for the second linear regression model

#generating storage vectors for all the simulations
predicted_degr_2_treat <- rep(0, 1000) #people with high school degrees given the treatment
predicted_degr_2_notreat <- rep(0, 1000) #people with high school degrees not given the treatment
predicted_nodegr_2_treat <- rep(0, 1000) #people without high school degrees given the treatment
predicted_nodegr_2_notreat <- rep(0, 1000) #people without high school degrees not given the treatment

#generating coefficients for the models
sim2 <- sim(lm2, 1000)

# for loops
# with degrees given treatment
for (i in 1:1000) {
    predicted_degr_2_treat[i] <- sim2@coef[i,1] + sim2@coef[i,2]*age_mean + sim2@coef[i,3]*12 + sim2@coef[i,4]*1 + sim2@coef[i,5]*0 + sim2@coef[i,6]*0 + 
    sim2@coef[i,7]*0 + sim2@coef[i,8]*0 + sim2@coef[i,9]*0 + sim2@coef[i,10]*0 + sim2@coef[i,11]*1 + rnorm(1, mean = 0, sd = sim2@sigma[i]) 
}
mean(predicted_degr_2_treat)
quantile(predicted_degr_2_treat, probs = c(0.025, 0.975))
# with degrees without treatment
for (i in 1:1000) {
    predicted_degr_2_notreat[i] <- sim2@coef[i,1] + sim2@coef[i,2]*age_mean + sim2@coef[i,3]*12 + sim2@coef[i,4]*1 + sim2@coef[i,5]*0 + sim2@coef[i,6]*0 + 
    sim2@coef[i,7]*0 + sim2@coef[i,8]*0 + sim2@coef[i,9]*0 + sim2@coef[i,10]*0 + sim2@coef[i,11]*0 + rnorm(1, mean = 0, sd = sim2@sigma[i]) 
}
mean(predicted_degr_2_notreat)
quantile(predicted_degr_2_notreat, probs = c(0.025, 0.975))
#without degrees with treatment
for (i in 1:1000) {
    predicted_nodegr_2_treat[i] <- sim2@coef[i,1] + sim2@coef[i,2]*age_mean + sim2@coef[i,3]*10 + sim2@coef[i,4]*1 + sim2@coef[i,5]*0 + sim2@coef[i,6]*0 + 
    sim2@coef[i,7]*1 + sim2@coef[i,8]*0 + sim2@coef[i,9]*0 + sim2@coef[i,10]*0 + sim2@coef[i,11]*1 + rnorm(1, mean = 0, sd = sim2@sigma[i]) 
}
mean(predicted_nodegr_2_treat)
quantile(predicted_nodegr_2_treat, probs = c(0.025, 0.975))
#without degrees without treatment
for (i in 1:1000) {
    predicted_nodegr_2_notreat[i] <- sim2@coef[i,1] + sim2@coef[i,2]*age_mean + sim2@coef[i,3]*10 + sim2@coef[i,4]*1 + sim2@coef[i,5]*0 + sim2@coef[i,6]*0 + 
    sim2@coef[i,7]*1 + sim2@coef[i,8]*0 + sim2@coef[i,9]*0 + sim2@coef[i,10]*0 + sim2@coef[i,11]*0 + rnorm(1, mean = 0, sd = sim2@sigma[i]) 
}
mean(predicted_nodegr_1_notreat)
quantile(predicted_nodegr_1_notreat, probs = c(0.025, 0.975))
