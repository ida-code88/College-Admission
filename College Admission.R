#College Admission
#Nurlaida

#################################################################################
#Predictive
########################
library(dplyr)
library("RColorBrewer")
library(randomForest)
library(caret)
library(caTools)
library(rpart)
library(rpart.plot)
library(FSelector)
library(data.tree)
library(ggpubr)
library(rcompanion)

#set working directory
setwd("G:/Data Science/R/Projects/College Admission")
getwd()


#import and explore data
college_admission <- read.csv("College_admission.csv")
View(college_admission)
str(college_admission)
class(college_admission)
summary(college_admission)


#Find the missing values. (if any, perform missing value treatment)
is.na(college_admission)
colSums(is.na(college_admission))
#from the output, there is no missing values

#checking empty values
colSums(college_admission==' ')
#from the output, there is no empty values

#Find outliers (if any, then perform outlier treatment)
plot(college_admission$gre, college_admission$gpa,
     pch = 19,         # Solid circle
     cex = 1.5,        # Make 150% size
     col = "#cc0000",  # Red
     main = "GRE as a function of GPA",
     xlab = "GRE",
     ylab = "GPA")

boxplot(college_admission$gre) #there are two outliers for gre 
boxplot(college_admission$gpa) #there is one outlier for gpa
boxplot(college_admission$rank) #there is no outlier for rank

#removing outliers from gre
college_admission1 <- college_admission
bench_gre <- 520 - 1.5*IQR(college_admission1$gre)
bench_gre
college_admission1 <- filter(college_admission1, gre > 310)
boxplot(college_admission1$gre)

#removing outliers from gpa
bench_gpa <- 3.13 - 1.5*IQR(college_admission1$gpa)
bench_gpa
college_admission1 <- filter(college_admission1, gpa > 2.32)
boxplot(college_admission1$gpa)

summary(college_admission1)

#Find whether the data is normally distributed or not. Use the plot to determine the same. 
#gre
hist(college_admission1$gre,
     xlab = 'gre',
     main = 'Histogram of gre',
     col = '#D95F02')

qqnorm(college_admission1$gre)
qqline(college_admission1$gre, col = '#D95F02') 
ggdensity(college_admission1$gre, main="gre", xlab = "gre disrtibution")

gr <- college_admission1$gre
plotNormalHistogram(gr)
#gre is normally distributed

#gpa
hist(college_admission1$gpa,
     xlab = 'gpa',
     main = 'Histogram of gpa',
     col = '#1B9E77')

qqnorm(college_admission1$gpa)
qqline(college_admission1$gpa, col = '#1B9E77') 
ggdensity(college_admission1$gpa, main="gpa", xlab = "gpa disrtibution")

gp <- college_admission1$gpa
plotNormalHistogram(gp)
#gpa is normally distributed

#rank
hist(college_admission1$rank,
     xlab = 'rank',
     main = 'Histogram of rank',
     col = '#1B9E77') 

qqnorm(college_admission1$rank)

rk <- college_admission1$rank
plotNormalHistogram(rk)
#rank is normally distributed


#Admit
qqnorm(college_admission1$admit)


#Find the structure of the data set and if required, transform the numeric data type to factor and vice-versa.
View(college_admission1)
str(college_admission1)
college_admission1$rank <- as.factor(college_admission1$rank) #transform rank into factor data type
college_admission1$admit <- as.factor(college_admission1$admit) #transform admit into factor data type

#Use variable reduction techniques to identify significant variables
#In this case I use random forest
set.seed(123)
id <- sample(2, nrow(college_admission1), prob = c(0.7, 0.3), replace = TRUE)
colforest_train <- college_admission1[id==1,]
colforest_test <- college_admission1[id==2,]
str(colforest_train)

bestmtry <- tuneRF(colforest_train, colforest_train$admit,stepFactor = 1.2, 
                   improve = 0.01, trace = TRUE, plot = TRUE)

college_admforest <- randomForest(admit~., data = colforest_train)
college_admforest

importance(college_admforest) #gpa, gre, and rank are significant variable
varImpPlot(college_admforest)

pred_college <- predict(college_admforest, newdata = colforest_test, type = "class")
pred_college

confusionMatrix(table(pred_college, colforest_test$admit))


#Run logistic model to determine the factors that influence the admission process of a student 
#(Drop insignificant variables) 
#Split the data set into training and testing model
split_logistic <- sample.split(college_admission1, SplitRatio = 0.8)
split_logistic
train_logistic <- subset(college_admission1, split = "TRUE")
test_logistic <- subset(college_admission1, split = "FALSE")


#Calculate the accuracy of the model and run validation techniques.
#Train the model, using independent variable: gre and gpa
college_model <- glm(admit ~ ., data = train_logistic, family = 'binomial')
summary(college_model)

#Train the model, using independent variable: gre, gpa, and rank
college_model1 <- glm(admit ~ gre + gpa + rank, data = train_logistic, family = 'binomial')
summary(college_model1)

#Train the model, using independent variable: gre and gpa
college_model2 <- glm(admit ~ gre + gpa, data = train_logistic, family = 'binomial')
summary(college_model2)

#Train the model, using independent variable: gpa and rank
college_model3 <- glm(admit ~ gpa + rank, data = train_logistic, family = 'binomial')
summary(college_model3)

#Calculate the accuracy
res <- predict(college_model3, test_logistic, type = "response")
res

res <- predict(college_model3, train_logistic, type = "response")
res


#Validation Technique
confmatrix <- table(Actual_value=train_logistic$admit, Predicted_value = res > 0.5)
confmatrix

#Accuracy
(confmatrix[[1,1]] + confmatrix[[2,2]]) / sum(confmatrix)

#Here we choose college_model3 (gpa+rank, model with the highest accuracy) for further analysis

###########################################################################################
#Try other modeling techniques like decision tree and SVM and select a champion model 
#Decision Tree
#Eliminating unmeaningful variable
college_admissionDT <- select(college_admission1, admit, gpa, rank)

#Split the data set into training and testing model
set.seed(123)
split_dt <- sample.split(college_admissionDT$admit, SplitRatio = 0.8)
split_dt
train_dt <- subset(college_admissionDT, split = "TRUE")
test_dt <- subset(college_admissionDT, split = "FALSE")

#Training Test
tree <- rpart(admit ~., data = train_dt)

#Prediction
tree.admit.predict <- predict(tree, test_dt, type = 'class')

#Confusion Matrix
confusionMatrix(tree.admit.predict, test_dt$admit)

prp(tree)
rpart.plot(tree,extra=1, cex=0.7)

############################################################################################
#SVM
#Split the data set into training and testing model
set.seed(123)
partitionsvm <- createDataPartition(y = college_admission1$admit, p = 0.8, list = FALSE)
training_svm <- college_admission1[partitionsvm,]
testing_svm <- college_admission1[-partitionsvm,]
dim(training_svm)
dim(testing_svm)

#Train the method
control_svm <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
svm_linear <- train(admit~ gpa + rank, data = training_svm, method = "svmLinear", 
                    trControl = control_svm, preProcess = c("center", "scale"),
                    tuneLength = 10)

#Testing the method
test_predsvm <- predict(svm_linear, newdata = testing_svm)
test_predsvm

#Validation 
confusionMatrix(table(test_predsvm, testing_svm$admit))

#Improve Model Performance
grid <- expand.grid(C = c(0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2))
set.seed(123)
svm_linear_grid <- train(admit~ gpa + rank, data = training_svm,
                         method = "svmLinear", 
                         preProcess = c ("center", "scale"),
                         tuneGrid = grid,
                         tuneLength = 10)
svm_linear_grid
plot(svm_linear_grid)

test_pred_grid <- predict(svm_linear_grid, newdata = testing_svm)
test_pred_grid

confusionMatrix(table(test_pred_grid, testing_svm$admit))


#############################################################################################
#Descriptive


#Categorize the average of grade point into High, Medium, and Low 
#(with admission probability percentages) and plot it on a point chart.  

?cut
max(college_admission1$gre)
cut(college_admission$gre, breaks = c(0, 440, 580, Inf), 
    labels = c("Low", "Medium", "High"))
college_admission_bin <- college_admission
college_admission_bin$grebin <- cut(college_admission$gre, breaks = c(0, 440, 580, Inf), 
                                 labels = c("Low", "Medium", "High"))
View(college_admission_bin)

collab.df <- college_admission_bin[,2:3]
kmeans <- kmeans(collab.df, 3)
plot(collab.df[c("gre", "gpa")], col = kmeans$cluster)
points(kmeans$centers[,c("gre", "gpa")], col = 1:3, pch = 8, cex = 2)
