# load libraries
library(caret)
library(ISLR)
library(caTools)
library(neuralnet)


#LOAD DATA
train_path="C:/Users/Utilizador/Desktop/Universidade/4ano/SI/Computação Natural/TP2/DNN_Regression/dados/tratados/treino/"
test_path="C:/Users/Utilizador/Desktop/Universidade/4ano/SI/Computação Natural/TP2/DNN_Regression/dados/tratados/teste/"

#training data
train_files = list.files(path=train_path,pattern="*.csv")
# ler o primeiro para iniciar data
file_path=paste(train_path,train_files[1],sep="")
train_data=read.csv(file_path,header=TRUE,sep=";")
#ler o resto
for (i in 2:length(train_files)){
  file_path=paste(train_path,train_files[i],sep="")
  train_data=rbind(train_data,read.csv(file_path,header=TRUE,sep=";"))
}


#test data
test_files = list.files(path=test_path,pattern="*.csv")
# ler o primeiro para iniciar data
file_path=paste(test_path,test_files[1],sep="")
test_data=read.csv(file_path, header=TRUE,sep=";")
#ler o resto
for (i in 2:length(test_files)){
  file_path=paste(test_path,test_files[i],sep="")
  test_data=rbind(test_data,read.csv(file_path,header=TRUE,sep=";"))
}

final_train_data = as.data.frame(train_data)
final_test_data= as.data.frame(test_data)

# #Normalization
# # Create Vector of Column Max and Min Values
# maxs_train <- apply(train_data[,2:12], 2, max)
# maxs_test <-  apply(test_data[,2:12],2,max)
# mins_train <- apply(train_data[,2:12], 2, min)
# mins_test <-  apply(test_data[,2:12], 2, min)
# 
# # Use scale() and convert the resulting matrix to a data frame
# scaled.train_data <- as.data.frame(scale(train_data[,2:12],center = mins_train, scale = maxs_train - mins_train))
# scaled.test_data <- as.data.frame(scale(test_data[,2:12],center = mins_test, scale = maxs_test - mins_test))

# # Normalize results
# maxr_train <- max(train_data[,13])
# maxr_test <-  max(test_data[,13])
# minr_train <- min(train_data[,13])
# minr_test <-  min(test_data[,13])
# scaled.results_train <- as.data.frame(scale(train_data[,13],center = minr_train, scale = maxr_train - minr_train))
# scaled.results_test <- as.data.frame(scale(test_data[,13],center = minr_test, scale = maxr_test - minr_test))
# #TREINO
# despesatotal= as.numeric(scaled.results_train$V1)
# final_train_data = cbind(despesatotal,scaled.train_data)
# #Teste
# despesatotal= as.numeric(scaled.results_test$V1)
# final_test_data = cbind(despesatotal,scaled.test_data)


# #Activation function
# feats <- names(scaled.train_data)
# 
# # Concatenate strings
# f <- paste(feats,collapse=' + ')
# f <- paste('despesatotal ~',f)
# 
# 
#  
# # Convert to formula
# f <- as.formula(f)



nn <- neuralnet(despesatotal~.,train_data,hidden=c(10,10,10),linear.output=TRUE)


# Compute Predictions off Test Set
#net.results <- compute(nn,test[2:12])

# Check out net.result
#print(head(predicted.nn.values$net.result))

#plot(nn)

#Lets see what properties net.sqrt has
# ls(net.results)

#Lets see the results
#print(net.results$net.result)

#Lets display a better version of the results
# cleanoutput <- cbind(test$despesatotal,test$despesatotal,
#                      as.data.frame(net.results$net.result))
# colnames(cleanoutput) <- c("Input","Expected Output","Neural Net Output")
# print(cleanoutput)



