# load libraries
library(neuralnet)
library(MASS)

set.seed(500)

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

lm.fit <- glm(despesatotal~., data=train_data)
summary(lm.fit)
pr.lm <- predict(lm.fit,test_data)
MSE.lm <- sum((pr.lm - test_data$despesatotal)^2)/nrow(test_data)

maxs_train <- apply(train_data, 2, max) 
maxs_test  <- apply(test_data, 2, max) 
mins_train <- apply(train_data, 2, min)
mins_test  <- apply(test_data, 2, min)

scaled_train <- as.data.frame(scale(train_data, center = mins_train, scale = maxs_train - mins_train))
scaled_test <- as.data.frame(scale(test_data, center = mins_test, scale = maxs_test - mins_test))

final_train_data <- scaled_train
final_test_data <- scaled_test

#exp = (final_test_data$despesatotal)*(max(test_data$despesatotal)-min(test_data$despesatotal))+min(test_data$despesatotal)

n <- names(final_train_data)
f <- as.formula(paste("despesatotal ~", paste(n[!n %in% "despesatotal"], collapse = " + ")))
nn <- neuralnet(f,data=final_train_data,hidden=c(10,10,10,10),linear.output=T,rep=10,lifesign="full")

#plot(nn)

pr.nn <- compute(nn,final_test_data[,1:12])

pr.nn_denormalized <- pr.nn$net.result*(max(test_data$despesatotal)-min(test_data$despesatotal))+min(test_data$despesatotal)
test.r <- (final_test_data$despesatotal)*(max(test_data$despesatotal)-min(test_data$despesatotal))+min(test_data$despesatotal)

MSE.nn <- sum((test.r - pr.nn_denormalized)^2)/nrow(final_test_data)


print(paste(MSE.lm,MSE.nn))

par(mfrow=c(1,2))
# 121292202748781
# plot(test_data$despesatotal,pr.nn_denormalized,col='red',main='Real vs predicted NN',pch=18,cex=0.7)
# abline(0,1,lwd=2)
# legend('bottomright',legend='NN',pch=18,col='red', bty='n')
# 
# plot(test_data$despesatotal,pr.lm,col='blue',main='Real vs predicted lm',pch=18, cex=0.7)
# abline(0,1,lwd=2)
# legend('bottomright',legend='LM',pch=18,col='blue', bty='n', cex=.95)

plot(test_data$despesatotal,pr.nn_denormalized,col='red',main='Real vs predicted NN',pch=18,cex=0.7)
points(test_data$despesatotal,pr.lm,col='blue',pch=18,cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend=c('NN','LM'),pch=18,col=c('red','blue'))






