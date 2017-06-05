library(ISLR)
library(caTools)
library(neuralnet)
set.seed(101)

#LOAD DATA
data = read.csv("C:/Users/Utilizador/Desktop/Universidade/4ano/SI/Computação Natural/TP2/dados/tratados/despesa/AIIIDM2010.csv",header=TRUE,sep=";")  # read csv file 


#Normalization
# Create Vector of Column Max and Min Values
maxs <- apply(data[,2:12], 2, max)
mins <- apply(data[,2:12], 2, min)

# Use scale() and convert the resulting matrix to a data frame
scaled.data <- as.data.frame(scale(data[,2:12],center = mins, scale = maxs - mins))

# Normalize results

maxr <- max(data[,13])
minr <- min(data[,13])
scaled.results <- as.data.frame(scale(data[,13],center = minr, scale = maxr - minr))
despesatotal = as.numeric(scaled.results$V1)
final_data = cbind(despesatotal,scaled.data)
#final_data=cbind()

# Create Split (any column is fine)
split = sample.split(final_data$despesatotal, SplitRatio = 0.70)

# Split based off of split Boolean Vector
train = subset(final_data, split == TRUE)

test = subset(final_data, split == FALSE)

feats <- names(scaled.data)

# Concatenate strings
f <- paste(feats,collapse=' + ')
f <- paste('despesatotal ~',f)



# Convert to formula
f <- as.formula(f)


nn <- neuralnet(f,train,hidden=c(10,10,10),linear.output=TRUE)


# Compute Predictions off Test Set
net.results <- compute(nn,test[2:12])

# Check out net.result
#print(head(predicted.nn.values$net.result))

#plot(nn)

#Lets see what properties net.sqrt has
ls(net.results)

#Lets see the results
#print(net.results$net.result)

#Lets display a better version of the results
cleanoutput <- cbind(test$despesatotal,test$despesatotal,
                     as.data.frame(net.results$net.result))
colnames(cleanoutput) <- c("Input","Expected Output","Neural Net Output")
print(cleanoutput)
