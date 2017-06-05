using MXNet
using Distributions
using PyPlot

srand(0)

input=zeros(0,13)
output=zeros(0)
input_teste=zeros(0,13)
output_teste=zeros(0)


for i=2010:2010
  f=open("C:\\Users\\Utilizador\\Desktop\\Universidade\\4ano\\SI\\Computação Natural\\TP2\\dados\\tratados\\despesa\\AIIIDM"*string(i)*".csv")
    first_line=1

    for ln in eachline(f)
      if first_line==1
        first_line=0
      else
        splited=split(ln,';')
        splited=map(x->parse(Float64,x),splited)
        input=[input;splited']
      end
    end

    input=input'
    output=input[13,:]
    input=input[1:12,:]
    close(f)
end

#for i=2011:2014
#  f=open("C:\\Users\\Utilizador\\Desktop\\Universidade\\4ano\\SI\\Computação Natural\\TP2\\dados\\tratados\\despesa\\AIIIDM"*string(i)*".csv")
#    first_line=1
#
#    for ln in eachline(f)
#      if first_line==1
#        first_line=0
#      else
#        splited=split(ln,';')
#        splited=map(x->parse(Float64,x),splited)
#        input=[input;splited']
#      end
#    end

#    input=input'
#    output=cat(output,input[13,:])
#    input=cat(input,input[1:12,:])
#    close(f)
#end

print(size(input))
print(size(output_teste))

f=open("C:\\Users\\Utilizador\\Desktop\\Universidade\\4ano\\SI\\Computação Natural\\TP2\\dados\\tratados\\despesa\\AIIIDM2015.csv")
  first_line=1

  for ln in eachline(f)
    if first_line==1
      first_line=0
    else
      splited=split(ln,';')
      splited=map(x->parse(Float64,x),splited)
      input_teste=[input_teste;splited']
    end
  end

input_teste=input_teste'
output_teste=input_teste[13,:]
input_teste=input_teste[1:12,:]



batchsize = 10 # can adjust this later, but must be defined now for next line
trainprovider = mx.ArrayDataProvider(:data => input,batch_size=batchsize,shuffle=true, :label => output)
evalprovider = mx.ArrayDataProvider(:data => input_teste, batch_size=batchsize,shuffle=true, :label => output_teste)

data = mx.Variable(:data)
label = mx.Variable(:label)

net = @mx.chain     mx.Variable(:data) =>
                    mx.FullyConnected(num_hidden=64) =>
                    mx.Activation(act_type=:tanh) =>
                    mx.FullyConnected(num_hidden=32) =>
                    mx.Activation(act_type=:tanh) =>
                    mx.FullyConnected(num_hidden=16) =>
                    mx.Activation(act_type=:tanh) =>
                    mx.FullyConnected(num_hidden=8) =>
                    mx.Activation(act_type=:tanh) =>
                    mx.FullyConnected(num_hidden=1)  =>
                    mx.LinearRegressionOutput(label)



# final model definition, don't change, except if using gp
model = mx.FeedForward(net, context=mx.cpu())
optimizer = mx.SGD(lr=0.01, momentum=0.9, weight_decay=0.00001)

# train, reporting loss for training and evaluation sets
# initial training with small batch size, to get to a good neighborhood
batchsize=10
#print(size(evalprovider))
mx.fit(model, optimizer, initializer=mx.NormalInitializer(0.0,0.1), eval_metric=mx.Accuracy(), trainprovider, eval_data=evalprovider, n_epoch = 10)
#mx.parameters(net)

plotprovider = mx.ArrayDataProvider(:data => input_teste, :label => output_teste)
fit = mx.predict(model, plotprovider)
print(fit)
