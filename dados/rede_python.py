#Exercicio  diabetes

import numpy as np
import math, time
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.models import model_from_yaml
import matplotlib.pyplot as plt
import os
import glob
from sklearn.preprocessing import StandardScaler

# fixar random seed para se puder reproduzir os resultados
#print("BLAbla")
seed = 9
np.random.seed(seed)
train_input_attributes_scaler=StandardScaler()
train_output_attributes_scaler=StandardScaler()
test_input_attributes_scaler=StandardScaler()
test_output_attributes_scaler=StandardScaler()

# Etapa 1 - preparar o dataset
'''
load dataset
'''
def read_cvs_dataset():
	train_path = 'tratados/treino/'
	test_path = 'tratados/teste/'
	primeiro=0
	# ler ficheiros csv de treino para matriz numpy, e separar o label que está em col_label (deve ser a ultima coluna)
	for infile in glob.glob( os.path.join(train_path, '*.csv') ):
		if primeiro==0:
			train_dataset = np.loadtxt(infile, delimiter=";",skiprows=1)
			primeiro=1
		else:
			train_dataset=np.concatenate([train_dataset,np.loadtxt(infile, delimiter=";",skiprows=1)])
	# ler ficheiros csv de teste para matriz numpy, e separar o label que está em col_label (deve ser a ultima coluna)
	primeiro=0
	for infile in glob.glob( os.path.join(test_path, '*.csv') ):
		if primeiro==0:
			test_dataset = np.loadtxt(infile, delimiter=";",skiprows=1)
			primeiro=1
		else:
			test_dataset=np.concatenate([test_dataset,np.loadtxt(infile, delimiter=";",skiprows=1)])
	# ler ficheiro csv para matriz numpy, e separar o label que está em col_label (deve ser a ultima coluna)
	train_input_attributes = train_dataset[:,0:12]
	train_output_attributes = train_dataset[:,12]
	test_input_attributes = test_dataset[:,0:12]
	test_output_attributes = test_dataset[:,12]
	return (train_input_attributes,train_output_attributes,test_input_attributes,test_output_attributes)

def pre_process_data(train_input_attributes,train_output_attributes,test_input_attributes,test_output_attributes):
	scaled_train_input_attributes = train_input_attributes_scaler.fit_transform(train_input_attributes)
	scaled_train_output_attributes = train_output_attributes_scaler.fit_transform(train_output_attributes)
	scaled_test_input_attributes = test_input_attributes_scaler.fit_transform(test_input_attributes)
	scaled_test_output_attributes = test_output_attributes_scaler.fit_transform(test_output_attributes)
	#return (scaled_train_input_attributes,scaled_train_output_attributes,scaled_test_input_attributes,scaled_test_output_attributes)
	return (train_input_attributes,train_output_attributes,test_input_attributes,test_output_attributes)

# Etapa 2 - Definir a topologia da rede (arquitectura do modelo)
def create_model():
	model = Sequential()
	model.add(Dense(12, input_dim=12, activation="relu", kernel_initializer="uniform"))
	model.add(Dense(8, activation="relu", kernel_initializer="uniform"))
	model.add(Dense(1, activation="linear", kernel_initializer="uniform"))
	model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])
	return model

#util para visualizar a topologia da rede num ficheiro em pdf ou png
def print_model(model,fich):
	from keras.utils import plot_model
	plot_model(model, to_file=fich, show_shapes=True, show_layer_names=True)

# Etapa 6 - Utilizar o modelo treinado e escrever as previsões para novos casos
def model_print_predictions(model,input_attributes,output_attributes):
	previsoes = model.predict(input_attributes)
	mse = np.mean((output_attributes-previsoes)**2)
	rms = mean_squared_error(output_attributes, previsoes)
	print("MSE: ",mse, "  RMS: ",rms)
	previsoes=test_output_attributes_scaler.inverse_transform(previsoes)
	output_attributes=test_output_attributes_scaler.inverse_transform(output_attributes)	
	LP=[]
	for prev in previsoes:
		LP.append(prev[0])
	#LP = [round(prev[0]) for prev in previsoes]
	for i in range(len(output_attributes)):
		print(" Class:",output_attributes[i]," previsão:",float(LP[i]))
		if i>10: break
	mse = np.mean((output_attributes-previsoes)**2)
	rms = mean_squared_error(output_attributes, previsoes)
	print("MSE: ",mse, "  RMS: ",rms)

#imprime um grafico com os valores de teste e com as correspondentes tabela de previsões
def print_series_prediction(y_test,predic):
	diff=[]
	racio=[]
	#predic=test_output_attributes_scaler.inverse_transform(predic)
	#y_test=test_output_attributes_scaler.inverse_transform(y_test)
	for i in range(len(y_test)): #para imprimir tabela de previsoes
		racio.append( (y_test[i]/predic[i])-1)
		diff.append( abs(y_test[i]- predic[i]))
		if i<10:
			print('valor: %f ---> Previsão: %f Diff: %f Racio: %f' % (y_test[i],predic[i], diff[i],racio[i]))
	plt.plot(y_test,color='blue', label='Input')
	plt.plot(predic,color='red', label='Previsão') #este deu uma linha em branco
	#plt.plot(diff,color='green', label='Diferença')
	#plt.plot(racio,color='yellow', label='Rácio')
	plt.legend(loc='upper left')
	plt.show()

 # Ciclo completo executando as Etapas 1,2,3,4,5 e 6
def ciclo_completo():
	#leitura de dados
	(train_input_attributes,train_output_attributes,test_input_attributes,test_output_attributes) = read_cvs_dataset()	
	#pré-processamento
	(final_train_input,final_train_output,final_test_input,final_test_output)=pre_process_data(train_input_attributes,train_output_attributes,test_input_attributes,test_output_attributes)
	model = create_model()
	#print_model(model,"model_MLP.png")
	model.fit(final_train_input, final_train_output, batch_size=10, epochs=500, verbose=2)
	trainScore = model.evaluate(final_train_input, final_train_output, verbose=0)
	print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))
	nmrse= math.sqrt(trainScore[0])/(max(final_train_output)-min(final_train_output))
	print('Normalized Train Score: %.10f NRMSE' % (nmrse))
	testScore = model.evaluate(final_test_input, final_test_output, verbose=0)
	print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))
	nmrse= math.sqrt(testScore[0])/(max(final_test_output)-min(final_test_output))
	print('Normalize Test Score: %.10f NRMSE' % (nmrse))
	#print(model.metrics_names)
	p = model.predict(final_test_input)
	predic = np.squeeze(np.asarray(p)) 
	print_series_prediction(final_test_output,predic)
	#print_history_accuracy(history)
	#print_history_loss(history)
	#model_print_predictions(model,final_test_input,final_test_output)


	
if __name__ == '__main__':
 #opção 1 - ciclo completo
	ciclo_completo()
 #opção 2 - ler,treinar o dataset e gravar. Depois ler o modelo e pesos e usar
		#(input_attributes,output_attributes)=ciclo_ler_dataset_treinar_gravar()
		#ciclo_ler_modelo_evaluate_usar(input_attributes,output_attributes)
