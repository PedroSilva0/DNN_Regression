#Exercicio  diabetes

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.models import model_from_yaml
import matplotlib.pyplot as plt
import os
import glob

# fixar random seed para se puder reproduzir os resultados
#print("BLAbla")
seed = 9
np.random.seed(seed)

# Etapa 1 - preparar o dataset
'''
load dataset
'''
def read_cvs_dataset():
	train_path = 'treino/'
	test_path = 'teste/'
	primeiro=0
	# ler ficheiros csv de treino para matriz numpy, e separar o label que está em col_label (deve ser a ultima coluna)
	for infile in glob.glob( os.train_path.join(train_path, '*.csv') ):
		if primeiro==0:
			train_dataset = np.loadtxt(infile, delimiter=";",skiprows=1)
			primeiro=1
		else:
			train_dataset=np.concatanate(train_dataset,np.loadtxt(infile, delimiter=";",skiprows=1))
	# ler ficheiros csv de teste para matriz numpy, e separar o label que está em col_label (deve ser a ultima coluna)
	primeiro=0
	for infile in glob.glob( os.test_path.join(test_path, '*.csv') ):
		if primeiro==0:
			test_dataset = np.loadtxt(infile, delimiter=";",skiprows=1)
			primeiro=1
		else:
			test_dataset=np.concatenate(test_dataset,np.loadtxt(infile, delimiter=";",skiprows=1))
	# ler ficheiro csv para matriz numpy, e separar o label que está em col_label (deve ser a ultima coluna)
	train_input_attributes = train_dataset[:,0:13]
	train_output_attributes = train_dataset[:,13]
	test_input_attributes = test_dataset[:,0:13]
	test_output_attributes = test_dataset[:,13]
	#print('Formato das variáveis de entrada (input variables): ',input_attributes.shape)
	#print('Formato da classe de saída (output variables): ',output_attributes.shape)
 	#print(X[0])
 	#print(Y[0])
	return (train_input_attributes,train_output_attributes,test_input_attributes,test_output_attributes)

def pre_process_data(train_input_attributes,train_output_attributes,test_input_attributes,test_output_attributes):

	return (scaled_train_input_attributes,scaled_train_output_attributes,scaled_test_input_attributes,scaled_test_output_attributes)

# Etapa 2 - Definir a topologia da rede (arquitectura do modelo)
def create_model():
 	model = Sequential()
 	model.add(Dense(12, input_dim=8, activation="relu", kernel_initializer="uniform"))
 	model.add(Dense(8, activation="relu", kernel_initializer="uniform"))
 	model.add(Dense(1, activation="sigmoid", kernel_initializer="uniform"))
 	return model

#util para visualizar a topologia da rede num ficheiro em pdf ou png
def print_model(model,fich):
	from keras.utils import plot_model
	plot_model(model, to_file=fich, show_shapes=True, show_layer_names=True)

# Etapa 3 - Compilar o modelo (especificar o modelo de aprendizagem a ser utilizado pela rede)
def compile_model(model):
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
	return model 

# Etapa 4 - treinar a rede (Fit the model) neste caso foi feito com os dados todos
def fit_model(model,input_attributes,output_attributes):
	history = model.fit(input_attributes, output_attributes,epochs=150, batch_size=10, verbose=2)
	return history

#utils para visulaização do historial de aprendizagem
def print_history_accuracy(history):
	print(history.history.keys())
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

def print_history_loss(history):
	print(history.history.keys())
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

 # Etapa 5 - Calcular o desempenho do modelo treinado (neste caso utilizando os dados usados no treino)
def model_evaluate(model,input_attributes,output_attributes):
 	print("###########inicio do evaluate###############################\n")
 	scores = model.evaluate(input_attributes, output_attributes)
 	print("\n metrica: %s: %.2f%%\n" % (model.metrics_names[1], scores[1]*100))

 # Etapa 6 - Utilizar o modelo treinado e escrever as previsões para novos casos
def model_print_predictions(model,input_attributes,output_attributes):
 	previsoes = model.predict(input_attributes)
 	# arredondar para 0 ou 1 pois pretende-se um output binário
 	LP=[]
 	for prev in previsoes:
 		LP.append(round(prev[0]))
 	#LP = [round(prev[0]) for prev in previsoes]
 	for i in range(len(output_attributes)):
 		print(" Class:",output_attributes[i]," previsão:",LP[i])
 		if i>10: break

 # Ciclo completo executando as Etapas 1,2,3,4,5 e 6
def ciclo_completo():
 	(train_input_attributes,train_output_attributes,test_input_attributes,test_output_attributes) = read_cvs_dataset()
 	#pré-processamento
 	scaled_train_input_attributes = preprocessing.StandardScaler().transform(train_input_attributes)
 	scaled_train_output_attributes = preprocessing.StandardScaler().transform(train_output_attributes)
 	scaled_test_input_attributes = preprocessing.StandardScaler().transform(test_input_attributes)
 	scaled_test_output_attributes = preprocessing.StandardScaler().transform(test_output_attributes)
 	model = create_model()
 	#print_model(model,"model_MLP.png")
 	compile_model(model)
 	history=fit_model(model,scaled_train_input_attributes,scaled_train_output_attributes)
 	#print_history_accuracy(history)
 	#print_history_loss(history)
 	model_evaluate(model,scaled_test_input_attributes,scaled_test_output_attributes)
 	model_print_predictions(model,scaled_test_input_attributes,scaled_test_output_attributes)


 	
if __name__ == '__main__':
 #opção 1 - ciclo completo
 	ciclo_completo()
 #opção 2 - ler,treinar o dataset e gravar. Depois ler o modelo e pesos e usar
 		#(input_attributes,output_attributes)=ciclo_ler_dataset_treinar_gravar()
 		#ciclo_ler_modelo_evaluate_usar(input_attributes,output_attributes)
