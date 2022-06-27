from termcolor import colored
import tensorflow as tf
import matplotlib.pyplot as plt


class Model():
    def __init__(self):
        self.links = []

    def _colored(self, r, g, b, text):
        return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)
      
    def _input(self):
        while True:
            modelo = input('\nDesea entrenar una Red Neuronal? (Si/No): \n')
            if modelo == 'No':
                print('Muchas gracias\n')
                break
            elif modelo == 'Si':
                print('Excelente...')
                scaler   = input('Desea escalar los valores? (Si/No)').lower()
                neuronas = int(input(f'Ingrese la cantidad de Neuronas:'))
                outputs  = int(input(f'Ingrese la cantidad de Outputs:'))
                capas    = int(input(f'Ingrese la cantidad de Capas Ocultas:'))
                epocas   = int(input(f'Ingrese la cantidad de Epocas:'))
                    
            if modelo == 'Si': break
            else:
                print(self._colored(238, 75, 43, '\nADVERTENCIA: Por favor colocar (Si/No)\n'))
        if modelo == 'Si':
            diccionario = {'capas' : capas, 'neuronas' : neuronas, 'outputs': outputs, 'scaler' : scaler, 'epocas' : epocas}
            print(self._colored(238, 75, 43,f'La cantidad de Capas es: {capas}'))
            print(self._colored(238, 75, 43,f'La cantidad de Neuronas es: {neuronas}'))
            print(self._colored(238, 75, 43,f'La cantidad de Epocas es: {epocas}'))
            return diccionario
        else:
            return None
        
    def _get_results(self, resultado):
        return print(resultado)

    def _get_model(self, inputs, data):
        (X_train, y_train), (X_test, y_test) = data
        capas    = inputs['capas']
        neuronas = inputs['neuronas']
        outputs  = inputs['outputs']
        scaler   = inputs['scaler']
        epocas   = inputs['epocas']
        model    = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten())
        if 'si' in scaler:
            X_train = tf.keras.utils.normalize(X_train, axis = 1)
            X_test = tf.keras.utils.normalize(X_test, axis = 1)
        for i in range(1, capas + 1):
            print(f'Añadiendo capa Numero {i} al Modelo')
            capa = tf.keras.layers.Dense(neuronas, activation = tf.nn.relu)
            model.add(capa)
        model.add(tf.keras.layers.Dense(outputs, activation = tf.nn.softmax))
        print(f'\nINPUTS:\nCapas:     {capas}\nNeuronas: {neuronas}\nOutputs:  {outputs}\nScaler:   {scaler}\n')
        print('\nEntrenando el modelo.....\n')
        print(self._colored(233, 172, 13,f'Por default: Optimizer adam, loss: categorical, metrics: accuracy'))
        model.compile(optimizer = 'adam',
                      loss      = 'sparse_categorical_crossentropy',
                      metrics   = ['accuracy'])
        historial = model.fit(X_train, y_train, epochs = epocas, verbose = 0)
        perdida = [round(loss, 3) for loss in historial.history['loss']]
        acc     = [round(acc, 3)  for acc  in historial.history['accuracy']]
        print(self._colored(233, 128 ,13,f'\n\nPerdida:  {perdida}'))
        print(self._colored(233, 75, 13,f'Accuracy: {acc}\n\n'))
        print(self._colored(233, 75, 13,'OK Model\n'))
        return historial

    def _get_plots(self, metricas, hyper_p, y_label = 'Metrics'):
        scaler = hyper_p['scaler']
        capas = hyper_p['capas']
        neuronas = hyper_p['neuronas']
        epocas = hyper_p['epocas']
        plt.figure(figsize = (8,5))
        plt.plot(
                metricas,
                color = 'blue', 
                linestyle = 'dashed', 
                marker = '*', 
                markerfacecolor = 'red', 
                markersize = 10)
        plt.title(f'Datos {scaler.title()} escalados.', fontsize=16)
        plt.suptitle(f'Capas: {capas}, Neuronas: {neuronas}, Epocas: {epocas}', fontsize=16)
        plt.xlabel('Epocas', fontsize=16); plt.ylabel(y_label, fontsize=16)
        return plt.grid(); plt.show()
        