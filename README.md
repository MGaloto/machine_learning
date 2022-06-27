# Aprendizaje Automatico

Iniciando repositorio sobre tecnicas de machine learning.


<p>
<a href="https://www.tensorflow.org/?hl=es-419" rel="nofollow"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/1915px-Tensorflow_logo.svg.png" align="right" width="130" style="max-width: 60%;"></a>
</p>


# Librerias

<ui>

<li>
{Matplotlib}
</li>

<li>
{Seaborn}
</li>

<li>
{Scikit-Learn}
</li>

<li>
{Tensorflow}
</li>

<li>
{NLTK}
</li>

 
 # Introduccion
  
Cualquier técnica que permita que la computadora replique de alguna forma procedimientos característicos de la inteligencia humana está encuadrada dentro de lo que llamamos <b>inteligencia artificial</b>. 

<b>Machine Learning</b> es un subconjunto de metodologías de inteligencia artificial que permite a las computadoras aprender con la experiencia, estos algoritmos son capaces de identificar patrones en conjuntos masivos de datos y de realizar análisis predictivos.

Dentro del Machine Learning hay un subconjunto llamado <b>Brain-inspired</b> que se basan principalmente en modelos y métodos que se replican el mecanismo del cerebro. 

Las <b>redes neuronales</b> son modelos simplificados que emulan el modo en el que el cerebro procesa la información, funciona basado en un número elevado de unidades de procesamiento simultaneas interconectadas que son versiones artificiales de las neuronas, y se organizan en capas. 

El <b>deep learning</b> es un subconjunto de las redes neuronales, lleva a cabo un proceso de machine learning utilizando una red neuronal artificial. 

Este repositorio contiene trabajos sobre entrenamiento de modelos con redes neuronales utilizando la libreria [Tensorflow](https://www.tensorflow.org/?hl=es-419). Para los trabajos de clasificacion se utiliza la libreria [Scikit-Learn](https://scikit-learn.org/stable/). 
  
<p align="center">
  <img 
    width="450"
    height="300"
    src="https://antoniofontanini.com/wp-content/uploads/2019/11/FOTOmit_image_datalabor2_2.gif"
  >
</p>

 # Inputs

 A continuacion se muestra una parte del codigo que se utiliza para generar los inputs de las Redes Neuronales:

```python
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

```
