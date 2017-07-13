# rnn_chatbot

Simple rrn chatbot 


## Instalación

Para obtener el código puede clonarlo o bajarlo

### Clonar

    git clone git@github.com:ivanvladimir/rnn_chatbot.git

### Bajarlo

Bajarlo de a 
[aquí](https://github.com/ivanvladimir/rnn_chatbot/archive/master.zip)

Descomprimir

## Habilitar virtualenv

Instalar virtualenv

    sudo apt install virtualenv

Activar virtualenv en el directorio con el código

    source rnn_chatbot/bin/activate

## Instalar los requierements

En el directorio del código hacer

    pip install -r requierements.txt

## Ejecución

Existen tres modos:

* Entrenamiento (_train_)
* Prueba iterativa (_test_)
* Limpieza (_clean_)

### Entrenamiento

El sistema aprende a construir respuestas

    python3 train_rnn.py data/conversations.fruta.json

El sistema comienza a iterar sobre los datos para determinar los parámetros 
(pesos) y genera los siguientes archivos:

* Datos: matrices de entrenamiento y prueba (prueba ignorada por ahora)
* Vocabulario: lista de indices a palabras, ya que el modelo no entiende 
  palabras todo se pasa a indices y nos sirve para recuperar a que se refiere 
  cada indice
* Modelo: pesos del modelo

Una vez que se generan estos archivos el sistema los lee en lugar de generarlos 
y se pude iterar multiples veces

### Prueba iterativa

En este modo se lanza un pequeño chatbot que responde de forma automática

    python3 train_rnn.py data/conversations.fruta.json --mode test

para salir teclear _exit_ o _ctrl+d_.

### Limpieza

Usar con mucha precaución, borra los modelos y archivos auxiliares y se pierde 
el modelo. Excelente para cuando se quiera probar otros argumentos de la línea 
de comandos.

    python3 train_rnn.py data/conversations.fruta.json --mode clean
