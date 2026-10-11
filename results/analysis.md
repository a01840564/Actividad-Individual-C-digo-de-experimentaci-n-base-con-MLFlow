# Análisis de resultados

En esta actividad pasé el modelo de Titanic del notebook a un archivo de Python. Así puedo cambiar sus parámetros desde la terminal y guardar los resultados en MLflow. Conservé la regresión logística del trabajo anterior. Probé seis configuraciones y repetí la base para revisar que diera el mismo resultado.

**Datos utilizados:**  
Total de pasajeros: 891  
Para entrenar: 712  
Para evaluar: 179

Elegí el modelo con el mejor F1 porque me interesaba identificar bien a los sobrevivientes. No me basé solo en el porcentaje de aciertos, ya que en los datos hay más personas que no sobrevivieron.

**Configuración elegida:** `C=1`, `class_weight=balanced`, `solver=lbfgs`.  
F1: **0.7552**  
Accuracy: **0.8045**, cerca del 80 % de aciertos  
ROC-AUC: **0.8462**

El F1 de la base fue 0.7244. Al usar `balanced`, el modelo encontró a más sobrevivientes, aunque también cometió más errores al decir que alguien sobrevivió. Identificó correctamente a 54 de los 69 sobrevivientes.

El parámetro que más cambió los resultados fue `C`. En esas pruebas, el F1 fue de 0.6286 a 0.7344. Con `C=0.01`, el modelo tuvo más dificultad para reconocer a los sobrevivientes. Cambiar a `liblinear` no mejoró el F1 ni la accuracy. `C=10` tuvo la mayor accuracy, pero elegí `balanced` por su mejor F1.

La repetición de la base dio los mismos resultados. También corregí el cálculo usado para limitar los valores altos de `Fare`, que es el precio del boleto. Ahora se calcula solo con los datos de entrenamiento.

Una limitación es que usé pocos datos y una sola división para comparar los modelos. Por eso no puedo asegurar que el ganador funcione igual con datos nuevos. Después probaría varias divisiones y dejaría datos aparte para una evaluación final.
