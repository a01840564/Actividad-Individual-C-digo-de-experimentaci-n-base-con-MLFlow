<!--
ARCHIVO EDITABLE PARA LA ENTREGA

- Reemplaza el espacio del repositorio con tu enlace real de GitHub.
- Al exportar, usa Arial o Times New Roman, tamaño 11 e interlineado 1.5.
- Mantén el análisis en una cuartilla y el contenido escrito en máximo dos,
  sin contar las capturas. Revisa las páginas después de exportar.
- Los bloques <div> separan páginas en los exportadores que admiten HTML.
  Puedes quitarlos o moverlos según el formato que prefieras.
- Las imágenes usan rutas relativas: conserva este archivo junto a evidence/.
- Exporta este Markdown desde tu editor; build_report.py genera el informe
  original a partir de los resultados JSON y no utiliza los cambios de este MD.
- Nombre final: TC5061_Experimentacion_MLFlow_HernandezAndres.pdf
-->

# Experimentación con MLflow

**Alumno:** José Andrés Hernández Guerra  
**Matrícula:** A01840564  
**Materia:** TC5061  
**Fecha:** 11 de octubre de 2026  
**Repositorio de GitHub:** [Pega aquí el enlace de tu repositorio]

## Análisis de resultados

En esta actividad pasé el modelo base de Titanic del notebook a un script de Python. La idea fue poder cambiar los parámetros desde la terminal y guardar en MLflow qué configuración usé y qué resultados dio. Conservé la regresión logística y las siete variables del modelo anterior. Trabajé con 891 pasajeros: 712 para entrenamiento y 179 para evaluación, siempre con la misma división y la semilla 42. En total hice seis configuraciones y una repetición de la base.

Para elegir el modelo me enfoqué en F1 de los sobrevivientes, porque hay más pasajeros que no sobrevivieron y la accuracy por sí sola no cuenta toda la historia. La mejor configuración para ese criterio fue `C=1`, `class_weight=balanced` y `solver=lbfgs`, con F1 de **0.7552**, accuracy de **0.8045** y ROC-AUC de **0.8462**. Frente a la base, el recall subió de 0.6667 a 0.7826, aunque la precision bajó de 0.7931 a 0.7297. Es decir, detectó a más sobrevivientes, pero también marcó como sobrevivientes a más personas que no lo fueron. En la matriz de confusión identificó correctamente a 54 de los 69 sobrevivientes; tuvo 20 falsos positivos y 15 falsos negativos.

El parámetro que más movió el F1 en las pruebas fue `C`: entre los valores probados, pasó de 0.6286 a 0.7344. Con `C=0.01`, la regularización fue más fuerte y el modelo dejó pasar a más sobrevivientes. Usar `balanced` mejoró el F1 de la base en 0.0308, mientras que cambiar a `liblinear` no cambió F1 ni accuracy. También vi que no hay una configuración que gane en todo: `C=10` tuvo la mayor accuracy (0.8101) y `C=0.1` el mayor ROC-AUC (0.8469).

La repetición de la base dio las mismas métricas y predicciones. Además, corregí el recorte de `Fare` para calcular su percentil solo con entrenamiento. Lo que todavía limita estos resultados es que usé un dataset pequeño y una sola división para comparar y elegir al ganador. Por eso no puedo asegurar que se comporte igual con datos nuevos. Como siguiente paso, probaría validación cruzada y dejaría un conjunto aparte para la evaluación final.

<div style="page-break-after: always;"></div>

## Cómo ejecutar el proyecto

El notebook quedó solo para explorar los datos. El entrenamiento, el preprocesamiento y la evaluación ahora se ejecutan desde `train.py`. Los comandos se corren desde la carpeta del proyecto, con el entorno de Python activado.

Para repetir las siete corridas:

```bash
python run_experiments.py
```

El comando utilizado para la configuración ganadora fue:

```bash
python train.py --C 1.0 --class-weight balanced --solver lbfgs --seed 42 --run-name 05_balanced
```

También se puede ejecutar `python train.py` sin argumentos para entrenar la base. Para ver los resultados en MLflow:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Después se abre `http://127.0.0.1:5000` y se selecciona el experimento **TC5061-Titanic-Experimentacion-Logistica**. Ahí se pueden seleccionar las corridas y usar **Compare** para revisar sus diferencias.

## Resultados de las corridas

| Corrida | C | class_weight | solver | Accuracy | F1 | ROC-AUC |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| 01_base | 1 | none | lbfgs | 0.8045 | 0.7244 | 0.8431 |
| 02_C_0.01 | 0.01 | none | lbfgs | 0.7821 | 0.6286 | 0.8345 |
| 03_C_0.1 | 0.1 | none | lbfgs | 0.7933 | 0.7087 | 0.8469 |
| 04_C_10 | 10 | none | lbfgs | 0.8101 | 0.7344 | 0.8431 |
| **05_balanced** | **1** | **balanced** | **lbfgs** | **0.8045** | **0.7552** | **0.8462** |
| 06_liblinear | 1 | none | liblinear | 0.8045 | 0.7244 | 0.8432 |
| 07_réplica_base | 1 | none | lbfgs | 0.8045 | 0.7244 | 0.8431 |

La réplica se usó para comprobar que los resultados se repiten; no se contó como una configuración adicional para elegir al ganador.

## Semilla, dependencias y registro

Usé la semilla **42** en Python, NumPy, la división de los datos y el clasificador. También fijé el cálculo numérico a un hilo. La imputación, el escalado, la codificación de categorías y el recorte de `Fare` se ajustan con los datos de entrenamiento.

El entorno utilizado fue Python **3.14.7**, MLflow **3.17.0**, scikit-learn **1.9.1**, NumPy **2.5.4**, pandas **3.0.6** y matplotlib **3.11.2**. Las versiones están en `requirements.txt` y `requirements-lock.txt`; el README explica cómo instalarlas.

Cada corrida guarda parámetros, métricas, gráficas, predicciones y el pipeline completo. También queda registrada la versión del código en Git y la huella del dataset. El modelo elegido se registró como **TC5061-Titanic-RegresionLogistica**, versión **1**, con el alias `candidate`. Se comprobó que al cargarlo de nuevo conserva las predicciones.

<div style="page-break-after: always;"></div>

## Evidencias de MLflow

### 1. Tabla de las siete corridas

En esta captura se ven las corridas del mismo experimento y sus métricas de evaluación.

![Tabla de siete corridas en MLflow](evidence/01_mlflow_runs.jpg)

<div style="page-break-after: always;"></div>

### 2. Vista de comparación

Seleccioné las siete corridas en **Compare Runs**. Esta parte de la vista muestra la réplica de la base, el cambio de solver y la configuración con balanceo de clases, junto con sus parámetros y métricas.

![Comparación de corridas en MLflow](evidence/02_mlflow_compare.jpg)

<div style="page-break-after: always;"></div>

### 3. Comparación de los valores de C

También comparé `C=10`, `C=0.1` y `C=0.01`, manteniendo los demás parámetros. Aquí se ve cómo cambia el F1 al modificar la regularización.

![Comparación de los valores de C](evidence/02b_mlflow_compare_remaining.jpg)

<div style="page-break-after: always;"></div>

### 4. Detalle de la corrida y sus artefactos

La corrida ganadora fue **05_balanced**, con ID `b8cb39f305004bf1a1df0c6a5b8fed1f`. En **Artifacts** quedaron la matriz de confusión, las métricas, las predicciones y los archivos que documentan cómo se obtuvo el resultado. El pipeline entrenado está disponible en **Models**.

![Artefactos y matriz de confusión del ganador](evidence/03_mlflow_artifacts.jpg)
