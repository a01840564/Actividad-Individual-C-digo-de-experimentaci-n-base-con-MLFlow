# TC5061 - Experimentación con MLflow

José Andrés Hernández Guerra · A01840564

[Repositorio en GitHub](https://github.com/a01840564/Actividad-Individual-C-digo-de-experimentaci-n-base-con-MLFlow)

Regresión logística para predecir la supervivencia en Titanic. El notebook se
conserva para explorar los datos. El entrenamiento se ejecuta desde `train.py`.

## Instalación

Entorno utilizado: Python 3.14.7 en macOS arm64. Ejecuta los comandos desde la
carpeta del proyecto. Las versiones están en `requirements.txt` y el entorno
completo de experimentación en `requirements-lock.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

En Windows, activa el entorno con `.venv\Scripts\activate`.

## Entrenamiento

Para usar los valores por defecto:

```bash
python train.py
```

Para cambiar los parámetros:

```bash
python train.py --C 1.0 --class-weight balanced --solver lbfgs --seed 42 --max-iter 1000 --run-name prueba_balanced
```

`--C`, `--class-weight`, `--solver` y `--max-iter` controlan el modelo.
`--seed` fija la semilla. `python train.py --help` muestra todos los argumentos.

Para repetir las seis configuraciones y la réplica de la base:

```bash
python run_experiments.py
```

## Comparación en MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Abre <http://127.0.0.1:5000> y selecciona
`TC5061-Titanic-Experimentacion-Logistica`. Selecciona las corridas y presiona
**Compare**. En **Artifacts** están las gráficas, métricas y predicciones.
El modelo completo está en **Models**.

Cada corrida registra sus parámetros, accuracy, F1, ROC-AUC, precision y recall.
También guarda el código, las dependencias y la versión de Git. El ganador se
elige por F1 y se registra como `TC5061-Titanic-RegresionLogistica`, con el alias
`candidate`. Cada ejecución agrega siete corridas y una versión del modelo.

## Reproducibilidad

Todas las corridas usan la semilla 42 y la misma división estratificada 80/20.
La preparación de los datos se ajusta solo con entrenamiento. El ejecutor
comprueba que la base y su réplica tengan las mismas métricas y predicciones,
y que el modelo guardado conserve sus predicciones al cargarlo de nuevo.

El mejor F1 registrado fue **0.7552**, con `C=1`, `class_weight=balanced` y
`solver=lbfgs`. Se usó una sola división para comparar y elegir al ganador.
Hace falta una evaluación con datos aparte para comprobar su desempeño en
casos nuevos.

## Archivos de la actividad

- `train.py` y `preprocessing.py`: entrenamiento y preparación de los datos.
- `run_experiments.py`: ejecución y comparación de las configuraciones.
- `train.csv`: dataset [Titanic de Kaggle](https://www.kaggle.com/competitions/titanic/data).
- Notebook: exploración de los datos.
- `results/`: comparación, comandos, comprobación de reproducibilidad y evaluación del ganador.
- `evidence/`: capturas de MLflow.
- Informe Markdown y `output/pdf/`: análisis y documento de entrega.

El `.gitignore` excluye entornos de Python, cachés, credenciales y almacenamiento
local de MLflow. Después de clonar, ejecuta `python run_experiments.py` para
recrear las corridas y modelos. Los IDs nuevos serán distintos.
