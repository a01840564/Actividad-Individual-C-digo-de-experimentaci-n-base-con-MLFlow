# TC5061 - Experimentación reproducible con MLflow

José Andrés Hernández Guerra · A01840564

[Repositorio en GitHub](https://github.com/a01840564/Actividad-Individual-C-digo-de-experimentaci-n-base-con-MLFlow)

Baseline de regresión logística para predecir `Survived` en Titanic. El notebook se
conserva solo para exploración; `train.py` carga los datos, divide, preprocesa,
entrena, evalúa y registra cada corrida en MLflow.

## Instalación

Validado con Python **3.14.7** en macOS arm64. Para reproducir el entorno exacto,
usa la misma versión de Python y `requirements-lock.txt`. `requirements.txt`
contiene las versiones directas utilizadas, incluidas las herramientas del PDF.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

En Windows, activa con `.venv\Scripts\activate`. Ejecuta los comandos desde la
raíz del repositorio. `train.csv` está incluido: 891 filas y 12 columnas, dataset
[Titanic de Kaggle](https://www.kaggle.com/competitions/titanic/data).

## Entrenamiento

Sin argumentos utiliza la configuración base:

```bash
python train.py
```

Ejemplo con parámetros explícitos:

```bash
python train.py --C 1.0 --class-weight balanced --solver lbfgs --seed 42 --max-iter 1000 --run-name prueba_balanced
```

`python train.py --help` muestra todos los argumentos. Hiperparámetros principales:
`--C` (inverso de la regularización L2), `--class-weight` (`none`/`balanced`) y
`--solver` (`lbfgs`/`liblinear`); también `--max-iter`. Se parametrizan la semilla,
el tamaño de prueba, el percentil de Fare, las rutas y el nombre del experimento.

Cada corrida registra parámetros, accuracy, F1, ROC-AUC, precision y recall;
matriz de confusión, curva ROC, métricas JSON, predicciones y división de datos;
código fuente, dependencias, hashes SHA-256, commit y estado de Git. El pipeline
completo se guarda con firma y ejemplo de entrada mediante
`mlflow.sklearn.log_model`.

## Siete corridas y comparación

```bash
python run_experiments.py
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Abre <http://127.0.0.1:5000>, elige
`TC5061-Titanic-Experimentacion-Logistica`, muestra las columnas `C`,
`class_weight`, `solver`, `f1_test`, `accuracy_test` y `roc_auc_test`, selecciona
las corridas y pulsa **Compare**. En una corrida, abre **Artifacts** para las
gráficas, archivos de evaluación y código; el pipeline está en **Models**.

El ejecutor prueba seis configuraciones y una réplica de la base, con semilla 42
y división estratificada 80/20 en todas. Las variantes cambian un hiperparámetro
a la vez. La selección usa **F1 de la clase sobreviviente**; desempata con
ROC-AUC y nombre de corrida. El modelo elegido se registra como
`TC5061-Titanic-RegresionLogistica`, con versión y alias `candidate`.
Cada nueva ejecución añade siete corridas y una versión; no borra las anteriores.

El ejecutor verifica igualdad exacta de las cinco métricas, predicciones y
partición entre base y réplica; recarga el modelo guardado y compara predicciones;
verifica que el recorte de Fare se aprendió solo con entrenamiento.

En la entrega preparada se seleccionó `C=1`, `class_weight=balanced`,
`solver=lbfgs`: F1 **0.7552**, accuracy **0.8045**, ROC-AUC **0.8462**. Entre los
rangos probados, `C` produjo el mayor cambio de F1. El análisis completo y sus
limitaciones están en `results/analysis.md` y en el PDF.

## Reproducibilidad y metodología

La semilla fija `random`, NumPy, la partición estratificada y el clasificador.
Los cálculos numéricos usan un hilo. La imputación, escalado, one-hot y percentil
de Fare se ajustan dentro del pipeline con el entrenamiento. En el notebook
anterior el percentil se calculaba con todo el dataset; se corrigió esa filtración.
Se preservan las siete variables predictoras del baseline y el umbral 0.5.

La igualdad se verificó en el entorno documentado; versiones o plataformas
distintas pueden producir pequeñas diferencias numéricas. F1 privilegia el
equilibrio entre precision y recall de sobrevivientes. Seleccionar con esta
misma partición hace que sus métricas no sean una estimación independiente del
modelo ganador: se requiere validación cruzada y una prueba final separada para
afirmaciones de generalización.

## Resultados y entrega

- `results/comparison.csv`: métricas reales y parámetros de las siete corridas.
- `results/commands.txt`: comandos exactos ejecutados.
- `results/experiment_summary.json`: IDs, selección y versión del modelo.
- `results/reproducibility.json`: evidencia de reproducibilidad y recarga.
- `results/default_execution_verified.json`: ejecución real sin argumentos.
- `results/selected_run/`: evaluación, gráficas y procedencia del ganador.
- `evidence/`: capturas reales de la interfaz de MLflow.
- `output/pdf/`: documento para Canvas.
- `TC5061_Experimentacion_MLFlow_HernandezAndres.md`: informe con análisis,
  comandos y capturas.
- `build_report.py`: exporta el informe Markdown a PDF con Arial 11 e
  interlineado 1.5.

Para generar el PDF después de editar el informe:

```bash
python build_report.py
```

El archivo se guarda en
`output/pdf/TC5061_Experimentacion_MLFlow_HernandezAndres.pdf`. El generador usa
las fuentes Arial de macOS. En otro sistema, indica `--font-dir` con una carpeta
que contenga `Arial.ttf` y `Arial Bold.ttf`.

El almacenamiento local de MLflow (`mlflow.db`, `mlruns/`) y `.venv/` se excluye
de Git: contiene rutas locales y datos regenerables. Los resultados y capturas
sí se publican. Después de clonar, ejecuta `python run_experiments.py` para
recrear corridas y modelos en una nueva instalación de MLflow; los nuevos IDs
serán diferentes. Para explorar el notebook, abre Jupyter con este entorno
(instala Jupyter por separado si lo necesitas).

El `.gitignore` también excluye otros entornos de Python, cachés, archivos de
configuración local, variables de entorno, claves, logs, temporales y bases de
datos locales. Se conservan el dataset, las evidencias, los resultados de la
comparación y los documentos de entrega. Las plantillas `.env.example` y
`.env.sample` pueden incluirse si se necesitan.

Referencias técnicas: [MLflow sklearn](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html)
y [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
