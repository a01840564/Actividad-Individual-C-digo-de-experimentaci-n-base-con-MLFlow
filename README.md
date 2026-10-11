# TC5061 - Experimentación reproducible con MLflow

José Andrés Hernández Guerra · A01840564

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
- `results/selected_run/`: evaluación, gráficas y procedencia del ganador.
- `evidence/`: capturas reales de la interfaz de MLflow.
- `output/pdf/`: documento para Canvas.
- `build_report.py`: regenera el PDF con resultados y capturas existentes.

El almacenamiento local de MLflow (`mlflow.db`, `mlruns/`) y `.venv/` se excluye
de Git: contiene rutas locales y datos regenerables. Los resultados y capturas
sí se publican. Después de clonar, ejecuta `python run_experiments.py` para
recrear corridas y modelos en una nueva instalación de MLflow; los nuevos IDs
serán diferentes. Para explorar el notebook, abre Jupyter con este entorno
(instala Jupyter por separado si lo necesitas).

### Publicación pendiente

1. Crea en GitHub un repositorio vacío accesible para el profesor.
2. Ejecuta los siguientes comandos sustituyendo la URL:

```bash
git remote add origin https://github.com/TU_USUARIO/TC5061-Experimentacion-MLFlow.git
git push -u origin main
```

3. Incorpora el enlace real al PDF y regénéralo:

```bash
python build_report.py --repo-url https://github.com/TU_USUARIO/TC5061-Experimentacion-MLFlow
git add output/pdf
git commit -m "Incluir enlace del repositorio en la entrega"
git push
```

4. Comprueba el acceso del profesor y sube
`output/pdf/TC5061_Experimentacion_MLFlow_HernandezAndres.pdf` a Canvas.

El PDF sin `--repo-url` indica explícitamente que falta el enlace y debe
regenerarse antes de entregar. Ningún enlace se inventa.

Referencias técnicas: [MLflow sklearn](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html)
y [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
