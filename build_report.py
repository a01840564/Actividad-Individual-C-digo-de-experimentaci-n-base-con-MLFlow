"""Genera el PDF de entrega a partir de resultados y capturas reales de MLflow."""

import argparse
from html import escape
import json
from pathlib import Path
from urllib.parse import urlparse

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                               PageBreak, Table, TableStyle, Image)

ROOT = Path(__file__).resolve().parent


def analysis(summary):
    rows = summary["runs"]
    best, base = summary["best"], rows[0]
    c_runs = rows[:4]
    delta_c = max(r["f1_test"] for r in c_runs) - min(r["f1_test"] for r in c_runs)
    return [
        "Se migró el baseline Titanic del notebook a un script parametrizado de regresión logística. "
        "Se evaluaron seis configuraciones y una réplica bajo el mismo experimento, con 891 pasajeros, "
        "siete variables predictoras y una división estratificada 80/20 (712 para entrenamiento y 179 "
        "para evaluación). Se fijó la semilla 42 y el umbral de clasificación 0.5. La comparación de "
        "MLflow permitió contrastar los parámetros y métricas de todas las corridas.",
        f"Se eligió F1 de la clase sobreviviente como criterio antes de comparar, por el desequilibrio "
        f"de clases (549 no sobrevivientes y 342 sobrevivientes). La mejor configuración fue "
        f"C={best['C']:g}, class_weight={best['class_weight']}, solver={best['solver']} y max_iter=1000: "
        f"F1={best['f1_test']:.4f}, accuracy={best['accuracy_test']:.4f} y ROC-AUC={best['roc_auc_test']:.4f}. "
        f"Frente a la base (F1={base['f1_test']:.4f}), el recall subió de {base['recall_test']:.4f} "
        f"a {best['recall_test']:.4f}, mientras la precision bajó de {base['precision_test']:.4f} "
        f"a {best['precision_test']:.4f}. El ganador recuperó 54 de los 69 sobrevivientes, con 20 falsos "
        "positivos y 15 falsos negativos. C=10 obtuvo la mayor accuracy (0.8101) y C=0.1 el mayor "
        "ROC-AUC (0.8469); la selección depende de la métrica priorizada.",
        f"C fue el hiperparámetro más influyente dentro de los valores probados: su variación "
        f"entre 0.01, 0.1, 1 y 10 cambió F1 de 0.6286 a 0.7344 (rango {delta_c:.4f}). "
        "C=0.01, con regularización más fuerte, redujo el recall a 0.4783. El balanceo de clases "
        f"mejoró F1 en {best['f1_test']-base['f1_test']:.4f} respecto a la base; cambiar a liblinear "
        "no cambió F1 ni accuracy. Se modificó un hiperparámetro a la vez, por lo que no se evaluaron "
        "sus interacciones ni puede extrapolarse esta influencia a otros rangos.",
        "La réplica produjo métricas, predicciones y partición idénticas. Se corrigió la filtración "
        "del notebook: el percentil 99 de Fare se aprende solo con entrenamiento, junto con la "
        "imputación, escalado y codificación. Se registraron el pipeline completo, dependencias, "
        f"huella del dataset y commit del código; el ganador quedó en Model Registry, versión {summary['registry_version']}, "
        "alias candidate. Las principales limitaciones son el dataset pequeño, una única partición "
        "y el uso de esa misma evaluación para seleccionar y reportar el ganador. Las métricas "
        "no estiman de forma independiente su generalización. El siguiente paso sería validar "
        "con particiones cruzadas y reservar una prueba final separada."
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-url", help="Enlace real al repositorio GitHub individual")
    parser.add_argument("--font-dir", type=Path, default=Path("/System/Library/Fonts/Supplemental"),
                        help="Carpeta con Arial.ttf y Arial Bold.ttf")
    args = parser.parse_args()
    if args.repo_url:
        parsed = urlparse(args.repo_url)
        if parsed.scheme != "https" or parsed.netloc != "github.com" or len(parsed.path.strip('/').split('/')) != 2:
            parser.error("--repo-url debe ser un enlace HTTPS real de GitHub con usuario y repositorio")
    for label, filename in [("Arial", "Arial.ttf"), ("Arial-Bold", "Arial Bold.ttf")]:
        path = args.font_dir / filename
        if not path.exists():
            parser.error(f"Falta la fuente {path}; indica --font-dir con los archivos de Arial")
        pdfmetrics.registerFont(TTFont(label, str(path)))
    pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial", boldItalic="Arial-Bold")
    summary = json.loads((ROOT / "results/experiment_summary.json").read_text())
    paragraphs = analysis(summary)
    styles = {
        "body": ParagraphStyle("body", fontName="Arial", fontSize=11, leading=16.5, spaceAfter=10),
        "title": ParagraphStyle("title", fontName="Arial-Bold", fontSize=11, leading=16.5, spaceAfter=13),
        "heading": ParagraphStyle("heading", fontName="Arial-Bold", fontSize=11, leading=16.5, spaceAfter=9),
        "caption": ParagraphStyle("caption", fontName="Arial", fontSize=11, leading=16.5, spaceAfter=10),
    }
    story = []
    def p(text, style="body"):
        return Paragraph(text, styles[style])
    story += [p("Experimentación reproducible con MLflow", "title"),
              p("TC5061 | José Andrés Hernández Guerra | A01840564<br/>11 de octubre de 2026"),
              p("Análisis de resultados", "heading")]
    story += [p(escape(t)) for t in paragraphs]
    story += [PageBreak(), p("Ejecución y trazabilidad", "title")]
    if args.repo_url:
        link = escape(args.repo_url)
        story += [p(f'Repositorio individual: <link href="{link}" color="#184c72">{link}</link>')]
    else:
        story += [p("<b>Repositorio GitHub pendiente de publicación.</b> Antes de entregar, regenerar este "
                    "documento con el enlace real mediante <b>build_report.py --repo-url</b>.")]
    story += [p("Entorno utilizado: Python 3.14.7, MLflow 3.17.0, scikit-learn 1.9.1, NumPy 2.5.4, "
                "pandas 3.0.6 y matplotlib 3.11.2. Dependencias directas en requirements.txt y "
                "entorno completo en requirements-lock.txt."),
              p("Comandos exactos desde la raíz del proyecto", "heading"),
              p("python run_experiments.py<br/>"
                "python train.py --C 1.0 --class-weight balanced<br/>"
                "--solver lbfgs --seed 42 --run-name 05_balanced"),
              p("El comando de entrenamiento anterior se escribe en una sola línea. "
                "Sin argumentos, <b>python train.py</b> ejecuta la base. "
                "results/commands.txt conserva los siete comandos ejecutados."),
              p("Interfaz: mlflow ui --backend-store-uri sqlite:///mlflow.db<br/>"
                "--host 127.0.0.1 --port 5000<br/>Abrir http://127.0.0.1:5000 y seleccionar el experimento "
                "TC5061-Titanic-Experimentacion-Logistica."),
              p("Resumen de las corridas registradas", "heading")]
    rows = [["Corrida", "C", "Peso", "Solver", "Acc.", "F1", "AUC"]]
    for r in summary["runs"]:
        rows.append([r["run_name"].replace("07_baseline_replica", "07_réplica base"), f"{r['C']:g}",
                     r["class_weight"], r["solver"], f"{r['accuracy_test']:.4f}",
                     f"{r['f1_test']:.4f}", f"{r['roc_auc_test']:.4f}"])
    table = Table(rows, colWidths=[118, 30, 65, 65, 58, 58, 58], rowHeights=24)
    table.setStyle(TableStyle([("FONTNAME", (0,0), (-1,-1), "Arial"),
                               ("FONTSIZE", (0,0), (-1,-1), 11),
                               ("FONTNAME", (0,0), (-1,0), "Arial-Bold"),
                               ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e5edf4")),
                               ("BACKGROUND", (0,5), (-1,5), colors.HexColor("#e8f3eb")),
                               ("LINEBELOW", (0,0), (-1,0), 0.6, colors.HexColor("#8ca3b7")),
                               ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                               ("LEFTPADDING", (0,0), (-1,-1), 4)]))
    story += [table, Spacer(1,12),
              p("La comparación excluye la réplica para seleccionar el ganador. "
                "La verificación de reproducibilidad y recarga está en results/reproducibility.json. "
                f"Commit de entrenamiento: {escape(json.loads((ROOT/'results/selected_run/provenance.json').read_text())['git_commit'][:12])}. "
                "Cada corrida guarda SHA-256 de código y datos."),
              p("Referencias: <link href='https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html' "
                "color='#184c72'>MLflow sklearn</link>; "
                "<link href='https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html' "
                "color='#184c72'>scikit-learn LogisticRegression</link>.")]
    captures = [
        ("01_mlflow_runs.jpg", "Evidencia 1 - Tabla de corridas",
         "Captura real de MLflow: siete corridas del mismo experimento. Se muestran accuracy, F1, ROC-AUC y C."),
        ("02_mlflow_compare.jpg", "Evidencia 2 - Vista de comparación",
         "Compare Runs con siete corridas seleccionadas. Primera parte de la vista horizontal: réplica, liblinear y balanced. F1 fundamenta la selección."),
        ("02b_mlflow_compare_remaining.jpg", "Evidencia 2 - Influencia de C",
         "Comparación complementaria de C=10, C=0.1 y C=0.01, manteniendo los demás parámetros. El resumen de las siete corridas aparece en la página 2."),
        ("03_mlflow_artifacts.jpg", "Evidencia 3 - Detalle y artefactos",
         f"Corrida ganadora 05_balanced, ID {summary['best']['run_id']}. "
         "Artifacts contiene la evaluación, matriz de confusión y procedencia; el pipeline completo se guarda en Models."),
    ]
    for filename, title, caption in captures:
        path = ROOT / "evidence" / filename
        if not path.exists():
            raise FileNotFoundError(f"Falta la captura real: {path}")
        image = Image(str(path))
        ratio = min(504 / image.imageWidth, 580 / image.imageHeight)
        image.drawWidth = image.imageWidth * ratio
        image.drawHeight = image.imageHeight * ratio
        story += [PageBreak(), p(title, "title"), p(escape(caption), "caption"), image]
    output = ROOT / "output/pdf/TC5061_Experimentacion_MLFlow_HernandezAndres.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    def footer(canvas, document):
        canvas.setFont("Arial", 11)
        canvas.setFillColor(colors.HexColor("#526476"))
        canvas.drawString(54, 28, "TC5061 | Experimentación con MLflow")
        canvas.drawRightString(558, 28, str(document.page))
    document = SimpleDocTemplate(str(output), pagesize=letter, rightMargin=54, leftMargin=54,
                                 topMargin=45, bottomMargin=45,
                                 title="TC5061 - Experimentación con MLflow", author="José Andrés Hernández Guerra")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    (ROOT / "results/analysis.md").write_text("# Análisis de resultados\n\n" + "\n\n".join(paragraphs) + "\n")
    print(output)


if __name__ == "__main__":
    main()
