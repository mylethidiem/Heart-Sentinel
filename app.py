import os
import gradio as gr
import plotly.graph_objects as go
import pandas as pd
import ui_template

from src.heart_disease_core import (
    CLEVELAND_FEATURES_ORDER,
    load_cleveland_dataframe, fit_all_models, predict_all, example_patient, get_example_labels
)

APP_PRIMARY = ui_template.PRIMARY_COLOR
APP_ACCENT = ui_template.ACCENT_COLOR
APP_BG = "#F7FAFC"

STATE = {
    "df": None,
    "models": None,
    "metrics": None,
}

DATA_PATH = "data/cleveland.csv"

ui_template.set_meta(
    project_name="Heart Sentinel",
    year="2025",
    description="Predict heart disease risk from patient data with optimized ML models trained on the Cleveland dataset.",
    meta_items=[
        ("Dataset", "Cleveland Heart Disease"),
        ("Models", "Decision Tree, k-NN, Naive Bayes, Random Forest, AdaBoost, Gradient Boosting, XGBoost"),
    ],
)

force_light_theme_js = """
() => {
  const params = new URLSearchParams(window.location.search);
  if (!params.has('__theme')) {
    params.set('__theme', 'light');
    window.location.search = params.toString();
  }
}
"""

def init_page(train_split):
    """Load dataset, train models, and return status, preview, metrics."""
    if not os.path.exists(DATA_PATH):
        msg = f"❌ Dataset not found at '{DATA_PATH}'. Please place Cleveland CSV there."
        return msg, pd.DataFrame(), pd.DataFrame()

    df = load_cleveland_dataframe(file_path=DATA_PATH)

    # Convert train_split percentage to test_size for sklearn
    test_size = (100 - train_split) / 100
    models, metrics = fit_all_models(df, test_size=test_size)
    STATE["df"] = df
    STATE["models"] = models
    STATE["metrics"] = metrics

    head = df.head(8)
    msg = f"✅ Cleveland dataset loaded from `data/cleveland.csv` and models trained ({train_split}/{100-train_split} split)."
    return msg, head, metrics


def fill_example(idx_text: str):
    import re
    match = re.search(r'Example (\d+)', idx_text)
    if match:
        idx = int(match.group(1)) - 1
    else:
        idx = 1

    ex = example_patient(idx)
    return [ex[c] for c in CLEVELAND_FEATURES_ORDER]


def _bar_for_models(results: dict):
    names = list(results.keys())
    confidences = []
    predictions_text = []
    bar_colors = []
    line_colors = []
    line_widths = []

    for n in names:
        r = results[n]
        if r["label"] == 1:
            confidences.append(r["prob_1"])
            predictions_text.append("🫀 Heart Disease")
            bar_colors.append("#C4314B")
        else:
            confidences.append(r["prob_0"])
            predictions_text.append("✅ No Heart Disease")
            bar_colors.append("#2E7D32")
        line_colors.append("rgba(0,0,0,0.15)")
        line_widths.append(1.0)

    if "Ensemble (Soft Voting)" in names:
        idx = names.index("Ensemble (Soft Voting)")
        line_colors[idx] = "#000000"
        line_widths[idx] = 2.5

    fig = go.Figure()
    fig.add_bar(x=names, y=confidences, text=predictions_text, textposition="auto")
    fig.update_layout(
        title="Model Predictions",
        yaxis_title="Prediction Confidence",
        xaxis_title="Model",
        yaxis=dict(range=[0, 1]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="black", size=12),
        height=420,
        margin=dict(l=30, r=20, t=60, b=40)
    )
    fig.data[0].marker.color = bar_colors
    fig.data[0].marker.line.color = line_colors
    fig.data[0].marker.line.width = line_widths
    return fig


def run_predict(*vals):
    if STATE["df"] is None or STATE["models"] is None:
        return None, "❌ Models not initialized. Reload the app.", pd.DataFrame()

    input_dict = {col: vals[i] for i, col in enumerate(CLEVELAND_FEATURES_ORDER)}
    results = predict_all(STATE["models"], input_dict)

    final = results["Ensemble (Soft Voting)"]
    ensemble_color = "#C4314B" if final["label"] == 1 else "#2E7D32"
    ensemble_prediction = "🫀 **Heart Disease Detected**" if final["label"] == 1 else "✅ **No Heart Disease**"

    ensemble_md = f"""
<div style=\"border: 3px solid {ensemble_color}; border-radius: 10px; padding: 20px; margin: 15px 0; background: white;\">
    <h3 style=\"margin: 0 0 15px 0; color: {ensemble_color};\">🎯 Ensemble Prediction (Final Result)</h3>
    <p style=\"margin: 10px 0; font-size: 18px; color: black;\"><strong>{ensemble_prediction}</strong></p>
    <p style=\"margin: 5px 0; font-size: 16px; color: black;\"><strong>Confidence:</strong> {final['prob_1']:.1%}</p>
</div>
"""

    model_predictions = []
    for name, r in results.items():
        prediction_text = "🫀 **Heart Disease Detected**" if r["label"] == 1 else "✅ **No Heart Disease**"
        confidence = r["prob_1"] if r["label"] == 1 else r["prob_0"]
        color = "#C4314B" if r["label"] == 1 else "#2E7D32"

        model_predictions.append(f"""
<div style=\"border: 2px solid {color}; border-radius: 8px; padding: 15px; margin: 10px 0; background: white;\">
    <h4 style=\"margin: 0 0 10px 0; color: {color};\">{name}</h4>
    <p style=\"margin: 5px 0; font-size: 16px; color: black;\"><strong>Prediction:</strong> {prediction_text}</p>
    <p style=\"margin: 5px 0; font-size: 14px; color: black;\"><strong>Confidence:</strong> {confidence:.1%}</p>
    <p style=\"margin: 5px 0; font-size: 12px; color: #666;\">
        P(No disease): {r['prob_0']:.3f} | P(Heart disease): {r['prob_1']:.3f}
    </p>
</div>
""")

    all_predictions = "\n".join(model_predictions)

    rows = []
    for name, r in results.items():
        confidence = r["prob_1"] if r["label"] == 1 else r["prob_0"]
        rows.append({
            "Model": name,
            "Prediction": "Heart Disease" if r["label"] == 1 else "No Heart Disease",
            "Confidence": f"{confidence:.1%}",
            "P(No disease)": round(r["prob_0"], 3),
            "P(Heart disease)": round(r["prob_1"], 3),
        })
    table_df = pd.DataFrame(rows)

    fig = _bar_for_models(results)

    return fig, "\n".join(model_predictions), table_df


with gr.Blocks(theme="gstaff/sketch", css=ui_template.custom_css, fill_width=True, js=force_light_theme_js) as demo:
    ui_template.create_header()
    gr.HTML(ui_template.render_info_card(icon="🫀", title="About this demo"))
    gr.HTML(ui_template.render_disclaimer(
        text=(
            "This interactive heart disease prediction demo is provided strictly for educational purposes. "
            "It is not intended for clinical use and must not be relied upon for medical advice, diagnosis, "
            "treatment, or decision-making. Always consult a qualified healthcare professional."
        )
    ))
    gr.Markdown("### 🫀 **How to Use**: Enter patient features → Run prediction → View ensemble results!")

    with gr.Row(equal_height=False, variant="panel"):
        # LEFT: data preview + inputs
        with gr.Column(scale=45):
            with gr.Accordion("📁 Dataset & Model Status", open=True):
                with gr.Row():
                    train_split = gr.Slider(
                        minimum=60,
                        maximum=90,
                        value=80,
                        step=5,
                        label="Training Split (%)",
                        info="Percentage of data used for training (remaining for validation)"
                    )
                    retrain_btn = gr.Button("🔄 Retrain Models", variant="secondary")

                status_md = gr.Markdown("Loading dataset and training models...")
                preview = gr.DataFrame(label="Cleveland Preview (first rows)", interactive=False)
                metrics_df = gr.DataFrame(label="Model Performance Comparison (Validation Set Results)", interactive=False)

            with gr.Accordion("✍️ Enter Patient Features", open=True):
                with gr.Row():
                    age = gr.Number(label="age (years)", value=58)
                    sex = gr.Dropdown(label="sex (0=female, 1=male)", choices=[0, 1], value=1)
                    cp = gr.Dropdown(label="cp (chest pain type 1..4)", choices=[1, 2, 3, 4], value=2)
                    trestbps = gr.Number(label="trestbps (resting BP mmHg)", value=130)

                with gr.Row():
                    chol = gr.Number(label="chol (serum cholesterol mg/dl)", value=250)
                    fbs = gr.Dropdown(label="fbs (>120 mg/dl? 1/0)", choices=[0, 1], value=0)
                    restecg = gr.Dropdown(label="restecg (0..2)", choices=[0, 1, 2], value=1)
                    thalach = gr.Number(label="thalach (max heart rate)", value=150)

                with gr.Row():
                    exang = gr.Dropdown(label="exang (exercise angina 1/0)", choices=[0, 1], value=0)
                    oldpeak = gr.Number(label="oldpeak (ST depression)", value=1.0)
                    slope = gr.Dropdown(label="slope (1..3)", choices=[1, 2, 3], value=1)
                    ca = gr.Dropdown(label="ca (major vessels 0..3)", choices=[0, 1, 2, 3], value=0)

                thal = gr.Dropdown(label="thal (3=normal, 6=fixed, 7=reversible)", choices=[3, 6, 7], value=3)

                with gr.Row():
                    # Get actual labels from the dataset - only 2 examples
                    try:
                        labels = get_example_labels()
                        choices = []
                        # Only use first two examples: one no disease, one disease
                        for i in range(min(2, len(labels))):
                            label_text = "No Heart Disease" if labels[i] == 0 else "Heart Disease"
                            choices.append(f"Example {i+1} ({label_text})")
                        default_choice = choices[0] if choices else "Example 1"
                    except:
                        choices = ["Example 1 (No Heart Disease)", "Example 2 (Heart Disease)"]
                        default_choice = "Example 1 (No Heart Disease)"

                    ex_selector = gr.Dropdown(
                        label="Select Example Patient",
                        choices=choices,
                        value=default_choice
                    )
                    predict_btn = gr.Button("🔍 Predict", variant="primary")

        # RIGHT: outputs
        with gr.Column(scale=55):
            gr.Markdown("### 📈 Model Predictions")
            bar_out = gr.Plot(label="Model Predictions Overview")
            sub_md = gr.Markdown("**Individual Model Results**")
            table_out = gr.DataFrame(label="All Model Predictions", interactive=False)

    gr.Markdown("""
    ## 📋 **Notes**

    - **Models are trained at launch** on `data/cleveland.csv` with customizable train/validation split (default 80/20).
    - **Target is binarized automatically** (0 = no disease, >0 = disease).
    - **Retrain functionality**: Adjust the split ratio and click "🔄 Retrain Models" to see how data size affects performance.
    - **Seven optimized models are compared**: Decision Tree, k-NN, Naive Bayes, Random Forest, AdaBoost, Gradient Boosting, and XGBoost.
    - **Hyperparameters are optimized** for heart disease prediction tasks using best practices.
    - **Ensemble uses weighted soft voting** with optimized weights based on model performance.
    - **Best performing model** on test set is highlighted with 🏆 in the validation metrics table.
    - **Optimization highlights**:
      - Decision Tree: entropy criterion, balanced classes, optimal depth
      - k-NN: distance weighting, Manhattan metric, optimized neighbors
      - Random Forest: 200 trees, class balancing, feature sampling
      - Gradient Boosting: regularization, subsampling, lower learning rate
      - AdaBoost: SAMME algorithm, increased estimators
      - XGBoost: L1/L2 regularization, optimal depth and learning rate
    - **Feature descriptions**:
      - `age`: Patient age in years
      - `sex`: Gender (0=female, 1=male)
      - `cp`: Chest pain type (1-4)
      - `trestbps`: Resting blood pressure (mmHg)
      - `chol`: Serum cholesterol (mg/dl)
      - `fbs`: Fasting blood sugar >120 mg/dl (1=true, 0=false)
      - `restecg`: Resting ECG results (0-2)
      - `thalach`: Maximum heart rate achieved
      - `exang`: Exercise induced angina (1=yes, 0=no)
      - `oldpeak`: ST depression induced by exercise
      - `slope`: Slope of peak exercise ST segment (1-3)
      - `ca`: Number of major vessels colored by fluoroscopy (0-3)
      - `thal`: Thalassemia (3=normal, 6=fixed defect, 7=reversible defect)
    """)

    ui_template.create_footer()

    # Bind events
    demo.load(fn=init_page, inputs=[train_split], outputs=[status_md, preview, metrics_df])

    # Retrain models when split changes or button is clicked
    retrain_btn.click(
        fn=init_page,
        inputs=[train_split],
        outputs=[status_md, preview, metrics_df]
    )

    # Auto-fill when example is selected
    ex_selector.change(
        fn=fill_example,
        inputs=[ex_selector],
        outputs=[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    )

    predict_btn.click(
        fn=run_predict,
        inputs=[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal],
        outputs=[bar_out, sub_md, table_out]
    )

if __name__ == "__main__":
    demo.launch(allowed_paths=["static/heart_sentinel.png", "static/heart_sentinel.png", "static"])
