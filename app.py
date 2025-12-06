import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import gradio as gr
import plotly.graph_objects as go
import pandas as pd
import ui_template

from src.heart_disease_core import (
    CLEVELAND_FEATURES_ORDER,
    load_cleveland_dataframe,
    fit_all_models,
    predict_all,
    example_patient,
    get_example_labels
)


# Configuration
class AppConfig:
    """Centralized application configuration."""
    DATA_PATH = Path("data/cleveland.csv")
    DEFAULT_TRAIN_SPLIT = 80
    MIN_TRAIN_SPLIT = 60
    MAX_TRAIN_SPLIT = 90

    # Colors
    COLOR_DISEASE = "#C4314B"
    COLOR_NO_DISEASE = "#2E7D32"
    COLOR_NEUTRAL = "#666"

    # Visualization
    PLOT_HEIGHT = 420
    ENSEMBLE_BORDER_WIDTH = 2.5


class AppState:
    """Application state management."""
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.models: Optional[Dict] = None
        self.metrics: Optional[pd.DataFrame] = None

    def is_initialized(self) -> bool:
        """Check if models are initialized."""
        return all([self.df is not None, self.models is not None, self.metrics is not None])

    def reset(self):
        """Reset application state."""
        self.df = None
        self.models = None
        self.metrics = None


# Global state
state = AppState()


def configure_ui():
    """Configure UI template with app-specific settings."""
    ui_template.configure(
        project_name="Heart Sentinel",
        year="2025",
        about="AI that protects your heart before problems arise.",
        description="Heart Sentinel is an intelligent health-monitoring and early-warning system designed to analyze cardiovascular signals, predict health risks, and provide personalized lifestyle guidance. The system integrates machine learning, risk prediction, chatbot health coaching, and is designed to extend into real-time wearable data for continuous health monitoring.",
        meta_items=[
            ("Dataset", "Cleveland Heart Disease"),
            ("Models", "Decision Tree, k-NN, Naive Bayes, Random Forest, AdaBoost, Gradient Boosting, XGBoost"),
        ],
    )


def force_light_theme_js() -> str:
    """JavaScript to force light theme."""
    return """
    () => {
        const params = new URLSearchParams(window.location.search);
        if (!params.has('__theme')) {
            params.set('__theme', 'light');
            window.location.search = params.toString();
        }
    }
    """


def init_page(train_split: float) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    """
    Initialize the application by loading dataset and training models.

    Args:
        train_split: Percentage of data to use for training

    Returns:
        Tuple of (status_message, data_preview, metrics_dataframe)
    """
    if not AppConfig.DATA_PATH.exists():
        error_msg = f"❌ Dataset not found at '{AppConfig.DATA_PATH}'. Please place Cleveland CSV there."
        return error_msg, pd.DataFrame(), pd.DataFrame()

    try:
        # Load data
        df = load_cleveland_dataframe(file_path=str(AppConfig.DATA_PATH))

        # Train models
        test_size = (100 - train_split) / 100
        models, metrics = fit_all_models(df, test_size=test_size)

        # Update state
        state.df = df
        state.models = models
        state.metrics = metrics

        # Prepare outputs
        data_preview = df.head(8)
        success_msg = (
            f"✅ Cleveland dataset loaded from `{AppConfig.DATA_PATH}` and models trained "
            f"({int(train_split)}/{100-int(train_split)} split)."
        )

        return success_msg, data_preview, metrics

    except Exception as e:
        error_msg = f"❌ Error during initialization: {str(e)}"
        return error_msg, pd.DataFrame(), pd.DataFrame()


def fill_example(example_selector: str) -> List:
    """
    Fill input fields with example patient data.

    Args:
        example_selector: Selected example text (e.g., "Example 1 (No Heart Disease)")

    Returns:
        List of values matching CLEVELAND_FEATURES_ORDER
    """
    import re

    # Extract example index from selector text
    match = re.search(r'Example (\d+)', example_selector)
    idx = int(match.group(1)) - 1 if match else 0

    example_data = example_patient(idx)
    return [example_data[col] for col in CLEVELAND_FEATURES_ORDER]


def create_prediction_visualization(results: Dict) -> go.Figure:
    """
    Create bar chart visualization of model predictions.

    Args:
        results: Dictionary of model predictions

    Returns:
        Plotly figure object
    """
    model_names = list(results.keys())
    confidences = []
    prediction_labels = []
    bar_colors = []
    border_colors = []
    border_widths = []

    for name in model_names:
        result = results[name]

        # Determine confidence and styling
        if result["label"] == 1:
            confidences.append(result["prob_1"])
            prediction_labels.append("🫀 Heart Disease")
            bar_colors.append(AppConfig.COLOR_DISEASE)
        else:
            confidences.append(result["prob_0"])
            prediction_labels.append("✅ No Heart Disease")
            bar_colors.append(AppConfig.COLOR_NO_DISEASE)

        # Default border styling
        border_colors.append("rgba(0,0,0,0.15)")
        border_widths.append(1.0)

    # Highlight ensemble model
    if "Ensemble (Soft Voting)" in model_names:
        idx = model_names.index("Ensemble (Soft Voting)")
        border_colors[idx] = "#000000"
        border_widths[idx] = AppConfig.ENSEMBLE_BORDER_WIDTH

    # Create figure
    fig = go.Figure()
    fig.add_bar(
        x=model_names,
        y=confidences,
        text=prediction_labels,
        textposition="auto",
        marker=dict(
            color=bar_colors,
            line=dict(color=border_colors, width=border_widths)
        )
    )

    fig.update_layout(
        title="Model Predictions",
        yaxis_title="Prediction Confidence",
        xaxis_title="Model",
        yaxis=dict(range=[0, 1]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="black", size=12),
        height=AppConfig.PLOT_HEIGHT,
        margin=dict(l=30, r=20, t=60, b=40)
    )

    return fig


def format_prediction_card(name: str, result: Dict, is_ensemble: bool = False) -> str:
    """
    Format a single prediction result as HTML card.

    Args:
        name: Model name
        result: Prediction result dictionary
        is_ensemble: Whether this is the ensemble model

    Returns:
        HTML string for the prediction card
    """
    # Determine styling based on prediction
    is_disease = result["label"] == 1
    color = AppConfig.COLOR_DISEASE if is_disease else AppConfig.COLOR_NO_DISEASE
    prediction_text = "🫀 **Heart Disease Detected**" if is_disease else "✅ **No Heart Disease**"
    confidence = result["prob_1"] if is_disease else result["prob_0"]

    # Different styling for ensemble
    if is_ensemble:
        return f"""
<div style="border: 3px solid {color}; border-radius: 10px; padding: 20px; margin: 15px 0; background: white;">
    <h3 style="margin: 0 0 15px 0; color: {color};">🎯 Ensemble Prediction (Final Result)</h3>
    <p style="margin: 10px 0; font-size: 18px; color: black;"><strong>{prediction_text}</strong></p>
    <p style="margin: 5px 0; font-size: 16px; color: black;"><strong>Confidence:</strong> {confidence:.1%}</p>
</div>
"""
    else:
        return f"""
<div style="border: 2px solid {color}; border-radius: 8px; padding: 15px; margin: 10px 0; background: white;">
    <h4 style="margin: 0 0 10px 0; color: {color};">{name}</h4>
    <p style="margin: 5px 0; font-size: 16px; color: black;"><strong>Prediction:</strong> {prediction_text}</p>
    <p style="margin: 5px 0; font-size: 14px; color: black;"><strong>Confidence:</strong> {confidence:.1%}</p>
    <p style="margin: 5px 0; font-size: 12px; color: {AppConfig.COLOR_NEUTRAL};">
        P(No disease): {result['prob_0']:.3f} | P(Heart disease): {result['prob_1']:.3f}
    </p>
</div>
"""


def create_results_table(results: Dict) -> pd.DataFrame:
    """
    Create a DataFrame from prediction results.

    Args:
        results: Dictionary of model predictions

    Returns:
        DataFrame with formatted results
    """
    rows = []
    for name, result in results.items():
        is_disease = result["label"] == 1
        confidence = result["prob_1"] if is_disease else result["prob_0"]

        rows.append({
            "Model": name,
            "Prediction": "Heart Disease" if is_disease else "No Heart Disease",
            "Confidence": f"{confidence:.1%}",
            "P(No disease)": round(result["prob_0"], 3),
            "P(Heart disease)": round(result["prob_1"], 3),
        })

    return pd.DataFrame(rows)


def run_prediction(*input_values) -> Tuple[Optional[go.Figure], str, pd.DataFrame]:
    """
    Run prediction on all models with given input values.

    Args:
        *input_values: Variable length input values matching CLEVELAND_FEATURES_ORDER

    Returns:
        Tuple of (visualization_figure, markdown_results, results_table)
    """
    if not state.is_initialized():
        error_msg = "❌ Models not initialized. Please reload the app or retrain models."
        return None, error_msg, pd.DataFrame()

    try:
        # Prepare input dictionary
        input_dict = {col: input_values[i] for i, col in enumerate(CLEVELAND_FEATURES_ORDER)}

        # Get predictions from all models
        results = predict_all(state.models, input_dict)

        # Format ensemble result (show first)
        ensemble_result = results.get("Ensemble (Soft Voting)")
        if ensemble_result:
            ensemble_html = format_prediction_card(
                "Ensemble (Soft Voting)",
                ensemble_result,
                is_ensemble=True
            )
        else:
            ensemble_html = ""

        # Format individual model results
        model_htmls = [
            format_prediction_card(name, result)
            for name, result in results.items()
            if name != "Ensemble (Soft Voting)"
        ]

        # Combine all results
        all_results_html = ensemble_html + "\n".join(model_htmls)

        # Create visualization and table
        fig = create_prediction_visualization(results)
        table_df = create_results_table(results)

        return fig, all_results_html, table_df

    except Exception as e:
        error_msg = f"❌ Prediction error: {str(e)}"
        return None, error_msg, pd.DataFrame()


def get_example_choices() -> Tuple[List[str], str]:
    """
    Get example patient choices with their labels.

    Returns:
        Tuple of (choices_list, default_choice)
    """
    try:
        labels = get_example_labels()
        choices = []

        # Only use first two examples
        for i in range(min(2, len(labels))):
            label_text = "No Heart Disease" if labels[i] == 0 else "Heart Disease"
            choices.append(f"Example {i+1} ({label_text})")

        default_choice = choices[0] if choices else "Example 1 (No Heart Disease)"

    except Exception:
        choices = ["Example 1 (No Heart Disease)", "Example 2 (Heart Disease)"]
        default_choice = "Example 1 (No Heart Disease)"

    return choices, default_choice


def create_feature_inputs() -> Dict[str, gr.components.Component]:
    """
    Create all feature input components.

    Returns:
        Dictionary mapping feature names to Gradio components
    """
    inputs = {}

    with gr.Row():
        inputs['age'] = gr.Number(label="age (years)", value=58)
        inputs['sex'] = gr.Dropdown(label="sex (0=female, 1=male)", choices=[0, 1], value=1)
        inputs['cp'] = gr.Dropdown(label="cp (chest pain type 1..4)", choices=[1, 2, 3, 4], value=2)
        inputs['trestbps'] = gr.Number(label="trestbps (resting BP mmHg)", value=130)

    with gr.Row():
        inputs['chol'] = gr.Number(label="chol (serum cholesterol mg/dl)", value=250)
        inputs['fbs'] = gr.Dropdown(label="fbs (>120 mg/dl? 1/0)", choices=[0, 1], value=0)
        inputs['restecg'] = gr.Dropdown(label="restecg (0..2)", choices=[0, 1, 2], value=1)
        inputs['thalach'] = gr.Number(label="thalach (max heart rate)", value=150)

    with gr.Row():
        inputs['exang'] = gr.Dropdown(label="exang (exercise angina 1/0)", choices=[0, 1], value=0)
        inputs['oldpeak'] = gr.Number(label="oldpeak (ST depression)", value=1.0)
        inputs['slope'] = gr.Dropdown(label="slope (1..3)", choices=[1, 2, 3], value=1)
        inputs['ca'] = gr.Dropdown(label="ca (major vessels 0..3)", choices=[0, 1, 2, 3], value=0)

    inputs['thal'] = gr.Dropdown(
        label="thal (3=normal, 6=fixed, 7=reversible)",
        choices=[3, 6, 7],
        value=3
    )

    return inputs


def create_documentation_section():
    """Create the documentation/notes section."""
    gr.Markdown("""
    ## 📋 **Notes**

    - **Models are trained at launch** on `data/cleveland.csv` with customizable train/validation split (default 80/20).
    - **Target is binarized automatically** (0 = no disease, >0 = disease).
    - **Retrain functionality**: Adjust the split ratio and click "🔄 Retrain Models" to see how data size affects performance.
    - **Seven optimized models are compared**: Decision Tree, k-NN, Naive Bayes, Random Forest, AdaBoost, Gradient Boosting, and XGBoost.
    - **Hyperparameters are optimized** for heart disease prediction tasks using best practices.
    - **Ensemble uses weighted soft voting** with optimized weights based on model performance.
    - **Best performing model** on test set is highlighted with 🏆 in the validation metrics table.

    ### **Optimization highlights**:
    - **Decision Tree**: entropy criterion, balanced classes, optimal depth
    - **k-NN**: distance weighting, Manhattan metric, optimized neighbors
    - **Random Forest**: 200 trees, class balancing, feature sampling
    - **Gradient Boosting**: regularization, subsampling, lower learning rate
    - **AdaBoost**: SAMME algorithm, increased estimators
    - **XGBoost**: L1/L2 regularization, optimal depth and learning rate

    ### **Feature descriptions**:
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

def create_references_section():
    """Create the references/sources section."""
    gr.Markdown("""
    ## 🔗 **References & Sources**

    ### **Dataset**
    - **Cleveland Heart Disease Dataset**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease)

    ### **Demo**
    - **Heart Disease Diagnosis Project**: [AIO2025M03_HEART_DISEASE_PREDICTION](https://huggingface.co/spaces/elizabethmyn/AIO2025M03_HEART_DISEASE_PREDICTION)
    ---
    **Disclaimer**: This application is for educational and research purposes only. All clinical decisions should be made by qualified healthcare professionals.
    """
)
def build_interface() -> gr.Blocks:
    """
    Build the complete Gradio interface.

    Returns:
        Gradio Blocks interface
    """
    configure_ui()

    with gr.Blocks(
        theme="gstaff/sketch",
        css=ui_template.get_custom_css(),
        fill_width=True,
        js=force_light_theme_js()
    ) as demo:

        # Header
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
            # LEFT COLUMN: Data & Inputs
            with gr.Column(scale=45):
                # Dataset section
                with gr.Accordion("📁 Dataset & Model Status", open=True):
                    with gr.Row():
                        train_split = gr.Slider(
                            minimum=AppConfig.MIN_TRAIN_SPLIT,
                            maximum=AppConfig.MAX_TRAIN_SPLIT,
                            value=AppConfig.DEFAULT_TRAIN_SPLIT,
                            step=5,
                            label="Training Split (%)",
                            info="Percentage of data used for training (remaining for validation)"
                        )
                        retrain_btn = gr.Button("🔄 Retrain Models", variant="secondary")

                    status_md = gr.Markdown("Loading dataset and training models...")
                    preview = gr.DataFrame(label="Cleveland Preview (first rows)", interactive=False)
                    metrics_df = gr.DataFrame(
                        label="Model Performance Comparison (Validation Set Results)",
                        interactive=False
                    )

                # Input features section
                with gr.Accordion("✍️ Enter Patient Features", open=True):
                    feature_inputs = create_feature_inputs()

                    with gr.Row():
                        example_choices, default_example = get_example_choices()
                        ex_selector = gr.Dropdown(
                            label="Select Example Patient",
                            choices=example_choices,
                            value=default_example
                        )
                        predict_btn = gr.Button("🔍 Predict", variant="primary")

            # RIGHT COLUMN: Results
            with gr.Column(scale=55):
                gr.Markdown("### 📈 Model Predictions")
                bar_out = gr.Plot(label="Model Predictions Overview")
                results_md = gr.Markdown("**Individual Model Results**")
                table_out = gr.DataFrame(label="All Model Predictions", interactive=False)

        # Documentation
        create_documentation_section()

        # Reference
        create_references_section()

        # Footer
        ui_template.create_footer()

        # Event bindings
        demo.load(
            fn=init_page,
            inputs=[train_split],
            outputs=[status_md, preview, metrics_df]
        )

        retrain_btn.click(
            fn=init_page,
            inputs=[train_split],
            outputs=[status_md, preview, metrics_df]
        )

        # Get input components in correct order
        input_components = [feature_inputs[col] for col in CLEVELAND_FEATURES_ORDER]

        ex_selector.change(
            fn=fill_example,
            inputs=[ex_selector],
            outputs=input_components
        )

        predict_btn.click(
            fn=run_prediction,
            inputs=input_components,
            outputs=[bar_out, results_md, table_out]
        )

    return demo


def main():
    """Main entry point."""
    demo = build_interface()
    demo.launch(
        allowed_paths=["static/heart_sentinel.png", "static"]
    )


if __name__ == "__main__":
    main()