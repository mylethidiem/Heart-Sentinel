import os, base64
import gradio as gr

# Theming (can be overridden by the host app)
PRIMARY_COLOR = "#0F6CBD"   # medical calm blue
ACCENT_COLOR = "#C4314B"    # medical alert red
SUCCESS_COLOR = "#2E7D32"   # positive/ok
BG1 = "#F0F7FF"
BG2 = "#E8F0FA"
BG3 = "#DDE7F8"
FONT_FAMILY = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', 'Liberation Sans', sans-serif"

# App metadata (overridable)
PROJECT_NAME = "Demo Project"
YEAR = "2025"
PROJECT_DESCRIPTION = ""
META_INFO = []  # list of (label, value)

def set_colors(primary: str = None, accent: str = None, bg1: str = None, bg2: str = None, bg3: str = None):
    """Allow host app to set theme colors dynamically."""
    global PRIMARY_COLOR, ACCENT_COLOR, BG1, BG2, BG3, custom_css
    if primary:
        PRIMARY_COLOR = primary
    if accent:
        ACCENT_COLOR = accent
    if bg1:
        BG1 = bg1
    if bg2:
        BG2 = bg2
    if bg3:
        BG3 = bg3
    # Rebuild CSS with new colors
    custom_css = _build_custom_css()

def set_font(font_family: str):
    """Allow host app to set a custom font stack (e.g., 'Inter', system fallbacks)."""
    global FONT_FAMILY, custom_css
    if font_family and isinstance(font_family, str):
        FONT_FAMILY = font_family
        custom_css = _build_custom_css()

def set_meta(project_name: str = None, year: str = None, description: str = None, meta_items: list = None):
    """Set project metadata used across the header and info sections."""
    global PROJECT_NAME, YEAR, PROJECT_DESCRIPTION, META_INFO
    if project_name is not None:
        PROJECT_NAME = project_name
    if year is not None:
        YEAR = year
    if description is not None:
        PROJECT_DESCRIPTION = description
    if meta_items is not None:
        META_INFO = meta_items

def configure(project_name: str = None, year: str = None, module: str = None, description: str = None,
              colors: dict = None, font_family: str = None, meta_items: list = None):
    """One-call configuration for meta, theme, and font."""
    if colors:
        set_colors(
            primary=colors.get("primary"),
            accent=colors.get("accent"),
            bg1=colors.get("bg1"),
            bg2=colors.get("bg2"),
            bg3=colors.get("bg3"),
        )
    if font_family:
        set_font(font_family)
    set_meta(project_name, year, description, meta_items)


def image_to_base64(image_path: str):
    # Construct the absolute path to the image
    current_dir = os.path.dirname(os.path.abspath(__file__))
    full_image_path = os.path.join(current_dir, image_path)
    with open(full_image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def create_header():
    with gr.Row():
        with gr.Column(scale=2):
            logo_base64 = image_to_base64("static/heart_sentinel.png")
            gr.HTML(
                f"""<img src="data:image/png;base64,{logo_base64}"
                        alt="Logo"
                        style="height:120px;width:auto;margin:0 auto;margin-bottom:16px; display:block;">"""
            )
        with gr.Column(scale=2):
            gr.HTML(f"""
<div style="display:flex;justify-content:flex-start;align-items:center;gap:30px;">
    <div>
        <h1 style="margin-bottom:0; color: {PRIMARY_COLOR}; font-size: 2.5em; font-weight: bold;"> {PROJECT_NAME} </h1>
        <h3 style="color: #888; font-style: italic"> </h3>
    </div>
</div>
""")

def create_footer():
    logo_base64_heart_sentinel = image_to_base64("static/heart_sentinel.png")
    footer_html = """
<style>
  .sticky-footer{position:fixed;bottom:0px;left:0;width:100%;background:#E8F5E8;
                 padding:10px;box-shadow:0 -2px 10px rgba(0,0,0,0.1);z-index:1000;}
  .content-wrap{padding-bottom:60px;}
</style>""" + f"""
<div class="sticky-footer">
  <div style="text-align:center;font-size:18px; color: #888">
    Created by
    <a href="my-personal-web" target="_blank" style="color:#465C88;text-decoration:none;font-weight:bold; display:inline-flex; align-items:center;"> Heart Sentinel
    <img src="data:image/png;base64,{logo_base64_heart_sentinel}" alt="Logo" style="height:20px; width:auto;">
    </a> from <a href="https://aivietnam.edu.vn/" target="_blank" style="color:#355724;text-decoration:none;font-weight:bold">AI VIET NAM</a>
  </div>
</div>
"""
    return gr.HTML(footer_html)

def _build_custom_css() -> str:
    return f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.gradio-container {{
    min-height: 100vh !important;
    width: 100vw !important;
    margin: 0 !important;
    padding: 0px !important;
    background: linear-gradient(135deg, {BG1} 0%, {BG2} 50%, {BG3} 100%);
    background-size: 600% 600%;
    animation: gradientBG 7s ease infinite;
}}

/* Global font setup */
body, .gradio-container, .gr-block, .gr-markdown, .gr-button, .gr-input,
.gr-dropdown, .gr-number, .gr-plot, .gr-dataframe, .gr-accordion, .gr-form,
.gr-textbox, .gr-html, table, th, td, label, h1, h2, h3, h4, h5, h6, p, span, div {{
    font-family: {FONT_FAMILY} !important;
}}

@keyframes gradientBG {{
    0% {{background-position: 0% 50%;}}
    50% {{background-position: 100% 50%;}}
    100% {{background-position: 0% 50%;}}
}}

/* Minimize spacing and padding */
.content-wrap {{
    padding: 2px !important;
    margin: 0 !important;
}}

/* Reduce component spacing */
.gr-row {{
    gap: 5px !important;
    margin: 2px 0 !important;
}}

.gr-column {{
    gap: 4px !important;
    padding: 4px !important;
}}

/* Accordion optimization */
.gr-accordion {{
    margin: 4px 0 !important;
}}

.gr-accordion .gr-accordion-content {{
    padding: 2px !important;
}}

/* Form elements spacing */
.gr-form {{
    gap: 2px !important;
}}

/* Button styling */
.gr-button {{
    margin: 2px 0 !important;
}}

/* DataFrame optimization */
.gr-dataframe {{
    margin: 4px 0 !important;
}}

/* Remove horizontal scroll from data preview */
.gr-dataframe .wrap {{
    overflow-x: auto !important;
    max-width: 100% !important;
}}

/* Plot optimization */
.gr-plot {{
    margin: 4px 0 !important;
}}

/* Reduce markdown margins */
.gr-markdown {{
    margin: 2px 0 !important;
}}

/* Footer positioning */
.sticky-footer {{
    position: fixed;
    bottom: 0px;
    left: 0;
    width: 100%;
    background: {BG1};
    padding: 6px !important;
    box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
    z-index: 1000;
}}
"""

# Initialize CSS using defaults
custom_css = _build_custom_css()

def render_info_card(description: str = None, meta_items: list = None, icon: str = "🧠", title: str = "About this demo") -> str:
    desc = description if description is not None else PROJECT_DESCRIPTION
    items = meta_items if meta_items is not None else META_INFO
    meta_html = " · ".join([f"<span><strong>{k}</strong>: {v}</span>" for k, v in items]) if items else ""
    return f"""
    <div style="margin: 8px 0 8px 0;">
      <div style="background:#F5F9FF;border-left:6px solid {PRIMARY_COLOR};padding:14px 16px;border-radius:10px;box-shadow:0 1px 3px rgba(0,0,0,0.06);">
        <div style="display:flex;gap:14px;align-items:flex-start;">
          <div style="font-size:22px;">{icon}</div>
          <div>
            <div style="font-weight:700;color:{PRIMARY_COLOR};margin-bottom:4px;">{title}</div>
            <div style="color:#000;font-size:14px;line-height:1.5;">{desc}</div>
            <div style="margin-top:8px;color:#000;font-size:13px;">{meta_html}</div>
          </div>
        </div>
      </div>
    </div>
    """

def render_disclaimer(text: str, icon: str = "⚠️", title: str = "Educational Use Only") -> str:
    return f"""
    <div style=\"margin: 8px 0 6px 0;\">
      <div style=\"background:#FFF4F4;border-left:6px solid {ACCENT_COLOR};padding:12px 16px;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,0.06);\">
        <div style=\"display:flex;gap:10px;align-items:flex-start;color:#000;\">
          <span style=\"font-size:20px\">{icon}</span>
          <div>
            <div style=\"font-weight:700; margin-bottom:4px;\">{title}</div>
            <div style=\"font-size:14px; line-height:1.4;\">{text}</div>
          </div>
        </div>
      </div>
    </div>
    """
