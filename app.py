# app.py (Fixed: full UI + Azure Blob load + Azure SQL save + Langflow Integration + Fixed Chatbot + Telegram)
import io, os, tempfile, json, subprocess, logging
from datetime import datetime
from typing import Dict
import streamlit as st
from login import login_component
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from inference import ModelBundle
from preprocessing import aggregate_from_user_spec, preprocess_for_model, build_results_dataframe
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
# -----------------------------------------------------------
# Configuration
# -----------------------------------------------------------F
# Langflow Configuration
LANGFLOW_API_KEY = "sk-pBcQb1AjHSkKIwD8Fij2GxzadElnBpwGCdWHLOhTVIM"
LANGFLOW_URL = "https://eca3b9a7c18e.ngrok-free.app/api/v1/run/0eec7153-2678-4a98-aa16-f3b59ba544f3"

# Azure Blob Storage Configuration
AZURE_ACCOUNT_NAME = "model1234"
AZURE_ACCOUNT_KEY = "PLAF6u9KVdKvIDsgBlnzzNgUan2LfkzfG6S+7teE6q2OgEpX1UUlNqLA3lj9qAddwm3baEeW7lcU+AStdIUY2g=="
AZURE_CONTAINER_NAME = "model"
MODEL_BLOB_NAME = "fraud_detection_precision_model.pkl"
ENCODERS_BLOB_NAME = "fraud_detection_encoders.pkl"
METADATA_BLOB_NAME = "fraud_detection_metadata.pkl"

# Azure SQL Database Configuration
AZURE_SQL_SERVER = "fraud-123.database.windows.net"
AZURE_SQL_DATABASE = "fraudfeaturesdb"
AZURE_SQL_USERNAME = "giri"
AZURE_SQL_PASSWORD = "Fanatic@123"
AZURE_SQL_TABLE = "FraudFeatures"

# Telegram Configuration
TELEGRAM_LINK = "https://t.me/Madhuggbot"  # Replace with your actual Telegram link

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------
# Fixed Langflow API Functions
# -----------------------------------------------------------
def call_langflow_api_with_requests(message: str, session_id: str = None) -> Dict:
    """Enhanced API call using requests library with correct payload and timeout handling"""
    payload = {
        "input_value": message,
        "input_type": "chat",
        "output_type": "chat"
    }
    if session_id:
        payload["session_id"] = session_id
    
    headers = {
        "x-api-key": LANGFLOW_API_KEY,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "ngrok-skip-browser-warning": "true"  # required if using ngrok URL
    }
    
    try:
        response = requests.post(
            LANGFLOW_URL,
            json=payload,
            headers=headers,
            timeout=60
        )
        
        logger.info(f"API Response Status: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            logger.info(f"API Response Data: {json.dumps(response_data, indent=2)}")
            
            output_text = (
                response_data.get("output_text") or 
                response_data.get("response") or 
                response_data.get("result") or
                response_data.get("message") or
                ""
            )
            
            session_id = response_data.get("session_id", session_id)
            
            return {
                "success": True,
                "output_text": output_text,
                "response": response_data,
                "session_id": session_id
            }
        else:
            logger.error(f"API Error: {response.status_code} - {response.text}")
            return {"success": False, "error": f"API returned {response.status_code}: {response.text}", "response": None}
    
    except requests.exceptions.Timeout:
        logger.error("API request timed out")
        return {"success": False, "error": "API request timed out. Please try again.", "response": None}
    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to Langflow API")
        return {"success": False, "error": "Cannot connect to Langflow API. Please check if it's running.", "response": None}
    except Exception as e:
        logger.error(f"API call failed: {str(e)}")
        return {"success": False, "error": str(e), "response": None}


def call_langflow_api(message: str, session_id: str = None) -> Dict:
    """Main Langflow API call using requests (no curl fallback needed)"""
    return call_langflow_api_with_requests(message, session_id)


def extract_bot_message(response_data: dict) -> str:
    """Extract the bot message from Langflow response - FIXED VERSION"""
    try:
        # First check if it's already in the expected format with 'text' field
        if isinstance(response_data, dict) and "text" in response_data:
            text_content = response_data["text"]
            if isinstance(text_content, str) and text_content.strip():
                return text_content.strip()
        
        # Check for nested structure with 'data' -> 'text'
        if isinstance(response_data, dict) and "data" in response_data:
            data = response_data["data"]
            if isinstance(data, dict) and "text" in data:
                text_content = data["text"]
                if isinstance(text_content, str) and text_content.strip():
                    return text_content.strip()
        
        # Try standard Langflow response structure
        if "outputs" in response_data:
            outputs = response_data["outputs"]
            if isinstance(outputs, list) and len(outputs) > 0:
                first_output = outputs[0]
                if "outputs" in first_output:
                    nested_outputs = first_output["outputs"]
                    if isinstance(nested_outputs, list) and len(nested_outputs) > 0:
                        if "results" in nested_outputs[0]:
                            results = nested_outputs[0]["results"]
                            if "message" in results:
                                message_content = results["message"]
                                # Fix: Check if message_content is a dict with 'message' key
                                if isinstance(message_content, dict) and "message" in message_content:
                                    return str(message_content["message"]).strip()
                                elif isinstance(message_content, str):
                                    return message_content.strip()
        
        # Recursive search for any message-like field
        def find_text_content(obj):
            if isinstance(obj, dict):
                # Priority order for text fields
                for key in ["text", "message", "content", "response"]:
                    if key in obj and isinstance(obj[key], str) and obj[key].strip():
                        return obj[key].strip()
                
                # Recursively search in nested objects
                for value in obj.values():
                    result = find_text_content(value)
                    if result:
                        return result
            elif isinstance(obj, list):
                for item in obj:
                    result = find_text_content(item)
                    if result:
                        return result
            return None
        
        message = find_text_content(response_data)
        if message:
            return message
            
        # If all else fails, try to convert the response to a readable format
        if isinstance(response_data, dict):
            # Look for any string values that might contain the actual response
            for key, value in response_data.items():
                if isinstance(value, str) and len(value) > 10:  # Reasonable length for a response
                    return f"Response from {key}: {value}"
        
        return "I received your message but couldn't extract a proper response. Please try rephrasing your question."
        
    except Exception as e:
        logger.error(f"Error extracting message: {e}")
        return f"Sorry, there was an error processing the response: {str(e)}"

# -----------------------------------------------------------
# Page Config
# -----------------------------------------------------------
st.set_page_config(page_title="Fanatic", layout="wide")

# ------------------------- Authentication -------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
# Show login card and stop further rendering until authenticated
if not st.session_state.authenticated:
    login_component()
    st.stop()
# -----------------------------------------------------------------

# -----------------------------------------------------------
# CSS: Header + UI cards + polished look + Telegram popup + Fixed chatbot styling
# -----------------------------------------------------------
st.markdown("""
<style>
/* Layout */
body, .main {
    background-color: #fafafa;
    font-family: 'Inter', sans-serif;
}
.main .block-container {
    padding-top: 0rem;
    padding-bottom: 3rem;
    max-width: 1300px;
}

/* Header Card */
.header-card {
    background: linear-gradient(90deg, #0033A1, #00A1E0);
    padding: 1.2rem 2rem;
    border-radius: 14px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    margin-bottom: 2rem;
    text-align: left;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.header-card .logo {
    width: 56px;
    height: 56px;
    border-radius: 10px;
    background: rgba(255,255,255,0.12);
    display:flex;
    align-items:center;
    justify-content:center;
    font-weight:700;
    color: white;
    font-size: 1.25rem;
}
.header-card h1 { color: white; font-size: 1.5rem; font-weight: 800; margin: 0; }
.header-card p { color: #f0f0f0; font-size: 0.95rem; margin: 0; }

.ui-card:hover { transform: translateY(-2px); box-shadow: 0 10px 28px rgba(0,0,0,0.08); }
.ui-card h3 { font-size: 1.12rem; font-weight: 700; color: #0033A1; margin-bottom: 0.6rem; }
.ui-card p { color: #444; font-size: 0.95rem; margin-bottom: 0.6rem; }

/* Fixed Telegram Popup */
.telegram-popup {
    position: fixed;
    bottom: 20px;
    right: 20px;
    width: 60px;
    height: 60px;
    background: #0088cc;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 12px rgba(0, 136, 204, 0.4);
    cursor: pointer;
    z-index: 9999;
    transition: all 0.3s ease;
    animation: pulse 2s infinite;
}

.telegram-popup:hover {
    transform: scale(1.1);
    box-shadow: 0 6px 20px rgba(0, 136, 204, 0.6);
}

.telegram-popup svg {
    width: 30px;
    height: 30px;
    fill: white;
}

@keyframes pulse {
    0% { box-shadow: 0 4px 12px rgba(0, 136, 204, 0.4); }
    50% { box-shadow: 0 6px 20px rgba(0, 136, 204, 0.8); }
    100% { box-shadow: 0 4px 12px rgba(0, 136, 204, 0.4); }
}


.user-message {
    background: linear-gradient(135deg, #0033A1, #00A1E0);
    color: white;
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0 8px 25%;
    text-align: left;
    box-shadow: 0 2px 8px rgba(0,51,161,0.3);
    font-weight: 500;
}

.bot-message {
    background: linear-gradient(135deg, #f5f7fa, #ffffff);
    border: 1px solid #e2e8f0;
    color: #333;
    padding: 12px 16px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 25% 8px 0;
    text-align: left;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    font-weight: 400;
}

.chat-timestamp {
    font-size: 0.75rem;
    color: #888;
    text-align: center;
    margin: 4px 0;
}

/* Buttons */
.stButton>button {
    border-radius: 12px;
    padding: 0.56rem 1rem;
    border: 1px solid #0033A1;
    background-color: #00A1E0;
    color: white;
    font-weight: 600;
    transition: all .18s ease-in-out;
    box-shadow: 0 3px 8px rgba(0,0,0,0.06);
}
.stButton>button:hover { background-color: #0033A1; border-color: #00A1E0; transform: translateY(-2px); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0033A1, #00A1E0);
}
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] p {
    color: white;
}
section[data-testid="stSidebar"] .stMarkdown { font-size: 0.9rem; }

/* Tables */
.dataframe tbody tr:nth-child(even) { background-color: #f8faff; }
.dataframe th { background-color: #0000FF; color: white; font-weight: bold; }

/* Footer */
footer, .st-emotion-cache-12fmjuu {
    text-align: center; color: #888 !important; font-size: 0.85rem;
}

/* ── Fix: align chat text input and Send button in the same row ── */
.chat-input-container .stTextInput,
.chat-input-container .stTextInput>div,
.chat-input-container input[type="text"],
.chat-input-container textarea {
  height: 44px !important;
  min-height: 44px !important;
  padding: 10px 12px !important;
  box-sizing: border-box !important;
  border-radius: 10px !important;
  display: flex !important;
  align-items: center !important;
}

/* Style and align the Send button to match input height */
.chat-input-container .stButton > button {
  height: 44px !important;
  margin-top: 0 !important;        /* remove stray top offset */
  padding: 0 18px !important;
  border-radius: 10px !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
}

/* Ensure column wrappers don't add extra top spacing */
.chat-input-container [data-testid="stExpander"] { margin-top: 0 !important; }
.chat-input-container .stColumn, .chat-input-container .stColumns {
  display: flex !important;
  align-items: center !important;
}

/* Optional — tighten gap between input & button */
.chat-input-container .row-widget.stButton { margin-left: 6px !important; }


/* Quick buttons styling */
.quick-button {
    background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
    border: 1px solid #0ea5e9;
    border-radius: 8px;
    padding: 8px 12px;
    margin: 4px;
    font-size: 0.85rem;
    color: #0369a1;
    cursor: pointer;
    transition: all 0.2s ease;
}

.quick-button:hover {
    background: linear-gradient(135deg, #0ea5e9, #0284c7);
    color: white;
    transform: translateY(-1px);
}
</style>
""", unsafe_allow_html=True)

# Add Telegram popup HTML
st.markdown(f"""
<a href="{TELEGRAM_LINK}" target="_blank" rel="noopener noreferrer" style="text-decoration:none">
  <div class="telegram-popup" role="button" aria-label="Open Telegram Bot">
    <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
      <path d="M9.78 18.65l.28-4.23 7.68-6.92c.34-.31-.07-.46-.52-.19L7.74 13.3 3.64 12c-.88-.25-.89-.86.2-1.3l15.97-6.16c.73-.33 1.43.18 1.15 1.3l-2.72 12.81c-.19.91-.74 1.13-1.5.71L12.6 16.3l-1.99 1.93c-.23.23-.42.42-.83.42z"/>
    </svg>
  </div>
</a>
""", unsafe_allow_html=True)


# -----------------------------------------------------------
# Header Card (Top of Page)
# -----------------------------------------------------------
st.markdown("""
<div class="header-card">
    <div class="logo">F</div>
    <div>
        <h1>Fanatic</h1>
        <p>AI-powered Fraud Risk Insights for Medicare Claims</p>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# Sidebar: Model Artifacts & quick controls
# -----------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Model Artifacts")
    model_path = os.path.join("models","fraud_detection_precision_model.pkl")
    encoders_path = os.path.join("models","fraud_detection_encoders.pkl")
    metadata_path = os.path.join("models","fraud_detection_metadata.pkl")

    st.markdown("---")
    st.markdown("**🤖 Chatbot Status**")
    # Test Langflow connection
    if st.button("Test Langflow Connection"):
        with st.spinner("Testing connection..."):
            test_result = call_langflow_api("Hello, are you working?")
            if test_result["success"]:
                st.success("✅ Langflow connection successful!")
                # Show extracted message for verification
                extracted_msg = extract_bot_message(test_result["response"])
                st.info(f"Bot response: {extracted_msg}")
            else:
                st.error(f"❌ Connection failed: {test_result['error']}")

# -----------------------------------------------------------
# Model Bundle Loading (with Azure fallback)
# -----------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model_bundle_from_azure():
    """
    Attempt to load ModelBundle from Azure Blob using the helper in inference.ModelBundle.
    Returns ModelBundle or None if failed.
    """
    try:
        if hasattr(ModelBundle, "load_from_azure"):
            return ModelBundle.load_from_azure()
    except Exception:
        return None
    return None

bundle = get_model_bundle_from_azure()
if bundle:
    st.sidebar.write("Model loaded from Azure Blob Storage (via ModelBundle.load_from_azure).")
else:
    try:
        bundle = ModelBundle(model_path=model_path, encoders_path=encoders_path, metadata_path=metadata_path)
        st.sidebar.write("Model loaded from local artifacts (fallback).")
    except Exception as e:
        bundle = None
        st.sidebar.write(f"Model load failed: {e}")

# -----------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------
if "agg_df" not in st.session_state: st.session_state.agg_df = None
if "info" not in st.session_state: st.session_state.info = {}
if "results_df" not in st.session_state: st.session_state.results_df = None
if "file_names" not in st.session_state: st.session_state.file_names = []
if "manual_thresholds" not in st.session_state: st.session_state.manual_thresholds = None
if "uploaded_file_map" not in st.session_state: st.session_state.uploaded_file_map = {}
if "last_processed_file_names" not in st.session_state: st.session_state.last_processed_file_names = []
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "session_id" not in st.session_state: st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
if "pending_chat_message" not in st.session_state: st.session_state.pending_chat_message = ""


# Replace your existing chatbot blocks (both the analysis/chat and waiting/chat ones)
# with this function + small call. It re-uses your call_langflow_api and extract_bot_message.


def reset_workflow():
    st.session_state.agg_df = None
    st.session_state.info = {}
    st.session_state.results_df = None
    st.session_state.file_names = []
    st.session_state.manual_thresholds = None
    st.session_state.uploaded_file_map = {}
    st.session_state.last_processed_file_names = []

# -----------------------------------------------------------
# Helper functions (downloads, plots, etc.)
# -----------------------------------------------------------
def df_to_excel_with_highlight(df: pd.DataFrame) -> bytes:
    buffer = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
    with pd.ExcelWriter(buffer.name, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Predictions")
        workbook  = writer.book
        worksheet = writer.sheets["Predictions"]
        fraud_format = workbook.add_format({"bg_color": "#ffcccc"})
        manual_format = workbook.add_format({"bg_color": "#ffe6cc"})
        normal_format = workbook.add_format({"bg_color": "#ffffff"})
        for row_idx in range(len(df)):
            is_model_fraud = False
            is_manual_flag = False
            if "PotentialFraud" in df.columns:
                try: is_model_fraud = str(df.iloc[row_idx].get("PotentialFraud","")).strip()=="Yes"
                except: is_model_fraud = False
            if "ManualFlag" in df.columns:
                try: is_manual_flag = str(df.iloc[row_idx].get("ManualFlag","")).strip()=="Yes"
                except: is_manual_flag = False
            if is_model_fraud:
                worksheet.set_row(row_idx + 1, cell_format=fraud_format)
            elif is_manual_flag:
                worksheet.set_row(row_idx + 1, cell_format=manual_format)
            else:
                worksheet.set_row(row_idx + 1, cell_format=normal_format)
    with open(buffer.name, "rb") as f: return f.read()

def df_to_pdf(df: pd.DataFrame) -> bytes:
    buffer = tempfile.NamedTemporaryFile(delete=False)
    doc = SimpleDocTemplate(buffer.name, pagesize=landscape(A4))
    elements = []
    style = getSampleStyleSheet()
    elements.append(Paragraph("Fraud Prediction Results", style['Title']))
    data = [df.columns.tolist()] + df.values.tolist()
    page_width = landscape(A4)[0] - 40
    col_count = len(df.columns) or 1
    col_width = page_width / col_count
    col_widths = [col_width] * col_count
    table = Table(data, colWidths=col_widths)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0033A1")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,0), 8),
        ("FONTSIZE", (0,1), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,0), 6),
        ("BACKGROUND", (0,1), (-1,-1), colors.whitesmoke),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
    ]))
    if ("PotentialFraud" in df.columns) or ("ManualFlag" in df.columns):
        for i in range(len(df)):
            is_model_fraud = False
            is_manual_flag = False
            if "PotentialFraud" in df.columns:
                try: is_model_fraud = str(df.iloc[i].get("PotentialFraud","")).strip()=="Yes"
                except: is_model_fraud = False
            if "ManualFlag" in df.columns:
                try: is_manual_flag = str(df.iloc[i].get("ManualFlag","")).strip()=="Yes"
                except: is_manual_flag = False
            if is_model_fraud:
                table.setStyle(TableStyle([("BACKGROUND", (0, i+1), (-1, i+1), colors.HexColor("#ffcccc")), ("TEXTCOLOR", (0, i+1), (-1, i+1), colors.black)]))
            elif is_manual_flag:
                table.setStyle(TableStyle([("BACKGROUND", (0, i+1), (-1, i+1), colors.HexColor("#ffe6cc")), ("TEXTCOLOR", (0, i+1), (-1, i+1), colors.black)]))
    elements.append(table); doc.build(elements)
    with open(buffer.name, "rb") as f: return f.read()

def pick_col(df: pd.DataFrame, candidates):
    """
    Return the first column name in candidates that exists in df.columns.
    If none found, return None.
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None

def make_plots_modern(results_df: pd.DataFrame):
    plots = []

    base_layout = dict(
        template="plotly_white",
        title_font=dict(size=18, color="#0033A1", family="Arial Black"),
        font=dict(size=12, color="#333"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
        margin=dict(l=40, r=20, t=60, b=40)
    )

    # 1) Fraud Probability Distribution (histogram)
    if "Fraud_Probability" in results_df.columns:
        fig = px.histogram(
            results_df,
            x="Fraud_Probability",
            nbins=30,
            title="Fraud Probability Distribution"
        )
        fig.update_traces(
            marker=dict(color="#00A1E0", line=dict(width=0.5, color="darkslategray")),
            hovertemplate="<b>Fraud probability bin:</b> %{x:.2f}<br><b>Count:</b> %{y}<extra></extra>",
            opacity=0.9
        )
        fig.update_layout(**base_layout, xaxis_title="Fraud Probability", yaxis_title="Count")
        plots.append(("", fig))

    # 2) Risk Level Breakdown (bar)
    if "Risk_Level" in results_df.columns:
        risk_counts = results_df["Risk_Level"].value_counts().reset_index()
        risk_counts.columns = ["Risk_Level", "Count"]
        fig = px.bar(
            risk_counts,
            x="Risk_Level",
            y="Count",
            title="Risk Level Breakdown",
            text="Count"
        )
        fig.update_traces(
            textposition="outside",
            marker=dict(line=dict(width=0.5, color="darkslategray")),
            hovertemplate="<b>Risk Level:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>"
        )
        seq = px.colors.sequential.Blues if hasattr(px.colors, "sequential") else ["#cfe9ff", "#7ec8ff", "#1f77b4"]
        colors = [seq[int(i * (len(seq) - 1) / max(1, len(risk_counts) - 1))] for i in range(len(risk_counts))]
        fig.update_traces(marker_color=colors)
        fig.update_layout(**base_layout, showlegend=False)
        plots.append(("", fig))

    # 3) Top Providers by TotalClaims (horizontal bar with gradient)
    pid_col = pick_col(results_df, ["ProviderID", "Provider ID", "Provider", "PROVIDER_ID"])
    claims_col = pick_col(results_df, ["TotalClaims", "Total Claims", "SumInscClaimAmtReimbursed", "SumClaims"])
    if claims_col and pid_col and claims_col in results_df.columns and pid_col in results_df.columns:
        top = results_df.sort_values(claims_col, ascending=False).head(15).copy()
        fig = px.bar(
            top,
            y=pid_col,
            x=claims_col,
            orientation="h",
            title="Top Providers by Total Claims",
            text=claims_col
        )
        fig.update_traces(
            texttemplate="%{x:,}",
            textposition="outside",
            hovertemplate=f"<b>Provider:</b> %{{y}}<br><b>{claims_col}:</b> %{{x:,}}<extra></extra>",
            marker=dict(line=dict(width=0.5, color="darkslategray"))
        )
        try:
            fig.update_traces(marker=dict(color=top[claims_col].tolist(), colorscale="Blues", showscale=False, opacity=0.9))
        except Exception:
            fig.update_traces(marker=dict(color="#00A1E0", opacity=0.9))
        fig.update_layout(**base_layout, yaxis={"categoryorder": "total ascending"}, xaxis_title=claims_col)
        plots.append(("", fig))

    # 4) Fraud Probability vs Total Claims (scatter, optional animation by Risk_Level)
    if "Fraud_Probability" in results_df.columns and claims_col and claims_col in results_df.columns:
        df = results_df.copy()
        if "PotentialFraud" not in df.columns and "Predicted_Optimized" in df.columns:
            df["PotentialFraud"] = np.where(df["Predicted_Optimized"] == 1, "Yes", "No")
        if "PotentialFraud" not in df.columns:
            df["PotentialFraud"] = "No"

        color_map = {"Yes": "#d62728", "No": "#1f77b4"}
        anim_frame = "Risk_Level" if "Risk_Level" in df.columns else None

        # pick provider id column for hover
        pid_col_local = pick_col(df, ["ProviderID", "Provider ID", "Provider", "PROVIDER_ID"])
        fraud_prob_col = "Fraud_Probability"

        hover_cols = [c for c in [pid_col_local, claims_col, fraud_prob_col] if c is not None]

        fig = px.scatter(
            df,
            x=claims_col,
            y=fraud_prob_col,
            color="PotentialFraud",
            color_discrete_map=color_map,
            size=claims_col if df[claims_col].nunique() > 1 else None,
            animation_frame=anim_frame,
            hover_data=hover_cols,
            title="Fraud Probability vs Total Claims"
        )

        # stable hover via customdata (px puts hover_data into customdata in the same order)
        # customdata[0] => pid_col_local (if present), [1] => claims_col, [2] => fraud_prob_col
        customdata_index = {}
        for i, c in enumerate(hover_cols):
            customdata_index[c] = i

        template_parts = []
        if pid_col_local: template_parts.append("<b>Provider:</b> %{customdata[" + str(customdata_index[pid_col_local]) + "]}") 
        template_parts.append("<b>Claims:</b> %{customdata[" + str(customdata_index[claims_col]) + "]:,}")
        template_parts.append("<b>Fraud Probability:</b> %{customdata[" + str(customdata_index[fraud_prob_col]) + "]:.2f}<extra></extra>")
        hovertemplate = "<br>".join(template_parts)

        fig.update_traces(
            marker=dict(opacity=0.75, line=dict(width=0.4, color="darkslategray")),
            hovertemplate=hovertemplate
        )

        # annotation for high-risk zone (if data present)
        try:
            max_claims = df[claims_col].max()
            fig.add_annotation(
                x=max_claims * 0.75 if pd.notna(max_claims) else 0,
                y=0.9,
                text="⚠ High Risk Zone",
                showarrow=False,
                font=dict(size=12, color="red"),
                bgcolor="rgba(255,255,255,0.8)"
            )
        except Exception:
            pass

        # Build visible slider (one step per frame) and Play/Pause buttons that work across Plotly versions
        steps = []
        frame_names = [f.name for f in fig.frames] if hasattr(fig, "frames") and fig.frames is not None else []
        for i, fname in enumerate(frame_names):
            step = {
                "args": [[fname], {"frame": {"duration": 700, "redraw": False}, "mode": "immediate", "transition": {"duration": 300}}],
                "label": str(fname),
                "method": "animate"
            }
            steps.append(step)

        slider = {
            "active": 0,
            "currentvalue": {"prefix": "Frame: ", "visible": True, "xanchor": "center"},
            "pad": {"t": 50},
            "len": 0.9,
            "x": 0.05,
            "y": -0.12,
            "steps": steps
        }

        updatemenus = [{
            "type": "buttons",
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 700, "redraw": False}, "fromcurrent": True, "transition": {"duration": 300, "easing": "quadratic-in-out"}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 60},
            "showactive": False,
            "x": 0.05,
            "y": -0.05
        }]

        fig.update_layout(
            **base_layout,
            xaxis_title=claims_col,
            yaxis_title="Fraud Probability",
            legend_title="Potential Fraud",
            sliders=[slider] if steps else [],
            updatemenus=updatemenus
        )

        if steps:
            fig.layout.sliders[0].active = 0

        plots.append(("", fig))

    return plots

# -----------------------------------------------------------
# Upload card — single uploader placed inside a two-column layout
# -----------------------------------------------------------
with st.container():
    st.markdown('', unsafe_allow_html=True)
    cols = st.columns([3, 2])  # left: text, right: uploader + start button

    with st.container():
        st.markdown('<div class="ui-card">', unsafe_allow_html=True)
        st.markdown("<h3>📂 Upload Claim Datasets</h3>", unsafe_allow_html=True)
        st.markdown("<p>Upload the four datasets (Beneficiary, Inpatient, Outpatient, Test/Provider). Multiple selection supported (.csv/.xlsx/.xls).</p>", unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Drag & drop or click to browse",
            type=["csv"],
            accept_multiple_files=True,
            help="Upload Beneficiary, Inpatient, Outpatient, Test (Provider) files."
        )

        if uploaded_files:
            st.markdown("**✅ Files ready:**")
            for f in uploaded_files:
                st.write("📄", f.name)

        st.markdown('</div>', unsafe_allow_html=True)


        # Start Analysis button (processing runs only when this button is clicked)
        start_clicked = st.button("Start Analysis", key="start_analysis")

    st.markdown('</div>', unsafe_allow_html=True)

# Reset button
col_reset = st.columns([1, 7, 1])
with col_reset[0]:
    if st.button("Reset"):
        reset_workflow()
with col_reset[2]:
    st.write("")

# -----------------------------------------------------------
# When Start Analysis is clicked: build file_map and run pipeline
# -----------------------------------------------------------
if start_clicked:
    if not uploaded_files:
        st.warning("Please upload files before starting analysis.")
    else:
        # Build file_map from the uploaded_files on this run
        file_map = {}
        for uf in uploaded_files:
            try:
                df = pd.read_csv(uf) if uf.name.lower().endswith(".csv") else pd.read_excel(uf)
            except Exception:
                uf.seek(0)
                df = pd.read_csv(uf, error_bad_lines=False)
            file_map[uf.name] = df

        uploaded_names = sorted(list(file_map.keys()))
        st.session_state.file_names = uploaded_names
        st.session_state.uploaded_file_map = file_map


        # Run the processing pipeline inside a spinner
        with st.spinner("Aggregating, preprocessing and running predictions..."):
            try:
                agg_df, info = aggregate_from_user_spec(file_map)
                st.session_state.agg_df = agg_df
                st.session_state.info = info
            except Exception as e:
                st.error(f"Aggregation failed: {e}")
                st.stop()

            try:
                X, available_features, threshold = preprocess_for_model(st.session_state.agg_df, encoders=bundle.encoders, metadata=bundle.metadata)
            except Exception as e:
                st.error(f"Preprocessing failed: {e}")
                st.stop()

            agg_df_local = st.session_state.agg_df.copy()
            manual_thresholds = st.session_state.manual_thresholds
            flagged_df = pd.DataFrame(); remaining_df = agg_df_local.copy()

            if manual_thresholds:
                cond = (
                    (agg_df_local.get("TotalClaims", 0) >= manual_thresholds.get("TotalClaims", 0)) |
                    (agg_df_local.get("SumInscClaimAmtReimbursed", 0) >= manual_thresholds.get("SumInscClaimAmtReimbursed", 0)) |
                    (agg_df_local.get("SumDeductibleAmtPaid", 0) >= manual_thresholds.get("SumDeductibleAmtPaid", 0))
                )
                flagged_df = agg_df_local[cond].copy()
                remaining_df = agg_df_local[~cond].copy()
                if not flagged_df.empty:
                    flagged_df["ManualFlag"] = "Yes"
                    flagged_df["FraudFlag"] = "⚠️ MANUAL_FLAG"

            try:
                X_remain, available_features_remain, threshold = preprocess_for_model(remaining_df, encoders=bundle.encoders, metadata=bundle.metadata)
                proba, std, opt = bundle.predict_dataframe(X_remain, threshold)
                results_df = build_results_dataframe(remaining_df, available_features_remain, proba, std, opt, threshold)
            except Exception as e:
                st.warning(f"Model prediction failed or model missing. Showing aggregated provider features only. ({e})")
                results_df = remaining_df.copy()
                results_df["Fraud_Probability"] = np.nan
                results_df["Predicted_Optimized"] = np.nan
                results_df["Risk_Level"] = np.nan
                results_df["Confidence"] = np.nan

            # Ensure ManualFlag exists and has default "No"
            if "ManualFlag" not in results_df.columns:
                results_df["ManualFlag"] = "No"
            else:
                results_df["ManualFlag"] = np.where(results_df["ManualFlag"].isnull(), "No", results_df["ManualFlag"])

            # ---- FORCE Predicted_Optimized = 1 when Risk_Level is High/Critical ----
            if "Risk_Level" in results_df.columns:
                try:
                    # some Risk_Level values may be categorical; compare string form
                    mask_high = results_df["Risk_Level"].astype(str).isin(["High", "Critical"])
                    if "Predicted_Optimized" in results_df.columns:
                        results_df.loc[mask_high, "Predicted_Optimized"] = 1
                    else:
                        results_df.loc[mask_high, "Predicted_Optimized"] = 1
                except Exception:
                    pass

            # Create PotentialFraud and FraudFlag from Predicted_Optimized (after override)
            if "Predicted_Optimized" in results_df.columns:
                results_df["PotentialFraud"] = np.where(results_df["Predicted_Optimized"] == 1, "Yes", "No")
                results_df["FraudFlag"] = np.where(results_df["PotentialFraud"] == "Yes", "⚠️ FRAUD", "")

            # If flagged manually at top, make sure schema alignment & mark as Flagged
            if not flagged_df.empty:
                for c in results_df.columns:
                    if c not in flagged_df.columns:
                        flagged_df[c] = np.nan
                flagged_df["Fraud_Probability"] = np.nan
                flagged_df["Predicted_Optimized"] = np.nan
                flagged_df["Risk_Level"] = "Flagged"
                flagged_df["Confidence"] = "Flagged"
                flagged_df["ManualFlag"] = "Yes"
                flagged_df = flagged_df[results_df.columns]
                results_df = pd.concat([results_df, flagged_df], ignore_index=True)

            st.session_state.results_df = results_df
            st.session_state.last_processed_file_names = uploaded_names

            # Save to Azure SQL (append)
            try:
                import DBconnect as dbc
                if hasattr(dbc, "save_predictions_to_database"):
                    saved = dbc.save_predictions_to_database(results_df.copy(), mode='append')
                    if saved:
                        st.success("Predictions saved to Azure SQL Database (table: FraudFeatures).")
                    else:
                        st.warning("Saving predictions to Azure SQL Database returned False. Check logs and DB settings.")
                else:
                    st.warning("DBconnect.save_predictions_to_database not found. Skipping DB save.")
            except Exception as e:
                st.error(f"Error while saving predictions to DB: {e}")

        st.success("Processing complete — results ready.")

# -----------------------------------------------------------
# Show results
# -----------------------------------------------------------
if st.session_state.results_df is not None:
    results_df = st.session_state.results_df.copy()
    st.markdown('<div class="ui-card"><h3>📊 Visual Insights</h3><p>Quick overview of model outputs and risk distribution.</p>', unsafe_allow_html=True)
    try:
        modern_plots = make_plots_modern(results_df)
        for i in range(0, len(modern_plots), 2):
            row = st.columns(2)
            for j, (title, fig) in enumerate(modern_plots[i:i+2]):
                with row[j]:
                    st.markdown(f"**{title}**")
                    st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Unable to render plots: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------------------
    # Predictions & Provider Summary
    # -----------------------------------------------------------
    st.markdown('<div class="ui-card"><h3>⚡ Predictions & Provider Summary</h3><p>Key provider-level features and model prediction flags.</p>', unsafe_allow_html=True)

    # Start from results_df but drop unwanted columns if present
    display_df = results_df.copy()

    # Columns user asked to remove (include common name variants)
    cols_to_remove = [
        "Avg Age", "AvgGender", "Avg_Gender", "Avg Gender",
        "Avg ChronicCond_Diabetes", "AvgChronicCond_Diabetes", "Avg ChronicCond_Diabetes",
        "Fraud Flag", "Manual Flag",  # spaced variants
        "FraudFlag", "ManualFlag"     # no-space variants
    ]
    cols_present_to_drop = [c for c in cols_to_remove if c in display_df.columns]
    if cols_present_to_drop:
        display_df.drop(columns=cols_present_to_drop, inplace=True)

    # Map of existing column-names -> desired UI display names (ordered)
    col_map = {
        "ProviderID": "Provider ID",
        "Provider ID": "Provider ID",
        "Provider": "Provider ID",
        "TotalClaims": "Total Claims",
        "Total Claims": "Total Claims",
        "TotalInpatientClaims": "Total Inpatient Claims",
        "Total Inpatient Claims": "Total Inpatient Claims",
        "TotalOutpatientClaims": "Total Outpatient Claims",
        "Total Outpatient Claims": "Total Outpatient Claims",
        "UniqueBeneIDs": "Unique Bene ID",
        "Unique Bene ID": "Unique Bene ID",
        "AvgClaimDuration": "Avg Claim Duration",
        "Avg Claim Duration": "Avg Claim Duration",
        "PotentialFraud": "Potential Fraud",
        "Potential Fraud": "Potential Fraud"
    }

    # Determine which source columns are present (keep the left-most match)
    chosen_source_cols = []
    for src_col in col_map.keys():
        if src_col in display_df.columns and col_map[src_col] not in chosen_source_cols:
            chosen_source_cols.append(src_col)

    # We want the final UI order to follow the target labels list (deduped)
    desired_order = [
        "Provider ID",
        "Total Claims",
        "Total Inpatient Claims",
        "Total Outpatient Claims",
        "Unique Bene ID",
        "Avg Claim Duration",
        "Potential Fraud"
    ]

    # Build final dataframe: keep only the chosen source columns that map to our desired labels, in order
    final_source_cols_in_order = []
    for label in desired_order:
        # find a source column that maps to this label and is present
        for src_col, tgt_label in col_map.items():
            if tgt_label == label and src_col in display_df.columns and src_col not in final_source_cols_in_order:
                final_source_cols_in_order.append(src_col)
                break

    # If nothing found (defensive), fallback to a minimal safe set if present
    if not final_source_cols_in_order:
        fallback = ["ProviderID", "TotalClaims", "PotentialFraud"]
        final_source_cols_in_order = [c for c in fallback if c in display_df.columns]

    # Create the display DataFrame and rename columns to exact UI labels
    if final_source_cols_in_order:
        display_df = display_df[final_source_cols_in_order].rename(columns=col_map)
    else:
        display_df = pd.DataFrame(columns=desired_order)

    # If any desired column is missing, add an empty column so UI column order is preserved
    for label in desired_order:
        if label not in display_df.columns:
            display_df[label] = ""

    # Reorder to exact desired order
    display_df = display_df[desired_order]

    # Highlight rows: preserve previous behavior using results_df as truth source
    def highlight_fraud_rows(row):
        # Look up the original Predicted/Manual flags from results_df (not the renamed display_df)
        is_model_fraud = False
        try:
            if "PotentialFraud" in results_df.columns:
                is_model_fraud = str(results_df.at[row.name, "PotentialFraud"]).strip() == "Yes"
            elif "Potential Fraud" in results_df.columns:
                is_model_fraud = str(results_df.at[row.name, "Potential Fraud"]).strip() == "Yes"
        except Exception:
            is_model_fraud = False

        is_manual_flag = False
        try:
            if "ManualFlag" in results_df.columns:
                is_manual_flag = str(results_df.at[row.name, "ManualFlag"]).strip() == "Yes"
            elif "Manual Flag" in results_df.columns:
                is_manual_flag = str(results_df.at[row.name, "Manual Flag"]).strip() == "Yes"
        except Exception:
            is_manual_flag = False

        if is_model_fraud:
            return ['background-color: #ffcccc'] * len(row)
        if is_manual_flag:
            return ['background-color: #ffe6cc'] * len(row)
        return [''] * len(row)

    # Render styled dataframe (falls back to plain dataframe if styling fails)
    try:
        # Header style: blue background, white bold text, centered
        header_style = [
            {
                "selector": "th",
                "props": [
                    ("background-color", "#0000FF"),
                    ("color", "Black"),
                    ("font-weight", "bold"),
                    ("text-align", "center"),
                ],
            }
        ]

        # Apply row highlighting + header styling
        styled_df = (
            display_df.style
            .apply(highlight_fraud_rows, axis=1)  # fraud/manual highlight
            .set_table_styles(header_style)       # header style
            .set_properties(**{"text-align": "center"})  # center align cells
        )

        st.dataframe(styled_df, use_container_width=True)
    except Exception:
        st.dataframe(display_df, use_container_width=True)

    col1, col2 = st.columns([1, 1])  
    with col1:
        st.download_button(
            "📥 Download Predictions Excel",
            data=df_to_excel_with_highlight(results_df),
            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    with col2:
        st.download_button(
            "📄 Download Predictions PDF",
            data=df_to_pdf(display_df),
            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    # -----------------------------------------------------------
    # 🤖 Chatbot (FIXED - Enhanced with Proper Message Extraction)
    # -----------------------------------------------------------
    def render_chatbot_area():
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        st.markdown('<div class="ui-card"><h3>🤖 Chat with AI Assistant</h3>', unsafe_allow_html=True)
        st.markdown("<p style='color:#555;'>Ask questions about the analysis or explore insights further.</p>", unsafe_allow_html=True)

        # context builder (reuse your create_context_for_chatbot)
        def create_context_for_chatbot():
            if st.session_state.get("results_df") is not None:
                results_df = st.session_state.results_df
                results_summary = {
                    "total_providers": len(results_df),
                    "fraud_detected": len(results_df[results_df.get("PotentialFraud","") == "Yes"]),
                    "manual_flags": len(results_df[results_df.get("ManualFlag","") == "Yes"]),
                    "avg_fraud_probability": results_df.get("Fraud_Probability", pd.Series([0])).mean() if "Fraud_Probability" in results_df.columns else 0,
                }
                context = (
                    f"Analysis Results Context:\n"
                    f"- Total Providers Analyzed: {results_summary['total_providers']}\n"
                    f"- Potential Fraud Cases: {results_summary['fraud_detected']}\n"
                    f"- Manual Flags: {results_summary['manual_flags']}\n"
                    f"- Average Fraud Probability: {results_summary['avg_fraud_probability']:.3f}\n\n"
                    "The user is working with a fraud detection system for Medicare claims."
                )
                return context
            return "No analysis results available yet."

        # Use st.chat_input for a better UX
        user_message = st.chat_input(placeholder="Type a message and press Enter (or click Send)…")

        # If user submitted a message, handle it right away (no st.rerun)
        if user_message:
            # Append the user message immediately so UI shows it even while waiting
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.chat_history.append(("You", user_message, timestamp))

            # Call API (show spinner)
            context = create_context_for_chatbot()
            enhanced_message = f"{context}\n\nUser Question: {user_message}"

            with st.spinner("🤔 AI is thinking..."):
                api_result = call_langflow_api(message=enhanced_message, session_id=st.session_state.get("session_id"))
                if api_result["success"]:
                    bot_reply = extract_bot_message(api_result["response"])
                    if bot_reply.startswith("Response from"):
                        parts = bot_reply.split(": ", 1)
                        if len(parts) > 1:
                            bot_reply = parts[1]
                    st.session_state.chat_history.append(("Bot", bot_reply, timestamp))
                else:
                    err = api_result.get("error", "Unknown error")
                    st.session_state.chat_history.append(("Bot", f"⚠️ Error: {err}", timestamp))

        # Render chat history (last 20)
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for sender, msg, ts in st.session_state.get("chat_history", [])[-40:]:
            if sender == "You":
                st.markdown(f'''
                    <div class="user-message"><strong>You:</strong> {msg}
                        {f'<div class="chat-timestamp">{ts}</div>' if ts else ''}
                    </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                    <div class="bot-message"><strong>🤖 Assistant:</strong> {msg}
                        {f'<div class="chat-timestamp">{ts}</div>' if ts else ''}
                    </div>
                ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # actions
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []

    # call it where you previously had the chatbot UI
        render_chatbot_area()

    # Create context from results for the chatbot
    def create_context_for_chatbot():
        if st.session_state.results_df is not None:
            results_summary = {
                "total_providers": len(st.session_state.results_df),
                "fraud_detected": len(st.session_state.results_df[st.session_state.results_df.get("PotentialFraud", "") == "Yes"]),
                "manual_flags": len(st.session_state.results_df[st.session_state.results_df.get("ManualFlag", "") == "Yes"]),
                "avg_fraud_probability": st.session_state.results_df.get("Fraud_Probability", pd.Series([0])).mean() if "Fraud_Probability" in st.session_state.results_df.columns else 0,
            }
            context = f"""
            Analysis Results Context:
            - Total Providers Analyzed: {results_summary['total_providers']}
            - Potential Fraud Cases: {results_summary['fraud_detected']}
            - Manual Flags: {results_summary['manual_flags']}
            - Average Fraud Probability: {results_summary['avg_fraud_probability']:.3f}
            
            The user is working with a fraud detection system for Medicare claims.
            """
            return context
        return "No analysis results available yet."

    # Process pending message if exists
    if st.session_state.pending_chat_message:
        user_input = st.session_state.pending_chat_message
        st.session_state.pending_chat_message = ""  # Clear it immediately
        
        try:
            # Create enhanced message with context
            context = create_context_for_chatbot()
            enhanced_message = f"{context}\n\nUser Question: {user_input}"
            
            with st.spinner("🤔 AI is thinking..."):
                # Call Langflow API with session persistence
                api_result = call_langflow_api(
                    message=enhanced_message,
                    session_id=st.session_state.session_id
                )
                
                if api_result["success"]:
                    # FIXED: Use the improved extract_bot_message function
                    bot_reply = extract_bot_message(api_result["response"])
                    
                    # Clean up bot reply if needed
                    if bot_reply.startswith("Response from"):
                        # Try to extract just the meaningful part
                        parts = bot_reply.split(": ", 1)
                        if len(parts) > 1:
                            bot_reply = parts[1]
                    
                    # Save conversation with timestamp
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    st.session_state.chat_history.append(("You", user_input, timestamp))
                    st.session_state.chat_history.append(("Bot", bot_reply, timestamp))
                    
                    # Log successful interaction
                    logger.info(f"Successful chat interaction - User: {user_input[:50]}... Bot: {bot_reply[:50]}...")
                    
                else:
                    st.error(f"⚠️ Error: {api_result['error']}")
                    logger.error(f"Chat API error: {api_result['error']}")
                    
        except Exception as e:
            st.error(f"⚠️ Unexpected error: {str(e)}")
            logger.error(f"Chat error: {str(e)}")
        
        # Force rerun to clear the input and show new message
        st.rerun()

    # Chat input with enhanced UX
    st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([4, 1])
    with col1:
        # Use a unique key that doesn't conflict
        user_input = st.text_input(
            label = "",
            key="chat_text_input",
            placeholder="Chat with AI for the Querrys",
            label_visibility="collapsed"
        )
    with col2:
        send_button = st.button("Send", key="send_chat_button", use_container_width=True)

    # Handle send button or enter key
    if send_button and user_input:
        st.session_state.pending_chat_message = user_input
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Display chat history with improved styling
    if st.session_state.chat_history:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Show recent messages (limit to last 10 exchanges)
        recent_history = st.session_state.chat_history[-20:]  # Last 10 exchanges (20 messages)
        
        for sender, msg, timestamp in recent_history:
            if len(st.session_state.chat_history[0]) == 2:
                # Handle old format without timestamp
                timestamp = ""
                sender, msg = st.session_state.chat_history[recent_history.index((sender, msg, timestamp))][:2]
            
            if sender == "You":
                st.markdown(f'''
                <div class="user-message">
                    <strong>You:</strong> {msg}
                    {f'<div class="chat-timestamp">{timestamp}</div>' if timestamp else ''}
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="bot-message">
                    <strong>🤖 Assistant:</strong> {msg}
                    {f'<div class="chat-timestamp">{timestamp}</div>' if timestamp else ''}
                </div>
                ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Clear chat history button
        if st.button("Clear Chat History", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.info("💡 Start a conversation by asking a question about your fraud analysis results!")

    # # Quick action buttons with improved functionality
    # st.markdown("**Quick Questions:**")
    # col1, col2, col3 = st.columns(3)
    
    # with col1:
    #     if st.button("Explain Provider", key="quick_explain"):
    #         st.session_state.pending_chat_message = "details of PRV51150"
    #         st.rerun()
            
    # with col2:
    #     if st.button("Risk Provider", key="quick_risk"):
    #         st.session_state.pending_chat_message = "Top 5 fraud users"
    #         st.rerun()
    # with col3:
    #     if st.button("🎯 Next Steps", key="quick_next"):
    #         st.session_state.pending_chat_message = "What should I do next with these fraud detection results?"
    #         st.rerun()
    # st.markdown('</div>', unsafe_allow_html=True) 
    # -----------------------------------------------------------
    # Pipeline Info
    # -----------------------------------------------------------
    # st.markdown('<div class="ui-card"><h3>🧾 Pipeline Info</h3>', unsafe_allow_html=True)
    # st.markdown(f"<p><strong>Aggregation strategy:</strong> {st.session_state.info.get('strategy','N/A')}</p>", unsafe_allow_html=True)
    # st.markdown(f"<p><strong>Aggregated shape:</strong> {st.session_state.info.get('shape','N/A')}</p>", unsafe_allow_html=True)
    # st.markdown('</div>', unsafe_allow_html=True)
else:
    # -----------------------------------------------------------
    # Waiting for datasets message + Chatbot available even without results
    # -----------------------------------------------------------
    st.markdown("""
    <div class="ui-card">
        <h3>Waiting for datasets</h3>
        <p>Upload the required files in the top upload area and click <strong>Start Analysis</strong> to run aggregation & predictions. You can optionally enable "manual pre-check" in the sidebar to flag extreme providers before model inference.</p>
        <p>Once analysis is complete, you can chat with the AI assistant about your results!</p>
    </div>
    """, unsafe_allow_html=True)
    # -----------------------------------------------------------
    # 🤖 Chatbot (Available even without analysis results)
    # -----------------------------------------------------------
# Footer
# st.markdown("---")
# st.caption("Built with Streamlit • Upload → Start Analysis → Predict → Chat with AI • Powered by Langflow")