import streamlit as st
import numpy as np
import pickle
import io
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
import os
import re

# Model loading without caching
def load_model():
    model_path = "C:/Users/brij1/OneDrive/Desktop/PROJECTS/Autisim Project ML/trained_model.sav"
    try:
        if not os.path.exists(model_path):
            st.error("Model file 'trained_model.sav' not found. Please ensure it is in the correct directory.")
            return None
        return pickle.load(open(model_path, 'rb'))
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

loaded_model = load_model()
if loaded_model is None:
    st.stop()

# Custom CSS for responsive design
st.markdown("""
    <style>
    .stTextInput > div > input {
        width: 100%;
        padding: 8px;
    }
    .stSelectbox > div > select {
        width: 100%;
        padding: 8px;
    }
    .st-expander > div {
        margin-bottom: 15px;
    }
    .stButton > button {
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Page setup
st.set_page_config(page_title="Autism Predictor", page_icon="🧠", layout="wide")
st.title("🧠 Autism Spectrum Disorder Predictor")
st.markdown("Fill out all the details below to predict the risk of Autism Spectrum Disorder (ASD).")

# Initialize session state for inputs
if "form_data" not in st.session_state:
    st.session_state.form_data = {
        "user_name": "",
        "user_id": "",
        "a_scores": [0] * 10,
        "age_input": "21",
        "gender": "Male",
        "ethnicity": "White-European",
        "austim": "No",
        "relation": "Parent",
        "country": "United States",
        "jaundice": "No",
        "used_app": "No",
        "result": 3.0
    }

# Reset button
if st.button("🔄 Reset Form"):
    st.session_state.form_data = {
        "user_name": "",
        "user_id": "",
        "a_scores": [0] * 10,
        "age_input": "21",
        "gender": "Male",
        "ethnicity": "White-European",
        "austim": "No",
        "relation": "Parent",
        "country": "United States",
        "jaundice": "No",
        "used_app": "No",
        "result": 3.0
    }
    st.rerun()

# Patient Information
with st.expander("🧑 Patient Information", expanded=True):
    user_name = st.text_input(
        "Patient Name",
        value=st.session_state.form_data["user_name"],
        placeholder="Enter full name",
        key="user_name",
        help="Enter the full name of the patient (e.g., John Doe)."
    )
    user_id = st.text_input(
        "Patient ID",
        value=st.session_state.form_data["user_id"],
        placeholder="Enter unique ID",
        key="user_id",
        help="Enter a unique identifier for the patient (alphanumeric and hyphens only, e.g., ABC123)."
    )

    # Validate name and ID
    name_valid = bool(user_name.strip())
    id_valid = bool(user_id.strip() and re.match(r'^[a-zA-Z0-9-]+$', user_id))
    if not name_valid:
        st.warning("Please enter a valid patient name.")
    if not id_valid:
        st.warning("Please enter a valid patient ID (alphanumeric characters and hyphens only).")

# Behavioral Scores
with st.expander("📋 Behavioral Scores (AQ-10)", expanded=True):
    col1, col2 = st.columns([1, 1])
    a_scores = []
    with col1:
        for i in range(1, 6):
            score = st.selectbox(
                f"A{i} Score",
                ["No", "Yes"],
                index=1 if st.session_state.form_data["a_scores"][i-1] else 0,
                key=f"a{i}",
                help=f"Answer 'Yes' if the individual shows the behavior described in AQ-10 question {i} (e.g., difficulty in social interaction or communication)."
            )
            a_scores.append(1 if score == "Yes" else 0)
    with col2:
        for i in range(6, 11):
            score = st.selectbox(
                f"A{i} Score",
                ["No", "Yes"],
                index=1 if st.session_state.form_data["a_scores"][i-1] else 0,
                key=f"a{i}",
                help=f"Answer 'Yes' if the individual shows the behavior described in AQ-10 question {i} (e.g., repetitive behaviors or attention to detail)."
            )
            a_scores.append(1 if score == "Yes" else 0)

# Demographic and Clinical Information
with st.expander("🌍 Demographic and Clinical Information", expanded=True):
    col3, col4 = st.columns([1, 1])
    with col3:
        age_input = st.text_input(
            "Age",
            value=st.session_state.form_data["age_input"],
            placeholder="Enter age (1-100)",
            key="age_input",
            help="Enter the patient's age as an integer between 1 and 100."
        )
        age_valid = False
        try:
            age = int(age_input)
            if 1 <= age <= 100:
                age_valid = True
            else:
                st.warning("Age must be between 1 and 100.")
        except ValueError:
            st.warning("Please enter a valid integer for age.")
        
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"],
            index=0 if st.session_state.form_data["gender"] == "Male" else 1,
            key="gender",
            help="Select the gender of the patient."
        )
        gender_val = 1 if gender == "Male" else 0
        ethnicity = st.selectbox(
            "Ethnicity",
            sorted(["White-European", "Black", "Asian", "Middle Eastern", "Latino", "Others", "Pacifica", "Hispanic", "South Asian", "Turkish", "Mixed", "?"]),
            index=sorted(["White-European", "Black", "Asian", "Middle Eastern", "Latino", "Others", "Pacifica", "Hispanic", "South Asian", "Turkish", "Mixed", "?"]).index(st.session_state.form_data["ethnicity"]),
            key="ethnicity",
            help="Select the ethnicity of the patient."
        )
        ethnicity_map = {e.lower(): i for i, e in enumerate(sorted(["White-European", "Black", "Asian", "Middle Eastern", "Latino", "Others", "Pacifica", "Hispanic", "South Asian", "Turkish", "Mixed", "?"]))}
        ethnicity_val = ethnicity_map[ethnicity.lower()]
        austim = st.selectbox(
            "Family Member with Autism",
            ["No", "Yes"],
            index=1 if st.session_state.form_data["austim"] == "Yes" else 0,
            key="austim",
            help="Indicate if a family member has been diagnosed with autism."
        )
        austim_val = 1 if austim == "Yes" else 0
        relation = st.selectbox(
            "Relation to Test Subject",
            sorted(["Parent", "Self", "Relative", "Health care professional", "Others", "?", "Teacher"]),
            index=sorted(["Parent", "Self", "Relative", "Health care professional", "Others", "?", "Teacher"]).index(st.session_state.form_data["relation"]),
            key="relation",
            help="Select your relation to the individual being screened."
        )
        relation_map = {r.lower(): i for i, r in enumerate(sorted(["Parent", "Self", "Relative", "Health care professional", "Others", "?", "Teacher"]))}
        relation_val = relation_map[relation.lower()]

    with col4:
        country = st.selectbox(
            "Country of Residence",
            sorted(["United States", "Brazil", "New Zealand", "Bhutan", "India", "United Kingdom", "Austria", "Argentina", "Japan", "China", "Ireland", "Australia", "Others", "?", "South Africa", "United Arab Emirates"]),
            index=sorted(["United States", "Brazil", "New Zealand", "Bhutan", "India", "United Kingdom", "Austria", "Argentina", "Japan", "China", "Ireland", "Australia", "Others", "?", "South Africa", "United Arab Emirates"]).index(st.session_state.form_data["country"]),
            key="country",
            help="Select the country of residence of the patient."
        )
        country_map = {c.lower(): i for i, c in enumerate(sorted(["United States", "Brazil", "New Zealand", "Bhutan", "India", "United Kingdom", "Austria", "Argentina", "Japan", "China", "Ireland", "Australia", "Others", "?", "South Africa", "United Arab Emirates"]))}
        country_val = country_map[country.lower()]
        jaundice = st.selectbox(
            "History of Jaundice",
            ["No", "Yes"],
            index=1 if st.session_state.form_data["jaundice"] == "Yes" else 0,
            key="jaundice",
            help="Indicate if the individual had jaundice at birth."
        )
        jaundice_val = 1 if jaundice == "Yes" else 0
        used_app = st.selectbox(
            "Used Screening App Before",
            ["No", "Yes"],
            index=1 if st.session_state.form_data["used_app"] == "Yes" else 0,
            key="used_app",
            help="Indicate if a screening app was used previously for this individual."
        )
        used_app_val = 1 if used_app == "Yes" else 0
        result = st.selectbox(
            "AQ-10 Screening App Score",
            [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            index=[3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0].index(st.session_state.form_data["result"]),
            key="result",
            help="Select the total score from the AQ-10 screening questionnaire."
        )

# Update session state
st.session_state.form_data.update({
    "user_name": user_name,
    "user_id": user_id,
    "a_scores": a_scores,
    "age_input": age_input,
    "gender": gender,
    "ethnicity": ethnicity,
    "austim": austim,
    "relation": relation,
    "country": country,
    "jaundice": jaundice,
    "used_app": used_app,
    "result": result
})

# Prepare input data
input_data = np.array([a_scores + [age, gender_val, ethnicity_val, jaundice_val, austim_val, country_val, used_app_val, result, relation_val]])

# Function to generate plots
def generate_plots(a_scores, proba):
    plots = []
    
    # Bar plot for A1-A10 scores
    try:
        plt.figure(figsize=(8, 4))
        sns.set_style("whitegrid")
        sns.barplot(x=[f"A{i}" for i in range(1, 11)], y=a_scores, palette="Blues_d")
        plt.title("A1-A10 Scores")
        plt.ylabel("Score (0 = No, 1 = Yes)")
        plt.ylim(0, 1.5)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close()
        buf.seek(0)
        plots.append(buf)
    except Exception as e:
        st.warning(f"Error generating bar plot: {str(e)}")
        plots.append(None)

    # Pie chart for prediction probability
    if proba is not None:
        try:
            plt.figure(figsize=(4, 4))
            labels = ["High Risk", "Low Risk"]
            sizes = [proba, 1 - proba]
            colors = ["#FF6B6B", "#6BCB77"]  # Use hex colors directly for matplotlib
            plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
            plt.title("Prediction Probability")
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            plt.close()
            buf.seek(0)
            plots.append(buf)
        except Exception as e:
            st.warning(f"Error generating pie chart: {str(e)}")
            plots.append(None)
    else:
        plots.append(None)
    
    return plots

# Generate Report
def generate_report(name, pid, summary, prediction_text, proba, a_scores):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    
    # Custom styles
    header_style = ParagraphStyle(
        name="Header",
        parent=styles["Title"],
        fontSize=16,
        textColor=colors.white,
        spaceAfter=12,
        alignment=1  # Center
    )
    normal_style = ParagraphStyle(
        name="NormalBold",
        parent=styles["Normal"],
        fontSize=10,
        leading=12,
        spaceAfter=6
    )
    section_style = ParagraphStyle(
        name="Section",
        parent=styles["Heading2"],
        fontSize=12,
        textColor=HexColor("#004aad"),
        spaceBefore=12,
        spaceAfter=6
    )
    disclaimer_style = ParagraphStyle(
        name="Disclaimer",
        parent=styles["Italic"],
        fontSize=8,
        textColor=colors.gray,
        spaceBefore=12
    )

    # Colors
    primary_color = HexColor("#004aad")  # Deep blue
    secondary_color = HexColor("#6BCB77")  # Soft green
    bg_color = HexColor("#F5F6F5")  # Light gray

    # Header and footer
    def add_header_footer(canvas, doc):
        canvas.saveState()
        # Header
        canvas.setFillColor(primary_color)
        canvas.rect(0, A4[1] - 0.5*inch, A4[0], 0.5*inch, fill=1)
        canvas.setFont("Helvetica-Bold", 12)
        canvas.setFillColor(colors.white)
        canvas.drawCentredString(A4[0]/2, A4[1] - 0.35*inch, "Autism Screening Report")
        # Footer
        canvas.setFillColor(colors.gray)
        canvas.setFont("Helvetica", 8)
        canvas.drawString(0.5*inch, 0.3*inch, "Generated by ASD Predictor | Not a medical diagnosis")
        canvas.restoreState()

    elements = []

    # Cover Page
    elements.append(Paragraph("Autism Spectrum Disorder Screening Report", header_style))
    elements.append(Spacer(1, 0.5*inch))
    elements.append(Paragraph(f"Patient: <b>{name}</b>", normal_style))
    elements.append(Paragraph(f"ID: <b>{pid}</b>", normal_style))
    elements.append(Paragraph(f"Date: <b>{datetime.now().strftime('%d-%m-%Y %H:%M')}</b>", normal_style))
    elements.append(Spacer(1, 1*inch))
    elements.append(Paragraph("Prepared by ASD Predictor", normal_style))
    elements.append(Spacer(1, 0.5*inch))
    elements.append(Paragraph("This report provides a preliminary screening assessment for Autism Spectrum Disorder (ASD) based on provided inputs. Consult a healthcare professional for a comprehensive evaluation.", disclaimer_style))
    
    # New page
    elements.append(Spacer(1, 0))

    # Introduction
    elements.append(Paragraph("Introduction", section_style))
    elements.append(Paragraph("This report presents the results of an Autism Spectrum Disorder (ASD) risk screening based on a machine learning model. The assessment uses responses to the AQ-10 questionnaire and additional demographic and clinical information to estimate the likelihood of ASD. The results are intended for informational purposes and should be followed up with a professional medical evaluation.", normal_style))

    # Prediction Result
    elements.append(Paragraph("Prediction Result", section_style))
    elements.append(Paragraph(f"<b>{prediction_text}</b>", normal_style))
    if proba is not None:
        risk_level = "High" if proba >= 0.5 else "Low"
        elements.append(Paragraph(f"<b>Confidence:</b> {proba:.2%} ({risk_level} Risk)", normal_style))
        if 0.45 <= proba <= 0.55:
            elements.append(Paragraph("Note: The confidence level is close to 50%, indicating uncertainty in the prediction. Professional evaluation is strongly recommended.", normal_style))
    elements.append(Paragraph("The prediction indicates the estimated risk of ASD based on the input data. A 'High Risk' result suggests a higher likelihood of ASD, while a 'Low Risk' result suggests a lower likelihood. Always consult a licensed professional for a definitive diagnosis.", normal_style))

    # AQ-10 Score Breakdown
    elements.append(Paragraph("AQ-10 Score Breakdown", section_style))
    score_sum = sum(a_scores)
    elements.append(Paragraph(f"Total AQ-10 Score: <b>{score_sum}/10</b>", normal_style))
    elements.append(Paragraph("The AQ-10 score is the sum of 'Yes' responses to the A1–A10 questions. Scores of 6 or higher typically indicate a higher likelihood of ASD, warranting further evaluation.", normal_style))
    table_data = [[f"A{i}", "Yes" if a_scores[i-1] else "No"] for i in range(1, 11)]
    table_data.insert(0, ["Question", "Response"])
    score_table = Table(table_data, colWidths=[100, 100])
    score_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('BACKGROUND', (0, 1), (-1, -1), bg_color),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('LEADING', (0, 0), (-1, -1), 12),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ]))
    elements.append(score_table)

    # Input Summary
    elements.append(Paragraph("Summary of Inputs", section_style))
    table_data = [[k, str(v)] for k, v in summary.items()]
    table = Table(table_data, colWidths=[200, 300])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('BACKGROUND', (0, 1), (-1, -1), bg_color),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('LEADING', (0, 0), (-1, -1), 12),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ]))
    elements.append(table)

    # Analysis Graphs
    elements.append(Paragraph("Analysis Visualizations", section_style))
    plots = generate_plots(a_scores, proba)
    if plots[0] is not None:
        elements.append(Paragraph("A1-A10 Scores Distribution", normal_style))
        elements.append(Image(plots[0], width=4*inch, height=2*inch))
    else:
        elements.append(Paragraph("Unable to generate A1-A10 Scores plot.", normal_style))
    if proba is not None and plots[1] is not None:
        elements.append(Paragraph("Prediction Probability", normal_style))
        elements.append(Image(plots[1], width=2*inch, height=2*inch))
    else:
        elements.append(Paragraph("Prediction probability plot not available.", normal_style))

    # ASD Information
    elements.append(Paragraph("About ASD Screening", section_style))
    elements.append(Paragraph("Autism Spectrum Disorder (ASD) is a developmental condition that affects communication, behavior, and social interaction. Screening tools like the AQ-10 help identify individuals who may benefit from further evaluation. Key indicators include responses to behavioral questions (A1-A10) and factors such as family history and medical conditions like jaundice.", normal_style))
    elements.append(Paragraph("This screening uses a machine learning model trained on historical data to predict ASD risk. While useful, it is not a substitute for professional diagnosis.", normal_style))

    # Disclaimer and Follow-up
    elements.append(Paragraph("Disclaimer and Follow-up", section_style))
    elements.append(Paragraph("<b>Disclaimer:</b> This report is not a medical diagnosis. It is a screening tool designed to provide preliminary insights.", disclaimer_style))
    elements.append(Paragraph("<b>Follow-up:</b> If the report indicates high risk, consult a licensed healthcare professional for a comprehensive evaluation.", disclaimer_style))

    # Build PDF with header and footer
    try:
        doc.build(elements, onFirstPage=add_header_footer, onLaterPages=add_header_footer)
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        return None
    buffer.seek(0)
    return buffer

# Prediction
all_inputs_valid = name_valid and id_valid and age_valid
if st.button("🔍 Predict ASD Risk", disabled=not all_inputs_valid):
    try:
        prediction = loaded_model.predict(input_data)[0]
        proba = None
        if hasattr(loaded_model, 'predict_proba'):
            proba = loaded_model.predict_proba(input_data)[0][1]

        prediction_text = "⚠ High Risk of Autism Spectrum Disorder" if prediction == 1 else "✅ Low Risk of Autism Spectrum Disorder"

        st.markdown("### 🧪 Result")
        if prediction == 1:
            st.error(prediction_text)
        else:
            st.success(prediction_text)
        if proba is not None:
            st.info(f"📊 Prediction Confidence: *{proba:.2%}*")
            if 0.45 <= proba <= 0.55:
                st.warning("The confidence level is close to 50%, indicating uncertainty in the prediction. Professional evaluation is strongly recommended.")

        input_summary = {
            "A1-A10 Scores": a_scores,
            "Age": age,
            "Gender": gender,
            "Ethnicity": ethnicity,
            "Jaundice": jaundice,
            "Family Member with Autism": austim,
            "Country": country,
            "Used App Before": used_app,
            "AQ-10 Score": result,
            "Relation": relation
        }

        pdf_buffer = generate_report(user_name, user_id, input_summary, prediction_text, proba, a_scores)
        if pdf_buffer is not None:
            st.download_button("📥 Download Report (PDF)", data=pdf_buffer, file_name="ASD_Report.pdf", mime="application/pdf")
        else:
            st.error("Failed to generate PDF report.")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")