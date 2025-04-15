import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import scipy.stats as stats
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter


# Set page configuration
st.set_page_config(
    page_title="Health Assistant",
    layout="wide", 
    page_icon="🧑‍⚕️",
    initial_sidebar_state="expanded"
)



# CSS for styling
# CSS for styling
css = """
<style>
    /* Set a soft background color for the entire app */
    .stApp {
        background-color: #f0f4f8;  /* Light grey background */
        color: #333;                 /* Dark text color for readability */
    }

    /* Style sidebar */
    .css-1d391kg {
        background-color: #34495e;   /* Darker sidebar color */
        color: #ecf0f1;               /* Light text color */
        padding: 20px;                /* Padding inside sidebar */
        border-radius: 8px;          /* Rounded corners */
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1); /* Subtle shadow */
    }

    /* Style sidebar headers */
    .stSidebar h2 {
        font-size: 20px;              /* Header font size */
        color: #ecf0f1;               /* Light color for sidebar headers */
    }

    /* Style sidebar images */
    .stSidebar img {
        border-radius: 8px;          /* Rounded corners for images */
        margin-bottom: 10px;         /* Space below image */
    }

    /* Style buttons in sidebar */
    .stButton {
        background-color: #1abc9c;  /* Button color */
        color: black;                /* White text color */
        border: none;                /* No border */
        border-radius: 5px;         /* Rounded corners */
        padding: 10px;               /* Padding */
        width: 100%;                 /* Full width */
        margin: 5px 0;              /* Space between buttons */
    }
    .stButton:hover {
        background-color: #16a085;  /* Darker shade on hover */
    }

    /* Style text inputs in sidebar */
    .stTextInput, .stTextArea {
        border-radius: 5px;          /* Rounded corners */
        border: 1px solid #bdc3c7;   /* Light grey border */
        padding: 10px;               /* Padding */
        background-color: #ffffff;   /* White background for input fields */
        margin-bottom: 10px;         /* Space below input fields */
    }
</style>
"""

# Apply the CSS styles
st.markdown(css, unsafe_allow_html=True)

    
# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# loading the saved models

diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))

heart_disease_model = pickle.load(open('heart_disease_model.sav', 'rb'))

parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))

# sidebar for navigation
with st.sidebar:
    st.image("download.png", use_column_width=True)
    st.title("Health Assistant")
    selected = option_menu('Disease Prediction',
                           ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
                           icons=['activity', 'heart', 'person'],
                           menu_icon='hospital',
                           default_index=0)


doctors_for_conditions = {
    'diabetes': [
        {'name': 'Dr. Alice Johnson', 'specialty': 'Endocrinologist', 'contact': 'alice.johnson@example.com'},
        {'name': 'Dr. Bob Smith', 'specialty': 'Diabetologist', 'contact': 'bob.smith@example.com'}
    ],
    'heart disease': [
        {'name': 'Dr. Sarah Williams', 'specialty': 'Cardiologist', 'contact': 'sarah.williams@example.com'},
        {'name': 'Dr. Mark Lee', 'specialty': 'Cardiologist', 'contact': 'mark.lee@example.com'}
    ],
    'Parkinson’s disease': [
        {'name': 'Dr. Michael Brown', 'specialty': 'Neurologist', 'contact': 'michael.brown@example.com'},
        {'name': 'Dr. Lisa Taylor', 'specialty': 'Neurologist', 'contact': 'lisa.taylor@example.com'}
    ]
}

import streamlit as st
import numpy as np
from scipy import stats

# Sigmoid function to convert raw model scores to probability
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.header('🩺 Diabetes Prediction using Machine Learning')
    st.markdown("Provide the following health parameters to assess diabetes risk:")

    patient_name = st.text_input("🧍 Patient Name")
    patient_id = st.text_input("🆔 Patient ID")

    # Input form section
    with st.expander("🔍 Input Parameters"):
        col1, col2, col3 = st.columns(3)
        with col1:
            Gender = st.text_input('Gender (0 - Female, 1 - Male)', help="Indicate gender. For males, pregnancy-related fields are 0.")
        with col2:
            Age = st.text_input('Age', help="Enter your age in years.")
        with col3:
            Urea = st.text_input('Urea (mg/dL)', help="Normal range: 7-20 mg/dL.")

        with col1:
            Cr = st.text_input('Creatinine (Cr)', help="Normal range: 0.6 to 1.3 mg/dL. Indicates kidney function.")
        with col2:
            HbA1c = st.text_input('HbA1c (%)', help="Average blood sugar levels over the past 2-3 months. >6.5% indicates diabetes.")
        with col3:
            Chol = st.text_input('Cholesterol (mg/dL)', help="Total cholesterol. Normal is <200 mg/dL.")

        with col1:
            TG = st.text_input('Triglycerides (TG)', help="Normal range: <150 mg/dL.")
        with col2:
            HDL = st.text_input('HDL (mg/dL)', help="High-Density Lipoprotein. >40 mg/dL is considered good.")
        with col3:
            LDL = st.text_input('LDL (mg/dL)', help="Low-Density Lipoprotein. Optimal <100 mg/dL.")

        with col1:
            VLDL = st.text_input('VLDL (mg/dL)', help="Very Low-Density Lipoprotein. Normal: 2–30 mg/dL.")
        with col2:
            BMI = st.text_input('BMI', help="Body Mass Index. Normal range: 18.5 - 24.9.")

    # Initialize result variable
    diab_diagnosis = ''

    # Prediction button logic
    if st.button('🚨 Check Diabetes Test Result'):
        try:
            # Convert input to float
            user_input = [
                float(Gender), float(Age), float(Urea), float(Cr), float(HbA1c),
                float(Chol), float(TG), float(HDL), float(LDL), float(VLDL), float(BMI)
            ]

            raw_score = diabetes_model.decision_function([user_input])[0]
            prob_diabetic = sigmoid(raw_score)
            probability_percentage = prob_diabetic * 100

            # Confidence interval
            std_dev = 0.10  # Ideally, calculate from model evaluation
            confidence_level = 0.90
            margin_of_error = stats.norm.ppf((1 + confidence_level) / 2) * std_dev
            lower_bound = max(0, (prob_diabetic - margin_of_error) * 100)
            upper_bound = min(100, (prob_diabetic + margin_of_error) * 100)

            with st.spinner('🔎 Analyzing your data...'):
                diab_prediction = diabetes_model.predict([user_input])

                if diab_prediction[0] == 1:
                    st.error(f'🚨 You are likely diabetic with a probability of {probability_percentage:.2f}% '
                             f'(Confidence Range: {lower_bound:.2f}% - {upper_bound:.2f}%)')
                    
                    def generate_pdf_report(name, uid, probability, lower_bound, upper_bound, result, doctor_list=None):
                        buffer = BytesIO()
                        c = canvas.Canvas(buffer, pagesize=letter)
                        width, height = letter

                        c.setFont("Helvetica-Bold", 16)
                        c.drawString(50, height - 50, "🧠 Diabetes Disease Prediction Report")

                        c.setFont("Helvetica", 12)
                        c.drawString(50, height - 100, f"Patient Name: {name}")
                        c.drawString(50, height - 120, f"Patient ID: {uid}")
                        c.drawString(50, height - 140, f"Prediction Result: {'High Risk' if result == 1 else 'Low Risk'}")
                        c.drawString(50, height - 160, f"Probability: {probability:.2f}%")
                        c.drawString(50, height - 180, f"Confidence Interval: {lower_bound:.2f}% - {upper_bound:.2f}%")

                        if result == 1 and doctor_list:
                            c.setFont("Helvetica-Bold", 14)
                            c.drawString(50, height - 220, "Recommended Doctors:")
                            y = height - 240
                            c.setFont("Helvetica", 11)
                            for doctor in doctor_list:
                                c.drawString(60, y, f"Name: {doctor['name']}")
                                y -= 15
                                c.drawString(60, y, f"Specialty: {doctor['specialty']}")
                                y -= 15
                                c.drawString(60, y, f"Contact: {doctor['contact']}")
                                y -= 25
                                if y < 100:
                                    c.showPage()
                                    y = height - 50

                        c.save()
                        buffer.seek(0)
                        return buffer
                    
                    pdf_buffer = generate_pdf_report(
                    name=patient_name,
                    uid=patient_id,
                    probability=probability_percentage,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    result=diab_prediction,
                    doctor_list=doctors_for_conditions['diabetes'] if diab_prediction == 1 else None
                )
                    
                    st.download_button(
                        label="📄 Download Report Card as PDF",
                        data=pdf_buffer,
                        file_name="Diabetes_Report.pdf",
                        mime="application/pdf"
                    )


                    st.subheader("👨‍⚕️ Recommended Doctors for Diabetes:")
                    for doctor in doctors_for_conditions['diabetes']:
                        st.markdown(f"""
                        **Name:** {doctor['name']}  
                        **Specialty:** {doctor['specialty']}  
                        **Contact:** {doctor['contact']}  
                         **---------------------------------------------------**
                        """)
                else:
                    st.success(f'✅ You are likely not diabetic with a probability of {probability_percentage:.2f}%')

        except ValueError:
            st.error("❌ Please fill in all fields with valid numeric values.")

    # Educational info
    st.markdown("""
    ### ℹ️ About Diabetes:
    Diabetes is a chronic condition affecting how your body processes blood sugar (glucose).  
    It occurs when insulin production is inadequate or when the body cannot effectively use it.

    **Types of Diabetes:**
    - **Type 1:** Autoimmune condition where the body attacks insulin-producing cells.
    - **Type 2:** Insulin resistance develops over time, often linked to lifestyle.
    - **Gestational:** Develops during pregnancy and usually resolves post-delivery.

    **Common Risk Factors:**
    - Obesity or being overweight
    - Sedentary lifestyle
    - Family history
    - Poor diet

    **Symptoms:**
    - Frequent urination and thirst
    - Fatigue and blurred vision
    - Slow healing wounds

    **Prevention & Management:**
    - Maintain a healthy diet
    - Exercise regularly
    - Monitor blood glucose levels
    """)

    st.success(diab_diagnosis)


# Heart Disease Prediction Page
# Heart Disease Prediction Page
import streamlit as st
import numpy as np
from scipy import stats

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.header('💓 Heart Disease Prediction using Machine Learning')
    st.markdown("Provide the following health parameters to assess your risk for heart disease:")

    patient_name = st.text_input("🧍 Patient Name")
    patient_id = st.text_input("🆔 Patient ID")

    # Input Section
    with st.expander("🔍 Input Parameters"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.text_input('Age', help="Enter age in years. Heart disease risk increases with age.")
        with col2:
            sex = st.text_input('Sex (0 - Female, 1 - Male)', help="0 for Female, 1 for Male.")
        with col3:
            cp = st.text_input('Chest Pain Type (0-3)', help="0: Typical Angina, 1: Atypical Angina, 2: Non-anginal, 3: Asymptomatic.")

        with col1:
            trestbps = st.text_input('Resting Blood Pressure (mmHg)', help="Normal is <120/80 mmHg.")
        with col2:
            chol = st.text_input('Serum Cholesterol (mg/dL)', help="Ideal level is <200 mg/dL.")
        with col3:
            fbs = st.text_input('Fasting Blood Sugar > 120 mg/dL (1 = Yes, 0 = No)', help="Enter 1 if >120 mg/dL, else 0.")

        with col1:
            restecg = st.text_input('Resting ECG Result (0-2)', help="0: Normal, 1: ST-T abnormality, 2: Left Ventricular Hypertrophy.")
        with col2:
            thalach = st.text_input('Max Heart Rate Achieved', help="Higher is generally better. Normal max is 220 - age.")
        with col3:
            exang = st.text_input('Exercise Induced Angina (1 = Yes, 0 = No)', help="Enter 1 if angina during exercise, else 0.")

        with col1:
            oldpeak = st.text_input('ST Depression (Oldpeak)', help="ST depression induced by exercise relative to rest.")
        with col2:
            slope = st.text_input('Slope of Peak Exercise ST Segment (0-2)', help="0: Upsloping, 1: Flat, 2: Downsloping.")
        with col3:
            ca = st.text_input('Number of Major Vessels (0-3)', help="Detected by fluoroscopy. More vessels = higher risk.")

        with col1:
            thal = st.text_input('Thalassemia (0-2)', help="0: Normal, 1: Fixed defect, 2: Reversible defect.")

    # Sigmoid function to convert raw score to probability
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Prediction Section
    if st.button('🧠 Analyze Heart Disease Risk'):
        try:
            # Gather and convert input values
            user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                          exang, oldpeak, slope, ca, thal]
            user_input = [float(x) for x in user_input]

            raw_score = heart_disease_model.decision_function([user_input])[0]
            prob_heart = sigmoid(raw_score)
            probability_percentage_heart = prob_heart * 100

            # Confidence interval calculation
            std_dev = 0.10  # Adjust based on your dataset
            confidence_level = 0.90
            margin_of_error = stats.norm.ppf((1 + confidence_level) / 2) * std_dev
            lower_bound = max(0, (prob_heart - margin_of_error) * 100)
            upper_bound = min(100, (prob_heart + margin_of_error) * 100)

            with st.spinner('⏳ Processing...'):
                heart_prediction = heart_disease_model.predict([user_input])

                if heart_prediction == 1:
                    st.error(f'🚨 High Risk of Heart Disease Detected! Probability: {probability_percentage_heart:.2f}% '
                             f'(Confidence Range: {lower_bound:.2f}% - {upper_bound:.2f}%)')
                    

                    def generate_pdf_report(name, uid, probability, lower_bound, upper_bound, result, doctor_list=None):
                        buffer = BytesIO()
                        c = canvas.Canvas(buffer, pagesize=letter)
                        width, height = letter

                        c.setFont("Helvetica-Bold", 16)
                        c.drawString(50, height - 50, "🧠 Heart Disease Prediction Report")

                        c.setFont("Helvetica", 12)
                        c.drawString(50, height - 100, f"Patient Name: {name}")
                        c.drawString(50, height - 120, f"Patient ID: {uid}")
                        c.drawString(50, height - 140, f"Prediction Result: {'High Risk' if result == 1 else 'Low Risk'}")
                        c.drawString(50, height - 160, f"Probability: {probability:.2f}%")
                        c.drawString(50, height - 180, f"Confidence Interval: {lower_bound:.2f}% - {upper_bound:.2f}%")

                        if result == 1 and doctor_list:
                            c.setFont("Helvetica-Bold", 14)
                            c.drawString(50, height - 220, "Recommended Doctors:")
                            y = height - 240
                            c.setFont("Helvetica", 11)
                            for doctor in doctor_list:
                                c.drawString(60, y, f"Name: {doctor['name']}")
                                y -= 15
                                c.drawString(60, y, f"Specialty: {doctor['specialty']}")
                                y -= 15
                                c.drawString(60, y, f"Contact: {doctor['contact']}")
                                y -= 25
                                if y < 100:
                                    c.showPage()
                                    y = height - 50

                        c.save()
                        buffer.seek(0)
                        return buffer
                    
                    pdf_buffer = generate_pdf_report(
                    name=patient_name,
                    uid=patient_id,
                    probability=probability_percentage_heart,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    result=heart_prediction,
                    doctor_list=doctors_for_conditions['heart disease'] if heart_prediction == 1 else None
                )
                    
                    st.download_button(
                        label="📄 Download Report Card as PDF",
                        data=pdf_buffer,
                        file_name="Heart_Report.pdf",
                        mime="application/pdf"
                    )
                    st.subheader("🩺 Recommended Cardiologists:")
                    for doctor in doctors_for_conditions['heart disease']:
                        st.markdown(f"""
                        **Name:** {doctor['name']}  
                        **Specialty:** {doctor['specialty']}  
                        **Contact:** {doctor['contact']}  
                        **---------------------------------------------------**
                        """)
                else:
                    st.success(f'❤️ You appear to have low risk of heart disease. Probability: {probability_percentage_heart:.2f}%')
        except ValueError:
            st.error("❌ Please ensure all input fields are filled with valid numbers.")

    # Info Section
    st.markdown("""
    ### ℹ️ About Heart Disease:
    Heart disease includes conditions like coronary artery disease, arrhythmias, and heart failure.  
    It is the **leading cause of death globally** but is **largely preventable**.

    **Common Risk Factors:**
    - High blood pressure
    - High cholesterol
    - Smoking
    - Diabetes
    - Obesity and inactivity

    **Symptoms May Include:**
    - Chest pain or discomfort
    - Shortness of breath
    - Palpitations or irregular heartbeat
    - Fatigue or weakness

    **Preventive Measures:**
    - Maintain a heart-healthy diet
    - Regular exercise
    - Monitor blood pressure & cholesterol
    - Quit smoking
    - Manage stress and sleep

    Early detection and lifestyle modification can significantly reduce the risk of heart complications.
    """)


# Parkinson's Prediction Page
import streamlit as st
import numpy as np
from scipy import stats

# Parkinson’s Disease Prediction Page
if selected == "Parkinsons Prediction":
    st.header("🧠 Parkinson's Disease Prediction using Machine Learning")
    st.markdown("Fill in the vocal measurements below to analyze the risk of Parkinson's Disease.")

    patient_name = st.text_input("🧍 Patient Name")
    patient_id = st.text_input("🆔 Patient ID")

    # Input Section
    with st.expander("🎙️ Input Parameters"):
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            fo = st.text_input('F0 (Hz)', help="Average pitch (MDVP: Foa)")
        with col2:
            fhi = st.text_input('Fhi (Hz)', help="Max pitch (MDVP: Fhi)")
        with col3:
            flo = st.text_input('Flo (Hz)', help="Min pitch (MDVP: Flo)")
        with col4:
            Jitter_percent = st.text_input('Jitter (%)', help="Pitch variation percentage")
        with col5:
            Jitter_Abs = st.text_input('Jitter (Abs)', help="Absolute pitch variation")

        with col1:
            RAP = st.text_input('RAP', help="Short-term pitch fluctuation")
        with col2:
            PPQ = st.text_input('PPQ', help="Longer-term pitch instability")
        with col3:
            DDP = st.text_input('DDP', help="Combined pitch variation")
        with col4:
            Shimmer = st.text_input('Shimmer (%)', help="Amplitude variation in %")
        with col5:
            Shimmer_dB = st.text_input('Shimmer (dB)', help="Amplitude variation in decibels")

        with col1:
            APQ3 = st.text_input('APQ3', help="Amplitude instability over 3 periods")
        with col2:
            APQ5 = st.text_input('APQ5', help="Amplitude instability over 5 periods")
        with col3:
            APQ = st.text_input('APQ', help="Overall amplitude perturbation")
        with col4:
            DDA = st.text_input('DDA', help="Combined shimmer measures")
        with col5:
            NHR = st.text_input('NHR', help="Noise-to-Harmonics Ratio")

        with col1:
            HNR = st.text_input('HNR', help="Harmonics-to-Noise Ratio")
        with col2:
            RPDE = st.text_input('RPDE', help="Signal complexity")
        with col3:
            DFA = st.text_input('DFA', help="Voice signal self-similarity")
        with col4:
            spread1 = st.text_input('Spread1', help="Frequency distribution spread")
        with col5:
            spread2 = st.text_input('Spread2', help="Additional frequency distribution detail")

        with col1:
            D2 = st.text_input('D2', help="Dynamical complexity of voice")
        with col2:
            PPE = st.text_input('PPE', help="Pitch variation over time")

    # Sigmoid Function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Prediction Section
    if st.button("🧪 Analyze Parkinson's Risk"):
        try:
            user_input = [
                fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP,
                Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR,
                RPDE, DFA, spread1, spread2, D2, PPE
            ]
            user_input = [float(x) for x in user_input]

            raw_score = parkinsons_model.decision_function([user_input])[0]
            prob_park = sigmoid(raw_score)
            probability_percentage_park = prob_park * 100

            # Confidence Interval
            std_dev = 0.10
            confidence_level = 0.90
            margin_of_error = stats.norm.ppf((1 + confidence_level) / 2) * std_dev

            lower_bound = max(0, (prob_park - margin_of_error) * 100)
            upper_bound = min(100, (prob_park + margin_of_error) * 100)

            with st.spinner('⏳ Analyzing...'):
                result = parkinsons_model.predict([user_input])
                if result == 1:
                    st.error(f"🚨 High risk of Parkinson's disease detected!\n\n"
                             f"Probability: **{probability_percentage_park:.2f}%**\n"
                             f"Confidence Interval: **{lower_bound:.2f}% - {upper_bound:.2f}%**")
                    
                    def generate_pdf_report(name, uid,probability, lower_bound, upper_bound, result, doctor_list=None):
                        buffer = BytesIO()
                        c = canvas.Canvas(buffer, pagesize=letter)
                        width, height = letter

                        c.setFont("Helvetica-Bold", 16)
                        c.drawString(50, height - 50, "🧠 Parkinson's Disease Prediction Report")

                        c.setFont("Helvetica", 12)
                        c.drawString(50, height - 100, f"Patient Name: {name}")
                        c.drawString(50, height - 120, f"Patient ID: {uid}")
                        c.drawString(50, height - 140, f"Prediction Result: {'High Risk' if result == 1 else 'Low Risk'}")
                        c.drawString(50, height - 160, f"Probability: {probability:.2f}%")
                        c.drawString(50, height - 180, f"Confidence Interval: {lower_bound:.2f}% - {upper_bound:.2f}%")

                        if result == 1 and doctor_list:
                            c.setFont("Helvetica-Bold", 14)
                            c.drawString(50, height - 220, "Recommended Doctors:")
                            y = height - 240
                            c.setFont("Helvetica", 11)
                            for doctor in doctor_list:
                                c.drawString(60, y, f"Name: {doctor['name']}")
                                y -= 15
                                c.drawString(60, y, f"Specialty: {doctor['specialty']}")
                                y -= 15
                                c.drawString(60, y, f"Contact: {doctor['contact']}")
                                y -= 25
                                if y < 100:
                                    c.showPage()
                                    y = height - 50

                        c.save()
                        buffer.seek(0)
                        return buffer
                    
                    # Generate and download the PDF report
                    pdf_buffer = generate_pdf_report(
                    name=patient_name,
                    uid=patient_id,
                    probability=probability_percentage_park,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    result=result,
                    doctor_list=doctors_for_conditions['Parkinson’s disease'] if result == 1 else None
                )


                    st.download_button(
                        label="📄 Download Report Card as PDF",
                        data=pdf_buffer,
                        file_name="Parkinson_Report.pdf",
                        mime="application/pdf"
                    )


                    st.subheader("🩺 Recommended Neurologists:")
                    for doctor in doctors_for_conditions["Parkinson’s disease"]:
                        st.markdown(f"""
                        **Name:** {doctor['name']}  
                        **Specialty:** {doctor['specialty']}  
                        **Contact:** {doctor['contact']}  
                         **---------------------------------------------------**
                        """)
                else:
                    st.success(f"✅ Low risk of Parkinson's disease.\n\n"
                               f"Probability: **{probability_percentage_park:.2f}%**")
        except ValueError:
            st.error("❌ Please ensure all input fields are filled with valid numbers.")

    # Educational Section
    st.markdown("""
    ---
    ### 🧾 About Parkinson's Disease
    Parkinson's disease is a chronic and progressive movement disorder caused by the degeneration of dopamine-producing neurons in the brain.

    **🧬 Risk Factors:**
    - Increasing age (typically after 60)
    - Genetic predisposition
    - Male gender
    - Exposure to toxins (e.g., pesticides, heavy metals)

    **⚠️ Common Symptoms:**
    - Tremors (especially at rest)
    - Muscle stiffness
    - Slowed movements (bradykinesia)
    - Balance problems and falls
    - Changes in speech (soft, monotone, or slurred)

    **🩹 Management:**
    - Medications like Levodopa
    - Physical and speech therapy
    - Deep brain stimulation (in advanced cases)

    Early detection can significantly help in managing the progression and improving quality of life.
    """)


