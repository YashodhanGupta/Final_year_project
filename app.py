import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import scipy.stats as stats
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.header('Diabetes Prediction using ML')
    
    #st.image("path_to_diabetes_image.png", use_column_width=True)

    with st.expander("Input Parameters"):
        col1, col2, col3 = st.columns(3)
        with col1:
            Pregnancies = st.text_input('Number of Pregnancies', help="The number of times you have been pregnant. For males, this value is 0.")
        with col2:
            Glucose = st.text_input('Glucose Level (mg/dL)', help="Enter your glucose level in milligrams per deciliter (mg/dL). Values above 200 mg/dL may indicate diabetes.")
        with col3:
            BloodPressure = st.text_input('Blood Pressure (mmHg)', help="Enter your diastolic blood pressure (mmHg). Normal levels are typically around 80 mmHg.")

        with col1:
            SkinThickness = st.text_input('Skin Thickness value',help="Enter the thickness of skinfold (in mm). This is measured at the triceps and is an indicator of body fat.")
    
        with col2:
            Insulin = st.text_input('Insulin Level',help="Enter your insulin level in micro units per milliliter (µU/mL). High levels may suggest insulin resistance.")
    
        with col3:
            BMI = st.text_input('BMI value',help="Enter your Body Mass Index (BMI). It is calculated as weight (kg) / height (m)^2. Normal BMI is between 18.5 and 24.9.")
    
        with col1:
            DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value',help="Enter your Diabetes Pedigree Function value. This indicates the likelihood of diabetes based on family history.")
    
        with col2:
            Age = st.text_input('Age of the Person',help="Enter your age in years. The risk of diabetes increases with age, particularly after age 45.")


    # code for Prediction
    diab_diagnosis = ''
    
    threshold = 0.5

    # creating a button for Prediction
    
    if st.button('🚨 Check Diabetes Test Result'):
        try:
            user_input = [float(Pregnancies), float(Glucose), float(BloodPressure),float(SkinThickness),float(Insulin),float(BMI),float(DiabetesPedigreeFunction),float(Age)]  # Extend for all inputs
            raw_score = diabetes_model.decision_function([user_input])[0]  # Raw score

# Convert to probability using sigmoid function
            prob_diabetic = sigmoid(raw_score)
            probability_percentage = prob_diabetic * 100
            threshold = 0.7

            # Confidence interval calculation (Assuming normal distribution)
        # You need a std_dev from your training data decision function scores
            std_dev = 0.15  # Example value; calculate from training scores
            confidence_level = 0.95  # 95% confidence interval
            margin_of_error = stats.norm.ppf((1 + confidence_level) / 2) * std_dev

            lower_bound = max(0, (prob_diabetic - margin_of_error) * 100)
            upper_bound = min(100, (prob_diabetic + margin_of_error) * 100)


            with st.spinner('Calculating result...'):
                
                diab_prediction = diabetes_model.predict([user_input])
                if diab_prediction[0] == 1:
                    st.error(f'🚨 The person is diabetic with a probability of {probability_percentage:.2f}% '
                         f' (Confidence Range: {lower_bound:.2f}% - {upper_bound:.2f}%)')
                    st.subheader("Doctors to Consult for Diabetes:")
                    for doctor in doctors_for_conditions['diabetes']:
                        st.markdown(f"""
                            **Name:** {doctor['name']}
                            
                            **Specialty:** {doctor['specialty']}
                            
                            **Contact:** {doctor['contact']}
                            
                            **---------------------------------------------------**
                        """)
                            
                else:
                    st.success(f'✅ The person is not diabetic with a probability of {probability_percentage:.2f}%')
                    
               

        except ValueError:
            st.error("Please enter all the input fields.")
            
    st.markdown("""
    ### About Diabetes:
    Diabetes is a chronic health condition that affects how your body turns food into energy. 
    It occurs when your blood glucose (sugar) levels are too high. There are three main types:
    - **Type 1 Diabetes:** The body doesn’t produce insulin.
    - **Type 2 Diabetes:** The body doesn’t use insulin properly.
    - **Gestational Diabetes:** Occurs during pregnancy.

    **Risk Factors:**
    - Being overweight or obese
    - Lack of physical activity
    - Family history of diabetes

    **Symptoms:**
    - Increased thirst and urination
    - Fatigue
    - Blurred vision

    Early detection and management can help prevent complications such as heart disease, kidney damage, and nerve problems.
    """)

    

    st.success(diab_diagnosis)

# Heart Disease Prediction Page
# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.header('Heart Disease Prediction using ML')
    # st.image("path_to_heart_image.png", use_column_width=True)

    with st.expander("Input Parameters"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.text_input('Age',help="Enter the age of the individual in years. Risk of heart disease increases with age.")
    
        with col2:
            sex = st.text_input('Sex',help="Enter the sex of the individual. Male is typically coded as 1 and female as 0.")
    
        with col3:
            cp = st.text_input('Chest Pain types',help="Enter the type of chest pain experienced: 0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic.")
    
        with col1:
            trestbps = st.text_input('Resting Blood Pressure',help="Enter the resting blood pressure in mmHg. Normal levels are typically below 120/80 mmHg.")
    
        with col2:
            chol = st.text_input('Serum Cholestoral in mg/dl',help="Enter the serum cholesterol level in milligrams per deciliter (mg/dL). A normal level is below 200 mg/dL.")
    
        with col3:
            fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl',help="Enter 1 if fasting blood sugar > 120 mg/dL, otherwise enter 0. High fasting blood sugar is a risk factor for heart disease.")
    
        with col1:
            restecg = st.text_input('Resting Electrocardiographic results',help="Enter the resting ECG results: 0 = normal, 1 = having ST-T wave abnormality, 2 = showing left ventricular hypertrophy.")
    
        with col2:
            thalach = st.text_input('Maximum Heart Rate achieved',help="Enter the maximum heart rate achieved during exercise. A higher heart rate during exercise is generally considered better.")
    
        with col3:
            exang = st.text_input('Exercise Induced Angina',help="Enter 1 if exercise induced angina is present, otherwise enter 0. Angina during exercise indicates higher heart disease risk.")
    
        with col1:
            oldpeak = st.text_input('ST depression induced by exercise',help="Enter the ST depression induced by exercise. A higher value suggests more significant heart disease.")
    
        with col2:
            slope = st.text_input('Slope of the peak exercise ST segment',help="Enter the slope of the peak exercise ST segment: 0 = upsloping, 1 = flat, 2 = downsloping.")
    
        with col3:
            ca = st.text_input('Major vessels colored by flourosopy',help="Enter the number of major vessels colored by fluoroscopy: 0, 1, 2, or 3. More vessels suggest a higher risk of heart disease.")
    
        with col1:
            thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect',help="Enter the thalassemia value: 0 for normal, 1 for fixed defect, and 2 for reversible defect.")

    # code for Prediction
   # heart_diagnosis = ''

    # creating a button for Prediction

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    if st.button('Heart Disease Test Result'):
        try:

            user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
            user_input = [float(x) for x in user_input]
            raw_score = heart_disease_model.decision_function([user_input])[0]  # Raw score

            prob_heart = sigmoid(raw_score)
            probability_percentage_heart = prob_heart * 100

            std_dev = 0.15  # Example value; calculate from training scores
            confidence_level = 0.95  # 95% confidence interval
            margin_of_error = stats.norm.ppf((1 + confidence_level) / 2) * std_dev

            lower_bound = max(0, (prob_heart - margin_of_error) * 100)
            upper_bound = min(100, (prob_heart + margin_of_error) * 100)

    
            with st.spinner('Calculating result...'):
                    heart_prediction = heart_disease_model.predict([user_input])  # Replace with model prediction logic
                    if heart_prediction == 1:
                        st.error(f'🚨 High risk of heart disease detected! {probability_percentage_heart:.2f}%'
                                 f' (Confidence Range: {lower_bound:.2f}% - {upper_bound:.2f}%)')
                        st.subheader("Doctors to Consult for Heart Diasease:")
                        for doctor in doctors_for_conditions['heart disease']:
                            st.markdown(f"""
                                **Name:** {doctor['name']}
                                
                                **Specialty:** {doctor['specialty']}
                                
                                **Contact:** {doctor['contact']}
                                
                                **---------------------------------------------------**
                            """)    
                    else:
                        st.success(f'❤️ No significant risk of heart disease detected. {probability_percentage_heart:.2f}%')
        except ValueError:
            st.error("Please enter all the input fields.")

    
    st.markdown("""
    ### About Heart Disease:
    Heart disease refers to various types of heart conditions, including coronary artery disease, arrhythmias, and heart failure. 
    It is one of the leading causes of death worldwide.

    **Risk Factors:**
    - High blood pressure
    - High cholesterol
    - Smoking
    - Diabetes
    - Obesity

    **Symptoms:**
    - Chest pain or discomfort
    - Shortness of breath
    - Fatigue
    - Irregular heartbeat

    Lifestyle changes, medications, and medical procedures can help manage and prevent heart disease.
    """)

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":
    st.header('Parkinson Disease Prediction using ML')
    
    # page title
    
    with st.expander("Input Parameters"):
        col1, col2, col3, col4, col5 = st.columns(5)
    
        with col1:
            fo = st.text_input('MDVP: Foa(Hz)',help="Fundamental frequency of the voice in Hertz. It represents the average pitch of the voice.")
    
        with col2:
            fhi = st.text_input('MDVP: Fhi(Hz)',help="Maximum fundamental frequency in Hertz. It indicates the highest pitch of the voice.")
    
        with col3:
            flo = st.text_input('MDVP: Flo(Hz)',help="Minimum fundamental frequency in Hertz. It represents the lowest pitch of the voice.")
    
        with col4:
            Jitter_percent = st.text_input('MDVP: Jitter(%)',help="Variation in pitch as a percentage. Higher values indicate more irregularity in pitch.")
    
        with col5:
            Jitter_Abs = st.text_input('MDVP: Jitter(Abs)',help="Absolute jitter in voice frequency. Indicates pitch variation in absolute terms.")
    
        with col1:
            RAP = st.text_input('MDVP: RAP',help="Relative Average Perturbation. It measures short-term variation in pitch.")
    
        with col2:
            PPQ = st.text_input('MDVP: PPQ',help="Pitch Period Perturbation Quotient. It quantifies pitch instability over a longer period.")
    
        with col3:
            DDP = st.text_input('Jitter: DDP',help="Difference of Differences of Pitch Periods. Indicates combined jitter values.")
    
        with col4:
            Shimmer = st.text_input('MDVP: Shimmer',help="Amplitude perturbation as a percentage. Measures the variation in voice amplitude.")
    
        with col5:
            Shimmer_dB = st.text_input('MDVP: Shimmer(dB)',help="Shimmer value in decibels. Indicates the degree of amplitude variation in the voice.")
    
        with col1:
            APQ3 = st.text_input('Shimmer: APQ3',help="Amplitude Perturbation Quotient over three periods. Measures short-term amplitude instability.")
    
        with col2:
            APQ5 = st.text_input('Shimmer: APQ5',help="Amplitude Perturbation Quotient over five periods. Indicates amplitude variation over a longer period.")
    
        with col3:
            APQ = st.text_input('MDVP: APQ',help="Amplitude Perturbation Quotient. Represents overall amplitude variation in the voice.")
    
        with col4:
            DDA = st.text_input('Shimmer: DDA',help="Difference of Differences of Amplitude. Indicates combined shimmer values.")
    
        with col5:
            NHR = st.text_input('NHR',help="Noise-to-Harmonics Ratio. Higher values suggest more noise in the voice signal.")
    
        with col1:
            HNR = st.text_input('HNR',help="Harmonics-to-Noise Ratio. Higher values indicate a cleaner, more harmonic voice signal.")
    
        with col2:
            RPDE = st.text_input('RPDE',help="Recurrence Period Density Entropy. Measures the complexity of the voice signal.")
    
        with col3:
            DFA = st.text_input('DFA',help="Detrended Fluctuation Analysis. Quantifies the self-similarity of the voice signal over time.")
    
        with col4:
            spread1 = st.text_input('spread1',help="Nonlinear measure related to voice frequency distribution. Indicates voice signal spread.")
    
        with col5:
            spread2 = st.text_input('spread2',help="Another nonlinear measure related to voice frequency distribution. Indicates additional spread details.")
    
        with col1:
            D2 = st.text_input('D2',help="Dynamic complexity of the voice signal. Higher values indicate greater variability in dynamics.")
    
        with col2:
            PPE = st.text_input('PPE',help="Pitch Period Entropy. Quantifies the overall variation in pitch over time.")

    # code for Prediction
    parkinsons_diagnosis = ''

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        try:

            user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                          RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5,
                          APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]
    
            user_input = [float(x) for x in user_input]

            raw_score = parkinsons_model.decision_function([user_input])[0]  # Raw score

            prob_park = sigmoid(raw_score)
            probability_percentage_park = prob_park * 100

            std_dev = 0.15  # Example value; calculate from training scores
            confidence_level = 0.95  # 95% confidence interval
            margin_of_error = stats.norm.ppf((1 + confidence_level) / 2) * std_dev

            lower_bound = max(0, (prob_park - margin_of_error) * 100)
            upper_bound = min(100, (prob_park + margin_of_error) * 100)

            with st.spinner('Calculating result...'):
                heart_prediction = parkinsons_model.predict([user_input])  # Replace with model prediction logic
                if heart_prediction == 1:
                    st.error(f'🚨 High risk of parkinson disease detected! {probability_percentage_park:.2f}%'
                             f' (Confidence Range: {lower_bound:.2f}% - {upper_bound:.2f}%)')
                    for doctor in doctors_for_conditions['Parkinson’s disease']:
                        st.markdown(f"""
                            **Name:** {doctor['name']}
                            
                            **Specialty:** {doctor['specialty']}
                            
                            **Contact:** {doctor['contact']}
                            
                            **---------------------------------------------------**
                        """)
                else:
                    st.success(f'❤️ No significant risk of parkinson disease detected. {probability_percentage_park:.2f}%')
        except ValueError:
            st.error("Please enter all the input fields.")
    
    st.markdown("""
    ### About Parkinson's Disease:
    Parkinson's disease is a neurodegenerative disorder that affects movement. It occurs when nerve cells in the brain that produce dopamine die, leading to symptoms like tremors, stiffness, and difficulty with balance and coordination. 

    **Risk Factors:**
    - Age (usually develops after age 60)
    - Genetics (family history)
    - Gender (men are more likely to develop it)
    - Exposure to toxins

    **Symptoms:**
    - Tremors (shaking of the hands, fingers, or other parts of the body)
    - Muscle rigidity (stiffness)
    - Bradykinesia (slowness of movement)
    - Postural instability (difficulty with balance)
    - Speech changes (soft or slurred speech)

    While there is no cure for Parkinson's disease, treatments such as medication, therapy, and surgery can help manage the symptoms and improve quality of life.
    """)
