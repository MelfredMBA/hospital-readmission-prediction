import os
import json
import pickle
import threading
import time
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from pyngrok import ngrok

app = Flask(__name__)
CORS(app)

# Determine the absolute path to the models directory (../models relative to src/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')

# Load saved artifacts
try:
    with open(os.path.join(MODELS_DIR, 'readmission_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'feature_info.json'), 'r') as f:
        feature_info = json.load(f)
    print("Model artifacts loaded successfully.")
except FileNotFoundError as e:
    print(f"Warning: Model artifacts not found ({e}). Ensure models/ directory is populated.")
    model, scaler, feature_info = None, None, None

sessions = {}

def calculate_comorbidity_score(comorbidities):
    """Calculate weighted comorbidity score (Charlson Comorbidity Index)."""
    scores = {
        # Weight 1
        'Myocardial infarction': 1,
        'Congestive heart failure': 1,
        'Peripheral vascular disease': 1,
        'Cerebrovascular disease': 1,
        'Dementia': 1,
        'Chronic pulmonary disease': 1,
        'Connective tissue disease': 1,
        'Peptic ulcer disease': 1,
        'Mild liver disease': 1,
        'Uncomplicated diabetes': 1,
        # Weight 2
        'Hemiplegia or paraplegia': 2,
        'Moderate-to-severe renal disease': 2,
        'Diabetes with complications': 2,
        'Solid localized tumor': 2,
        'Leukemia': 2,
        'Lymphoma': 2,
        # Weight 3
        'Moderate-to-severe liver disease': 3,
        # Weight 6
        'Metastatic solid tumor': 6,
        'AIDS': 6,
    }
    total = sum(scores.get(condition, 0) for condition in comorbidities)
    return min(total, 10)


def estimate_glucose_from_diabetes(diabetes_status, diabetes_stage, bmi, age):
    """Estimate glucose based on diabetes status, stage, BMI, and age."""
    base_glucose = 90

    if age > 65:
        base_glucose += (age - 65) * 0.2

    if bmi >= 30:
        base_glucose += 10
    elif bmi >= 25:
        base_glucose += 5

    if diabetes_status == "yes":
        if "Prediabetes" in diabetes_stage:
            return min(base_glucose + 20, 140)
        elif "Type 1" in diabetes_stage:
            return min(base_glucose + 40 + (bmi - 22) * 2, 200)
        elif "Type 2" in diabetes_stage:
            if bmi >= 30:
                return min(base_glucose + 60 + (bmi - 30) * 3, 250)
            else:
                return min(base_glucose + 50 + (bmi - 22) * 2, 200)
        elif "Gestational" in diabetes_stage:
            return min(base_glucose + 30, 160)
        else:
            return min(base_glucose + 40, 180)
    elif diabetes_status == "not_sure":
        return min(base_glucose + 15, 130)
    else:
        return min(base_glucose, 110)


def map_comorbidity_to_realistic_values(comorbidities, age, bmi, diabetes_status, diabetes_stage):
    """Map comorbidities to realistic num_diagnoses and num_medications values."""
    num_diagnoses = 1
    num_medications = 1

    num_diagnoses += len(comorbidities)

    if diabetes_status == "yes":
        num_diagnoses += 1
        num_medications += 1

    severe_conditions = [
        'Congestive heart failure',
        'Moderate-to-severe renal disease',
        'Moderate-to-severe liver disease',
        'Metastatic solid tumor',
        'AIDS'
    ]
    for condition in severe_conditions:
        if condition in comorbidities:
            num_medications += 2

    if age >= 65:
        num_diagnoses += 1
        num_medications += 1
    if age >= 75:
        num_diagnoses += 1
        num_medications += 1

    num_diagnoses = min(num_diagnoses, 15)
    num_medications = min(num_medications, 20)

    return {
        'num_diagnoses': num_diagnoses,
        'num_medications': num_medications
    }


def predict_readmission(features):
    """Run the trained model on the 15-element feature vector."""
    if len(features) != 15:
        print(f"Error: Expected 15 features, got {len(features)}")
        while len(features) < 15:
            features.append(0)
        features = features[:15]

    features_scaled = scaler.transform([features])
    prob = model.predict_proba(features_scaled)[0][1]
    pred = 1 if prob > 0.35 else 0
    return pred, prob


def calculate_risk_score_from_factors(age, bmi, diabetes_status, diabetes_stage,
                                       admission_count, comorbidity_score,
                                       length_of_stay, emergency_count,
                                       discharge_type, glucose):
    """Rule-based risk score from clinical factors."""
    risk_points = 0
    max_points = 100

    if age >= 75:
        risk_points += 15
    elif age >= 65:
        risk_points += 10
    elif age >= 55:
        risk_points += 5

    if bmi >= 35:
        risk_points += 15
    elif bmi >= 30:
        risk_points += 10
    elif bmi >= 25:
        risk_points += 5

    if diabetes_status == "yes":
        if "Type 2" in diabetes_stage and bmi >= 30:
            risk_points += 20
        elif "Type 2" in diabetes_stage:
            risk_points += 15
        elif "Type 1" in diabetes_stage:
            risk_points += 12
        else:
            risk_points += 15
    elif diabetes_status == "not_sure":
        risk_points += 8

    if admission_count >= 5:
        risk_points += 20
    elif admission_count >= 3:
        risk_points += 15
    elif admission_count >= 1:
        risk_points += 8

    risk_points += min(comorbidity_score * 5, 50)

    if length_of_stay >= 10:
        risk_points += 10
    elif length_of_stay >= 7:
        risk_points += 7
    elif length_of_stay >= 5:
        risk_points += 4

    if emergency_count >= 2:
        risk_points += 10
    elif emergency_count >= 1:
        risk_points += 7

    if discharge_type >= 3:
        risk_points += 5
    elif discharge_type >= 2:
        risk_points += 3
    elif discharge_type >= 1:
        risk_points += 2

    if glucose >= 250:
        risk_points += 10
    elif glucose >= 200:
        risk_points += 8
    elif glucose >= 140:
        risk_points += 5

    probability = risk_points / max_points
    scaled_probability = 0.05 + (probability * 0.85)

    return min(scaled_probability, 0.95)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/start', methods=['POST'])
def start_assessment():
    try:
        data = request.json
        session_id = data.get('session_id')

        age = data.get('age', 0)
        height_cm = data.get('height_cm', 170)
        weight_kg = data.get('weight_kg', 70)
        bmi = data.get('bmi', 24)
        bmi_category = data.get('bmi_category', 'Normal weight')

        admission_count = data.get('admission_count', 0)
        visits = data.get('visits', [])

        diabetes = data.get('diabetes', False)
        diabetes_status = data.get('diabetes_status', 'no')
        diabetes_stage = data.get('diabetes_stage', '')

        comorbidities = data.get('comorbidities', [])

        length_of_stay = 0
        emergency_count = 0
        discharge_types = []

        if visits and len(visits) > 0:
            length_of_stay = max([v.get('length_of_stay_days', 0) for v in visits])
            emergency_count = sum(1 for v in visits if v.get('admission_type') == 'Emergency/Urgent')
            discharge_types = [v.get('discharge_type', 'Standard Discharge')
                               for v in visits if v.get('discharge_type')]

        discharge_type_map = {
            'Standard Discharge': 0,
            'With Follow-up Appointment': 1,
            'Complex Discharge (Multiple Specialties)': 2,
            'Needs Home Health Care': 3,
            'Skilled Nursing Facility': 4
        }

        discharge_type = 0
        if discharge_types:
            last_discharge = discharge_types[-1] if discharge_types else 'Standard Discharge'
            discharge_type = discharge_type_map.get(last_discharge, 0)

        comorbidity_score = calculate_comorbidity_score(comorbidities)

        glucose = estimate_glucose_from_diabetes(
            diabetes_status, diabetes_stage, bmi, age
        )

        mapped_features = map_comorbidity_to_realistic_values(
            comorbidities, age, bmi, diabetes_status, diabetes_stage
        )

        has_diabetes_binary = 1 if diabetes_status == "yes" else 0
        prolonged_stay = 1 if length_of_stay >= 7 else 0
        frequent_admissions = 1 if admission_count >= 3 else 0
        high_diagnoses = 1 if mapped_features['num_diagnoses'] >= 10 else 0
        high_medications = 1 if mapped_features['num_medications'] >= 15 else 0
        high_glucose = 1 if glucose >= 200 else 0
        obese = 1 if bmi >= 30 else 0

        patient_data = {
            'age': age,
            'bmi': bmi,
            'bmi_category': bmi_category,
            'glucose': glucose,
            'diabetes_status': diabetes_status,
            'diabetes_stage': diabetes_stage,
            'discharge_type': discharge_type,
            'admission_count': admission_count,
            'length_of_stay': length_of_stay,
            'emergency_count': emergency_count,
            'comorbidity_score': comorbidity_score,
            'num_diagnoses': mapped_features['num_diagnoses'],
            'num_medications': mapped_features['num_medications'],
            'visits': visits,
            'comorbidities': comorbidities
        }

        sessions[session_id] = patient_data

        features = [
            age,
            length_of_stay,
            mapped_features['num_diagnoses'],
            mapped_features['num_medications'],
            admission_count,
            glucose,
            bmi,
            has_diabetes_binary,
            discharge_type,
            prolonged_stay,
            frequent_admissions,
            high_diagnoses,
            high_medications,
            high_glucose,
            obese
        ]

        model_pred, model_prob = predict_readmission(features)

        risk_prob = calculate_risk_score_from_factors(
            age, bmi, diabetes_status, diabetes_stage,
            admission_count, comorbidity_score,
            length_of_stay, emergency_count,
            discharge_type, glucose
        )

        MODEL_WEIGHT = 0.5
        final_prob = MODEL_WEIGHT * model_prob + (1 - MODEL_WEIGHT) * risk_prob
        risk_percent = final_prob * 100

        if risk_percent > 25:
            risk_level = 'HIGH'
            outcome = 'High Risk - Unplanned Readmission Within 30 Days'
            action = 'Intensive transitional care and close follow-up'
        elif risk_percent >= 10:
            risk_level = 'MODERATE'
            outcome = 'Moderate Risk - Monitor Closely'
            action = 'Enhanced follow-up and care coordination'
        else:
            risk_level = 'LOW'
            outcome = 'Low Risk - Standard Follow-up'
            action = 'Standard discharge protocol'

        risk_drivers = []

        if age >= 75:
            risk_drivers.append({'factor': f'Elderly patient ({age} years)', 'impact': '+20% risk', 'direction': 'increase'})
        elif age >= 65:
            risk_drivers.append({'factor': f'Senior patient ({age} years)', 'impact': '+10% risk', 'direction': 'increase'})

        if bmi >= 35:
            risk_drivers.append({'factor': f'Severe obesity (BMI: {bmi:.1f})', 'impact': '+25% risk', 'direction': 'increase'})
        elif bmi >= 30:
            risk_drivers.append({'factor': f'Obesity (BMI: {bmi:.1f})', 'impact': '+15% risk', 'direction': 'increase'})
        elif bmi >= 25:
            risk_drivers.append({'factor': f'Overweight (BMI: {bmi:.1f})', 'impact': '+5% risk', 'direction': 'increase'})

        if diabetes_status == "yes":
            if "Type 2" in diabetes_stage and bmi >= 30:
                risk_drivers.append({'factor': 'Type 2 Diabetes with obesity', 'impact': '+30% risk', 'direction': 'increase'})
            elif "Type 2" in diabetes_stage:
                risk_drivers.append({'factor': 'Type 2 Diabetes', 'impact': '+20% risk', 'direction': 'increase'})
            elif "Type 1" in diabetes_stage:
                risk_drivers.append({'factor': 'Type 1 Diabetes', 'impact': '+15% risk', 'direction': 'increase'})
            else:
                risk_drivers.append({'factor': 'Diabetes present', 'impact': '+20% risk', 'direction': 'increase'})
        elif diabetes_status == "not_sure":
            risk_drivers.append({'factor': 'Possible pre-diabetes (at risk)', 'impact': '+10% risk', 'direction': 'increase'})

        if admission_count >= 5:
            risk_drivers.append({'factor': f'Frequent admissions ({admission_count} in 12 months)', 'impact': '+30% risk', 'direction': 'increase'})
        elif admission_count >= 3:
            risk_drivers.append({'factor': f'Multiple admissions ({admission_count} in 12 months)', 'impact': '+20% risk', 'direction': 'increase'})
        elif admission_count >= 1:
            risk_drivers.append({'factor': f'Previous admissions ({admission_count})', 'impact': '+10% risk', 'direction': 'increase'})

        if comorbidity_score >= 6:
            risk_drivers.append({'factor': f'High comorbidity burden (score: {comorbidity_score})', 'impact': '+30% risk', 'direction': 'increase'})
        elif comorbidity_score >= 4:
            risk_drivers.append({'factor': f'Moderate comorbidity burden (score: {comorbidity_score})', 'impact': '+20% risk', 'direction': 'increase'})
        elif comorbidity_score >= 2:
            risk_drivers.append({'factor': f'Multiple comorbidities ({len(comorbidities)})', 'impact': '+10% risk', 'direction': 'increase'})

        if length_of_stay >= 10:
            risk_drivers.append({'factor': f'Prolonged hospitalization ({length_of_stay} days)', 'impact': '+20% risk', 'direction': 'increase'})
        elif length_of_stay >= 7:
            risk_drivers.append({'factor': f'Extended stay ({length_of_stay} days)', 'impact': '+15% risk', 'direction': 'increase'})
        elif length_of_stay >= 4:
            risk_drivers.append({'factor': f'Moderate stay ({length_of_stay} days)', 'impact': '+5% risk', 'direction': 'increase'})

        if emergency_count >= 2:
            risk_drivers.append({'factor': f'Multiple emergency admissions ({emergency_count})', 'impact': '+25% risk', 'direction': 'increase'})
        elif emergency_count >= 1:
            risk_drivers.append({'factor': 'Emergency/Urgent admission', 'impact': '+15% risk', 'direction': 'increase'})

        if discharge_type >= 3:
            risk_drivers.append({'factor': 'Complex discharge (needs home health care)', 'impact': '+15% risk', 'direction': 'increase'})
        elif discharge_type >= 2:
            risk_drivers.append({'factor': 'Complex discharge (multiple specialties)', 'impact': '+10% risk', 'direction': 'increase'})
        elif discharge_type >= 1:
            risk_drivers.append({'factor': 'Follow-up appointment scheduled', 'impact': '+5% risk', 'direction': 'increase'})

        if glucose >= 250:
            risk_drivers.append({'factor': f'Very high glucose ({glucose:.0f} mg/dL)', 'impact': '+20% risk', 'direction': 'increase'})
        elif glucose >= 200:
            risk_drivers.append({'factor': f'High glucose ({glucose:.0f} mg/dL)', 'impact': '+15% risk', 'direction': 'increase'})
        elif glucose >= 140:
            risk_drivers.append({'factor': f'Elevated glucose ({glucose:.0f} mg/dL)', 'impact': '+5% risk', 'direction': 'increase'})

        if not risk_drivers:
            risk_drivers.append({'factor': 'No major elevated risk drivers identified', 'impact': 'Low risk profile', 'direction': 'decrease'})

        risk_drivers.sort(
            key=lambda x: int(x['impact'].replace('+', '').replace('% risk', '').replace('Low risk profile', '0')),
            reverse=True
        )
        risk_drivers = risk_drivers[:5]

        response = {
            'success': True,
            'prediction': int(final_prob > 0.35),
            'predicted_outcome': outcome,
            'probability': final_prob,
            'probability_score': round(risk_percent, 1),
            'risk_level': risk_level,
            'risk_category': risk_level,
            'risk_drivers': risk_drivers,
            'patient_id': session_id,
            'recommended_action': action,
            'bmi': round(bmi, 1),
            'bmi_category': bmi_category,
            'estimated_glucose': round(glucose, 0),
            'diabetes_status': diabetes_status,
            'diabetes_stage': diabetes_stage,
            'num_diagnoses': mapped_features['num_diagnoses'],
            'num_medications': mapped_features['num_medications'],
            'comorbidity_score': comorbidity_score
        }

        return jsonify(response)

    except Exception as e:
        print(f"Error in assessment: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/summary', methods=['GET'])
def get_summary():
    try:
        session_id = request.args.get('session_id')

        if session_id not in sessions:
            return jsonify({'success': False, 'error': 'Session not found'}), 404

        patient_data = sessions[session_id]

        features = [
            patient_data['age'],
            patient_data['length_of_stay'],
            patient_data['num_diagnoses'],
            patient_data['num_medications'],
            patient_data['admission_count'],
            patient_data['glucose'],
            patient_data['bmi'],
            1 if patient_data['diabetes_status'] == "yes" else 0,
            patient_data['discharge_type'],
            1 if patient_data['length_of_stay'] >= 7 else 0,
            1 if patient_data['admission_count'] >= 3 else 0,
            1 if patient_data['num_diagnoses'] >= 10 else 0,
            1 if patient_data['num_medications'] >= 15 else 0,
            1 if patient_data['glucose'] >= 200 else 0,
            1 if patient_data['bmi'] >= 30 else 0
        ]

        model_pred, model_prob = predict_readmission(features)

        risk_prob = calculate_risk_score_from_factors(
            patient_data['age'],
            patient_data['bmi'],
            patient_data['diabetes_status'],
            patient_data['diabetes_stage'],
            patient_data['admission_count'],
            patient_data['comorbidity_score'],
            patient_data['length_of_stay'],
            patient_data['emergency_count'],
            patient_data['discharge_type'],
            patient_data['glucose']
        )

        MODEL_WEIGHT = 0.5
        final_prob = MODEL_WEIGHT * model_prob + (1 - MODEL_WEIGHT) * risk_prob
        risk_percent = final_prob * 100

        if risk_percent > 25:
            risk_level = 'HIGH'
            outcome = 'High Risk - Unplanned Readmission Within 30 Days'
        elif risk_percent >= 10:
            risk_level = 'MODERATE'
            outcome = 'Moderate Risk - Monitor Closely'
        else:
            risk_level = 'LOW'
            outcome = 'Low Risk - Standard Follow-up'

        return jsonify({
            'success': True,
            'prediction': int(final_prob > 0.35),
            'predicted_outcome': outcome,
            'probability': final_prob,
            'probability_score': round(risk_percent, 1),
            'risk_level': risk_level,
            'risk_category': risk_level,
            'patient_id': session_id
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})


def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)


if __name__ == '__main__':
    # OPTIONAL: Set your ngrok token here for Colab demos
    # If running locally, just leave it as None and access http://localhost:5000
    NGROK_AUTH_TOKEN = None  # Replace with "your_token" if using ngrok

    if NGROK_AUTH_TOKEN:
        print("Starting ngrok tunnel...")
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)
        public_url = ngrok.connect(5000)
        print(f"Public URL: {public_url}")

        thread = threading.Thread(target=run_flask)
        thread.daemon = True
        thread.start()

        time.sleep(3)
        print(f"\nOPEN THIS URL IN YOUR BROWSER: {public_url}")
        print(f"Health check: {public_url}/health")
        print("\nPress Ctrl+C to stop")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nServer stopped.")
    else:
        # Local run
        print("Starting Flask app on http://localhost:5000")
        run_flask()
