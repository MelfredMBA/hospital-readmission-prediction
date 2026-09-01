# 14. Comorbidity Scoring according to Charlson Comorbidity Index.

def calculate_comorbidity_score(comorbidities):
    """Calculate weighted comorbidity score (Charlson-like)"""
    scores = {
        'Myocardial infarction': 1,
        'Congestive heart failure': 2,
        'Peripheral vascular disease': 1,
        'Cerebrovascular disease': 1,
        'Dementia': 1,
        'Chronic pulmonary disease': 1,
        'Connective tissue disease': 1,
        'Peptic ulcer disease': 1,
        'Mild liver disease': 1,
        'Uncomplicated diabetes': 1,
        'Hemiplegia or paraplegia': 2,
        'Moderate-to-severe renal disease': 2,
        'Solid localized tumor': 2,
        'Leukemia': 2,
        'Lymphoma': 2,
        'Moderate-to-severe liver disease': 3,
        'Metastatic solid tumor': 4,
        'AIDS': 4
    }
    total = sum(scores.get(condition, 0) for condition in comorbidities)
    return min(total, 10)

# 15. Use diabetes information to assume glucose level.
# The assumption is done because not every user would know their glucose level.
# In general public, they know their diabetes type and the presence of it compared to knowing what their glucose level is.

def estimate_glucose_from_diabetes(diabetes_status, diabetes_stage, bmi, age):
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

  # 16. Converting features into values understandable by the code.

def map_comorbidity_to_realistic_values(comorbidities, age, bmi, diabetes_status, diabetes_stage):
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

# 17. Checks if it uses the features to determine the output.
# If no error was found, it would print blank.

def predict_readmission(features):
    if len(features) != 15:
        print(f"Error: Expected 15 features, got {len(features)}")
        while len(features) < 15:
            features.append(0)
        features = features[:15]

    features_scaled = scaler.transform([features])
    prob = model.predict_proba(features_scaled)[0][1]
    pred = 1 if prob > 0.35 else 0
    return pred, prob

# 18. Actual risk calculation according to collected data.

def calculate_risk_score_from_factors(age, bmi, diabetes_status, diabetes_stage,
                                       admission_count, comorbidity_score,
                                       length_of_stay, emergency_count,
                                       discharge_type, glucose):
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

    if comorbidity_score >= 6:
        risk_points += 15
    elif comorbidity_score >= 4:
        risk_points += 10
    elif comorbidity_score >= 2:
        risk_points += 5

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
