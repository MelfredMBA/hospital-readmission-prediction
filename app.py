# app.py
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import pickle
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Load saved artifacts
with open('models/readmission_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('models/feature_info.json', 'r') as f:
    feature_info = json.load(f)


sessions = {}


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Will AI be Readmitted?</title>
  <link rel="stylesheet" href="https://www.w3schools.com/w3css/5/w3.css">
  <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Oswald">
  <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Open+Sans">

  <style>
    :root {
      --dark-purple:#35104f;
      --purple:#7043a1;
      --light-purple:#eadcf8;
    }

    * { box-sizing:border-box; }

    body {
      margin:0;
      background:#f7f2fb;
      color:#30213d;
      font-family:"Open Sans",sans-serif;
    }

    h1,h2,h3,h4,h5,h6 { font-family:"Oswald",sans-serif; }

    .site-header {
      background:white;
      text-align:center;
      padding:40px 20px;
    }

    .site-header h1 {
      margin:0;
      font-size:42px;
    }

    .gradient-text {
      background:linear-gradient(90deg,#c084fc,#8b5cf6,#35104f);
      -webkit-background-clip:text;
      background-clip:text;
      color:transparent;
    }

    .w3-bar,
    .site-footer {
      background:var(--dark-purple);
      color:white;
    }

    .w3-bar { height:40px; }

    .site-footer {
      padding:28px;
      margin-top:30px;
    }

    .w3-sand {
      background:linear-gradient(180deg,#fcf9ff,#eadcf8)!important;
    }

    .w3-tag {
      background:linear-gradient(90deg,#c084fc,#8b5cf6,#35104f)!important;
      color:white!important;
    }

    .unit-choice {
      display:flex;
      gap:18px;
      flex-wrap:wrap;
      margin:5px 0 10px;
    }

    .unit-choice label {
      cursor:pointer;
    }

    .unit-choice input {
      margin-right:6px;
    }

    .input-row {
      display:grid;
      grid-template-columns:1fr 1fr;
      gap:10px;
    }

    .diabetes-stages {
      display:none;
      background:#f5edfc;
      border-left:4px solid var(--purple);
      padding:10px 15px;
      margin:8px 0 15px 25px;
      border-radius:5px;
    }

    .diabetes-stages.visible { display:block; }

    .comorbidity-category {
      background:#fcf9ff;
      border-left:4px solid #a855f7;
      padding:12px 15px;
      margin:12px 0;
      border-radius:5px;
    }

    .comorbidity-option {
      display:block;
      margin:8px 0;
    }

    .visit-table {
      width:100%;
      border-collapse:collapse;
      margin:15px 0;
      background:white;
    }

    .visit-table th,
    .visit-table td {
      border:1px solid #c4a7e7;
      padding:10px;
      text-align:center;
      vertical-align:middle;
    }

    .visit-table th { background:var(--light-purple); }

    .visit-table input,
    .visit-table select {
      width:100%;
      padding:8px;
      border:1px solid #c4a7e7;
      border-radius:4px;
    }

    .result-section {
      background:white;
      border-radius:8px;
      padding:18px;
      margin:14px 0;
      box-shadow:0 2px 8px rgba(112,67,161,.1);
    }

    .result-section h4 {
      color:var(--dark-purple);
      margin-top:0;
    }

    .risk-card {
      color:white;
      text-align:center;
      border-radius:10px;
      padding:24px 15px;
      margin-bottom:18px;
    }

    .risk-card.low { background:#388e3c; }
    .risk-card.moderate { background:#f9a825; color:#332400; }
    .risk-card.high { background:#c62828; }

    .risk-level {
      font-size:30px;
      font-weight:bold;
    }

    .risk-driver,
    .next-step,
    .summary-item {
      background:#f8f2fd;
      padding:10px 12px;
      margin:7px 0;
      border-radius:5px;
    }

    .risk-driver { border-left:4px solid #c62828; }
    .risk-driver.decrease { border-left-color:#388e3c; }
    .next-step { border-left:4px solid #388e3c; }

    .summary-grid {
      display:grid;
      grid-template-columns:repeat(2,minmax(0,1fr));
      gap:10px;
    }

    .summary-item strong {
      display:block;
      color:var(--purple);
    }

    @media (max-width:700px) {
      .site-header h1 { font-size:32px; }
      .input-row,
      .summary-grid { grid-template-columns:1fr; }
    }
  </style>
</head>

<body>
  <div class="w3-bar"></div>

  <header class="site-header">
    <h1><b><span class="gradient-text">Will AI be Readmitted?</span></b></h1>
    <h6>
      AI-assisted hospital readmission assessment made
      <span class="w3-tag">Easier</span>
    </h6>
  </header>

  <main class="w3-sand w3-large">
    <div class="w3-container" id="about">
      <div class="w3-content" style="max-width:700px">
        <h5 class="w3-center w3-padding-64">
          <span class="w3-tag w3-wide">ENTER YOUR DETAILS</span>
        </h5>

        <div id="setupSection">
          <p>
            <label for="ageInput">Age:</label>
            <input class="w3-input w3-padding-16 w3-border"
                   type="number" id="ageInput" min="0" max="120" step="1"
                   placeholder="Enter your age">
          </p>

          <p>
            <strong>Height unit:</strong>
            <span class="unit-choice">
              <label>
                <input type="radio" name="heightUnit" value="cm"
                       checked onchange="updateHeightUnit()">
                Centimeters (cm)
              </label>
              <label>
                <input type="radio" name="heightUnit" value="ft"
                       onchange="updateHeightUnit()">
                Feet
              </label>
            </span>

            <label id="heightLabel" for="heightInput">Height in centimeters:</label>
            <input class="w3-input w3-padding-16 w3-border"
                   type="number" id="heightInput" min="1" max="300"
                   step="0.1" placeholder="Example: 170">
          </p>

          <p>
            <strong>Weight unit:</strong>
            <span class="unit-choice">
              <label>
                <input type="radio" name="weightUnit" value="kg"
                       checked onchange="updateWeightUnit()">
                Kilograms (kg)
              </label>
              <label>
                <input type="radio" name="weightUnit" value="lb"
                       onchange="updateWeightUnit()">
                Pounds (lb)
              </label>
            </span>

            <label id="weightLabel" for="weightInput">Weight in kilograms:</label>
            <input class="w3-input w3-padding-16 w3-border"
                   type="number" id="weightInput" min="1" max="500"
                   step="0.1" placeholder="Example: 70">
          </p>

          <p>
            <label for="admissionCountInput">
              How many times have you been admitted in the past 12 months?
            </label>
            <input class="w3-input w3-padding-16 w3-border"
                   type="number" id="admissionCountInput" min="0" max="12"
                   step="1" placeholder="Enter a number from 0 to 12"
                   oninput="createVisitTable()">
          </p>

          <div id="visitTableContainer"></div>

          <p>
            <strong>Do you have Diabetes?</strong>
            <label class="comorbidity-option">
              <input type="checkbox" id="diabetesYes"
                     onchange="selectDiabetesStatus('yes')">
              Yes
            </label>
            <label class="comorbidity-option">
              <input type="checkbox" id="diabetesNo"
                     onchange="selectDiabetesStatus('no')">
              No
            </label>
            <label class="comorbidity-option">
              <input type="checkbox" id="diabetesNotSure"
                     onchange="selectDiabetesStatus('not_sure')">
              Not sure
            </label>
          </p>

          <div id="diabetesStages" class="diabetes-stages">
            <strong>Select one diabetes stage/type:</strong>
            <label class="comorbidity-option">
              <input type="radio" name="diabetesStage" value="Prediabetes">
              Prediabetes
            </label>
            <label class="comorbidity-option">
              <input type="radio" name="diabetesStage" value="Type 1 Diabetes">
              Type 1 Diabetes
            </label>
            <label class="comorbidity-option">
              <input type="radio" name="diabetesStage" value="Type 2 Diabetes">
              Type 2 Diabetes
            </label>
            <label class="comorbidity-option">
              <input type="radio" name="diabetesStage" value="Gestational Diabetes">
              Gestational Diabetes
            </label>
          </div>

          <p>
            <strong>Do you have any existing comorbidities?</strong><br>
            <small>Check all that apply. Diabetes is recorded separately above.</small>
          </p>

          <div class="comorbidity-category">
            <h6>Category 1</h6>
            <label class="comorbidity-option"><input type="checkbox" name="comorbidity" value="Myocardial infarction"> Myocardial infarction</label>
            <label class="comorbidity-option"><input type="checkbox" name="comorbidity" value="Congestive heart failure"> Congestive heart failure</label>
            <label class="comorbidity-option"><input type="checkbox" name="comorbidity" value="Peripheral vascular disease"> Peripheral vascular disease</label>
            <label class="comorbidity-option"><input type="checkbox" name="comorbidity" value="Cerebrovascular disease"> Cerebrovascular disease</label>
            <label class="comorbidity-option"><input type="checkbox" name="comorbidity" value="Dementia"> Dementia</label>
            <label class="comorbidity-option"><input type="checkbox" name="comorbidity" value="Chronic pulmonary disease"> Chronic pulmonary disease</label>
            <label class="comorbidity-option"><input type="checkbox" name="comorbidity" value="Connective tissue disease"> Connective tissue disease</label>
            <label class="comorbidity-option"><input type="checkbox" name="comorbidity" value="Peptic ulcer disease"> Peptic ulcer disease</label>
            <label class="comorbidity-option"><input type="checkbox" name="comorbidity" value="Mild liver disease"> Mild liver disease</label>
            <label class="comorbidity-option"><input type="checkbox" name="comorbidity" value="Uncomplicated diabetes"> Uncomplicated diabetes</label>
          </div>

          <div class="comorbidity-category">
            <h6>Category 2</h6>
            <label class="comorbidity-option"><input type="checkbox" name="comorbidity" value="Hemiplegia or paraplegia"> Hemiplegia or paraplegia</label>
            <label class="comorbidity-option"><input type="checkbox" name="comorbidity" value="Moderate-to-severe renal disease"> Moderate-to-severe renal disease</label>
            <label class="comorbidity-option"><input type="checkbox" name="comorbidity" value="Solid localized tumor"> Solid localized tumor</label>
            <label class="comorbidity-option"><input type="checkbox" name="comorbidity" value="Leukemia"> Leukemia</label>
            <label class="comorbidity-option"><input type="checkbox" name="comorbidity" value="Lymphoma"> Lymphoma</label>
          </div>

          <div class="comorbidity-category">
            <h6>Category 3</h6>
            <label class="comorbidity-option"><input type="checkbox" name="comorbidity" value="Moderate-to-severe liver disease"> Moderate-to-severe liver disease</label>
          </div>

          <div class="comorbidity-category">
            <h6>Category 4</h6>
            <label class="comorbidity-option"><input type="checkbox" name="comorbidity" value="Metastatic solid tumor"> Metastatic solid tumor</label>
            <label class="comorbidity-option"><input type="checkbox" name="comorbidity" value="AIDS"> AIDS</label>
          </div>

          <button class="w3-button w3-blue" id="startBtn"
                  onclick="startAssessment()" style="width:100%">
            Start Processing
          </button>

          <div id="setupMessage" class="w3-panel"></div>
        </div>
      </div>
    </div>

    <div class="w3-container" id="menu">
      <div class="w3-content" style="max-width:900px">
        <h5 class="w3-center w3-padding-48">
          <span class="w3-tag w3-wide">Results</span>
        </h5>

        <div class="w3-container w3-padding-48 w3-card">
          <h4 class="w3-center">Readmission Risk Assessment</h4>
          <div id="resultsContainer">
            <div class="result-section w3-center w3-text-grey">
              Complete the patient details to view the assessment.
            </div>
          </div>

          <button id="summaryBtn" class="w3-button w3-black"
                  onclick="startNewAssessment()"
                  style="display:none;width:100%">
            Start New Assessment
          </button>
        </div>
      </div>
    </div>
  </main>

  <footer class="site-footer">
    <p>Will AI be Readmitted? — AI-Assisted Hospital Readmission Assessment</p>
    <p>Created by: Melfred P. Sumaya, PTRP, MBA</p>
  </footer>

  <script>
    const API_BASE_URL = window.location.origin;
    let sessionId = localStorage.getItem("interviewSessionId");
    let patientDetails = {};

    function createSession() {
      sessionId = "session_" + Date.now() + "_" +
        Math.random().toString(36).substring(2, 11);
      localStorage.setItem("interviewSessionId", sessionId);
    }

    if (!sessionId) createSession();

    function getSelectedUnit(name) {
      return document.querySelector(`input[name="${name}"]:checked`).value;
    }

    function updateHeightUnit() {
      const unit = getSelectedUnit("heightUnit");
      const label = document.getElementById("heightLabel");
      const input = document.getElementById("heightInput");

      if (unit === "ft") {
        label.textContent = "Height in feet:";
        input.max = "10";
        input.step = "0.01";
        input.placeholder = "Example: 5.58";
      } else {
        label.textContent = "Height in centimeters:";
        input.max = "300";
        input.step = "0.1";
        input.placeholder = "Example: 170";
      }
    }

    function updateWeightUnit() {
      const unit = getSelectedUnit("weightUnit");
      const label = document.getElementById("weightLabel");
      const input = document.getElementById("weightInput");

      if (unit === "lb") {
        label.textContent = "Weight in pounds:";
        input.max = "1100";
        input.step = "0.1";
        input.placeholder = "Example: 154";
      } else {
        label.textContent = "Weight in kilograms:";
        input.max = "500";
        input.step = "0.1";
        input.placeholder = "Example: 70";
      }
    }

    function getMetricMeasurements() {
      const valueHeight = Number(document.getElementById("heightInput").value);
      const valueWeight = Number(document.getElementById("weightInput").value);
      const heightUnit = getSelectedUnit("heightUnit");
      const weightUnit = getSelectedUnit("weightUnit");

      return {
        enteredHeight: valueHeight,
        enteredWeight: valueWeight,
        heightUnit,
        weightUnit,
        heightCm: heightUnit === "ft" ? valueHeight * 30.48 : valueHeight,
        weightKg: weightUnit === "lb" ? valueWeight * 0.45359237 : valueWeight
      };
    }

    function selectDiabetesStatus(status) {
      const checkboxes = {
        yes: document.getElementById("diabetesYes"),
        no: document.getElementById("diabetesNo"),
        not_sure: document.getElementById("diabetesNotSure")
      };

      Object.keys(checkboxes).forEach(key => {
        checkboxes[key].checked = key === status;
      });

      const stages = document.getElementById("diabetesStages");
      const stageInputs = document.querySelectorAll(
        "input[name='diabetesStage']"
      );

      if (status === "yes") {
        stages.classList.add("visible");
        stageInputs.forEach(input => {
          input.disabled = false;
          if (input.value === "Prediabetes") {
            input.checked = false;
          }
        });
      } else if (status === "not_sure") {
        stages.classList.remove("visible");
        stageInputs.forEach(input => {
          input.disabled = true;
          input.checked = input.value === "Prediabetes";
        });
      } else {
        stages.classList.remove("visible");
        stageInputs.forEach(input => {
          input.disabled = false;
          input.checked = false;
        });
      }
    }

    function getDiabetesStatus() {
      if (document.getElementById("diabetesYes").checked) return "yes";
      if (document.getElementById("diabetesNotSure").checked) return "not_sure";
      if (document.getElementById("diabetesNo").checked) return "no";
      return "";
    }

    function createVisitTable() {
      const count = Number(document.getElementById("admissionCountInput").value);
      const container = document.getElementById("visitTableContainer");

      if (!Number.isInteger(count) || count < 1) {
        container.innerHTML = "";
        return;
      }

      let rows = "";

      for (let i = 1; i <= Math.min(count, 12); i++) {
        rows += `
          <tr>
            <td><strong>Admission ${i}</strong></td>
            <td>
              <input type="number" name="stay_${i}" min="1"
                     step="1" placeholder="Days">
            </td>
            <td>
              <select name="type_${i}">
                <option value="">Choose</option>
                <option value="Emergency/Urgent">Emergency/Urgent</option>
                <option value="Elective">Elective</option>
              </select>
            </td>
            <td>
              <select name="discharge_${i}">
                <option value="">Choose</option>
                <option value="Standard Discharge">Standard Discharge</option>
                <option value="With Follow-up Appointment">With Follow-up Appointment</option>
                <option value="Complex Discharge (Multiple Specialties)">Complex Discharge (Multiple Specialties)</option>
                <option value="Needs Home Health Care">Needs Home Health Care</option>
                <option value="Skilled Nursing Facility">Skilled Nursing Facility</option>
              </select>
            </td>
          </tr>`;
      }

      container.innerHTML = `
        <h6><strong>Previous admission details</strong></h6>
        <div style="overflow-x:auto">
          <table class="visit-table">
            <thead>
              <tr>
                <th>Admission</th>
                <th>Length of stay</th>
                <th>Admission type</th>
                <th>Discharge type</th>
              </tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>
        </div>`;
    }

    function getVisitDetails() {
      const count = Number(document.getElementById("admissionCountInput").value);
      const visits = [];

      for (let i = 1; i <= count; i++) {
        visits.push({
          visit:i,
          length_of_stay_days:Number(
            document.querySelector(`[name="stay_${i}"]`)?.value || 0
          ),
          admission_type:
            document.querySelector(`[name="type_${i}"]`)?.value || "",
          discharge_type:
            document.querySelector(`[name="discharge_${i}"]`)?.value || ""
        });
      }

      return visits;
    }

    function calculateBMI(heightCm, weightKg) {
      const heightMeters = heightCm / 100;
      return weightKg / (heightMeters * heightMeters);
    }

    function getBMICategory(bmi) {
      if (bmi < 18.5) return "Underweight";
      if (bmi < 25) return "Normal weight";
      if (bmi < 30) return "Overweight";
      return "Obesity";
    }

    function showMessage(message, error = false) {
      const element = document.getElementById("setupMessage");
      element.innerHTML =
        `<div class="w3-panel ${error ? "w3-pale-red" : "w3-pale-green"}">
          ${message}
        </div>`;
    }

    function normalizeRisk(data) {
      const probability = Number(
        data.probability ??
        data.probability_score ??
        data.readmission_probability ??
        0
      );

      const percent = probability <= 1 ? probability * 100 : probability;
      let risk = String(
        data.risk_level || data.risk_category || ""
      ).toUpperCase();

      if (!["LOW", "MODERATE", "HIGH"].includes(risk)) {
        risk = percent > 25 ? "HIGH" : percent >= 10 ? "MODERATE" : "LOW";
      }

      return {
        risk,
        percent:Math.max(0, Math.min(100, percent))
      };
    }

    function renderRiskResult(data) {
      const normalized = normalizeRisk(data);
      const risk = normalized.risk;
      const outcome = data.predicted_outcome ||
        (risk === "HIGH"
          ? "Unplanned Readmission Within 30 Days"
          : "No Readmission Predicted");

      const visits = patientDetails.visits;
      const admissionTypes = [...new Set(
        visits.map(visit => visit.admission_type)
      )].join(", ") || "None";
      const dischargeTypes = [...new Set(
        visits.map(visit => visit.discharge_type)
      )].join(", ") || "None";

      const diabetesSummary = patientDetails.diabetesStatus === "yes"
        ? `Yes — ${patientDetails.diabetesStage}`
        : patientDetails.diabetesStatus === "not_sure"
          ? "Not sure — scored as Prediabetes"
          : patientDetails.diabetesStatus === "no"
            ? "No"
            : "Not answered";

      let riskDriversHtml = '';
      if (data.risk_drivers && data.risk_drivers.length > 0) {
        riskDriversHtml = data.risk_drivers.map(driver => `
          <div class="risk-driver ${driver.direction === 'decrease' ? 'decrease' : ''}">
            ${driver.direction === 'decrease' ? 'Down' : 'Up'}
            <strong>${driver.factor}</strong>
            ${driver.impact ? `→ ${driver.impact}` : ''}
          </div>
        `).join('');
      }

      document.getElementById("resultsContainer").innerHTML = `
        <div class="result-section">
          <h4>Prediction Result</h4>
          <div class="risk-card ${risk.toLowerCase()}">
            <div class="risk-level">Readmission Risk: ${risk}</div>
            <div>${outcome}</div>
            <h2>${normalized.percent.toFixed(1)}%</h2>
            <div>Probability Score</div>
          </div>
          ${data.recommended_action ? `<p><strong>Recommended Action:</strong> ${data.recommended_action}</p>` : ''}
        </div>

        <div class="result-section">
          <h4>BMI Assessment</h4>
          <p><strong>BMI:</strong> ${patientDetails.bmi ? patientDetails.bmi.toFixed(1) : 'N/A'}</p>
          <p><strong>Category:</strong> ${patientDetails.bmiCategory || 'N/A'}</p>
          ${data.estimated_glucose ? `<p><strong>Estimated Glucose:</strong> ${data.estimated_glucose} mg/dL</p>` : ''}
          ${data.num_diagnoses ? `<p><strong>Estimated Diagnoses:</strong> ${data.num_diagnoses}</p>` : ''}
          ${data.num_medications ? `<p><strong>Estimated Medications:</strong> ${data.num_medications}</p>` : ''}
        </div>

        ${riskDriversHtml ? `
          <div class="result-section">
            <h4>Key Risk Drivers</h4>
            ${riskDriversHtml}
          </div>
        ` : ''}

        <div class="result-section">
          <h4>Patient Summary</h4>
          <div class="summary-grid">
            <div class="summary-item">
              <strong>Age</strong>${patientDetails.age}
            </div>
            <div class="summary-item">
              <strong>Height</strong>
              ${patientDetails.enteredHeight} ${patientDetails.heightUnit}
              (${patientDetails.heightCm ? patientDetails.heightCm.toFixed(1) : 'N/A'} cm)
            </div>
            <div class="summary-item">
              <strong>Weight</strong>
              ${patientDetails.enteredWeight} ${patientDetails.weightUnit}
              (${patientDetails.weightKg ? patientDetails.weightKg.toFixed(1) : 'N/A'} kg)
            </div>
            <div class="summary-item">
              <strong>BMI</strong>
              ${patientDetails.bmi ? patientDetails.bmi.toFixed(1) : 'N/A'} — ${patientDetails.bmiCategory || 'N/A'}
            </div>
            <div class="summary-item">
              <strong>Previous admissions</strong>${patientDetails.admissionCount}
            </div>
            <div class="summary-item">
              <strong>Diabetes</strong>${diabetesSummary}
            </div>
            <div class="summary-item">
              <strong>Admission type</strong>${admissionTypes}
            </div>
            <div class="summary-item">
              <strong>Discharge type</strong>${dischargeTypes}
            </div>
            <div class="summary-item">
              <strong>Comorbidities</strong>
              ${patientDetails.comorbidities && patientDetails.comorbidities.length
                ? patientDetails.comorbidities.join(", ")
                : "None reported"}
            </div>
          </div>
        </div>`;
    }

    async function startAssessment() {
      const age = Number(document.getElementById("ageInput").value);
      const measurements = getMetricMeasurements();
      const admissionCount = Number(
        document.getElementById("admissionCountInput").value
      );
      const visits = getVisitDetails();
      const diabetesStatus = getDiabetesStatus();
      const diabetes = diabetesStatus === "yes" ||
        diabetesStatus === "not_sure";

      const diabetesStageInput = document.querySelector(
        "input[name='diabetesStage']:checked"
      );
      const diabetesStage = diabetesStageInput
        ? diabetesStageInput.value
        : "";

      const comorbidities = [
        ...document.querySelectorAll("input[name='comorbidity']:checked")
      ].map(input => input.value);

      if (!Number.isInteger(age) || age < 0 || age > 120) {
        showMessage("Please enter a valid age.", true);
        return;
      }

      if (!Number.isFinite(measurements.heightCm) ||
          measurements.heightCm <= 0 ||
          measurements.heightCm > 300) {
        showMessage("Please enter a valid height.", true);
        return;
      }

      if (!Number.isFinite(measurements.weightKg) ||
          measurements.weightKg <= 0 ||
          measurements.weightKg > 500) {
        showMessage("Please enter a valid weight.", true);
        return;
      }

      if (!Number.isInteger(admissionCount) ||
          admissionCount < 0 || admissionCount > 12) {
        showMessage("Please enter a number of admissions from 0 to 12.", true);
        return;
      }

      if (!diabetesStatus) {
        showMessage("Please select Yes, No, or Not sure for diabetes.", true);
        return;
      }

      if (diabetesStatus === "yes" && !diabetesStage) {
        showMessage("Please select one diabetes stage/type.", true);
        return;
      }

      for (const visit of visits) {
        if (!Number.isInteger(visit.length_of_stay_days) ||
            visit.length_of_stay_days < 1 ||
            !visit.admission_type ||
            !visit.discharge_type) {
          showMessage("Please complete all previous admission details.", true);
          return;
        }
      }

      const bmi = calculateBMI(
        measurements.heightCm,
        measurements.weightKg
      );

      patientDetails = {
        age,
        ...measurements,
        bmi,
        bmiCategory:getBMICategory(bmi),
        admissionCount,
        visits,
        diabetes,
        diabetesStatus,
        diabetesStage: diabetesStatus === "not_sure"
          ? "Prediabetes"
          : diabetesStage,
        comorbidities
      };

      const button = document.getElementById("startBtn");
      button.disabled = true;
      showMessage("Generating readmission assessment...");

      try {
        const response = await fetch(`${API_BASE_URL}/api/start`, {
          method:"POST",
          headers:{"Content-Type":"application/json"},
          body:JSON.stringify({
            session_id:sessionId,
            age,
            height_cm:measurements.heightCm,
            weight_kg:measurements.weightKg,
            bmi,
            bmi_category:patientDetails.bmiCategory,
            admission_count:admissionCount,
            visits,
            diabetes,
            diabetes_status:diabetesStatus,
            diabetes_stage:patientDetails.diabetesStage,
            comorbidities
          })
        });

        const data = await response.json();

        if (!data.success) {
          showMessage(`Error: ${data.error}`, true);
          return;
        }

        renderRiskResult(data);
        document.getElementById("summaryBtn").style.display = "block";
        document.getElementById("menu")
          .scrollIntoView({behavior:"smooth"});
        showMessage("Assessment completed.");
      } catch (error) {
        showMessage(`Connection error: ${error.message}`, true);
      } finally {
        button.disabled = false;
      }
    }

    function startNewAssessment() {
      patientDetails = {};
      createSession();

      [
        "ageInput",
        "heightInput",
        "weightInput",
        "admissionCountInput"
      ].forEach(id => document.getElementById(id).value = "");

      document.getElementById("visitTableContainer").innerHTML = "";
      document.getElementById("setupMessage").innerHTML = "";

      [
        "diabetesYes",
        "diabetesNo",
        "diabetesNotSure"
      ].forEach(id => {
        document.getElementById(id).checked = false;
      });

      document.getElementById("diabetesStages")
        .classList.remove("visible");

      document.querySelectorAll(
        "input[name='diabetesStage'], input[name='comorbidity']"
      ).forEach(input => {
        input.checked = false;
        input.disabled = false;
      });

      document.getElementById("resultsContainer").innerHTML = `
        <div class="result-section w3-center w3-text-grey">
          Complete the patient details to view the assessment.
        </div>`;

      document.getElementById("summaryBtn").style.display = "none";
      document.getElementById("startBtn").disabled = false;
    }

    updateHeightUnit();
    updateWeightUnit();
  </script>
</body>
</html>"""  # Your HTML code from the original file

# Your helper functions from the original code
def calculate_comorbidity_score(comorbidities):
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

def map_comorbidity_to_realistic_values(comorbidities, age, bmi, diabetes_status, diabetes_stage):
    num_diagnoses = 1
    num_medications = 1
    num_diagnoses += len(comorbidities)
    if diabetes_status == "yes":
        num_diagnoses += 1
        num_medications += 1
    severe_conditions = ['Congestive heart failure', 'Moderate-to-severe renal disease', 
                         'Moderate-to-severe liver disease', 'Metastatic solid tumor', 'AIDS']
    for condition in severe_conditions:
        if condition in comorbidities:
            num_medications += 2
    if age >= 65:
        num_diagnoses += 1
        num_medications += 1
    if age >= 75:
        num_diagnoses += 1
        num_medications += 1
    return {
        'num_diagnoses': min(num_diagnoses, 15),
        'num_medications': min(num_medications, 20)
    }

def predict_readmission(features):
    if len(features) != 15:
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
    risk_points = 0
    max_points = 100
    if age >= 75: risk_points += 15
    elif age >= 65: risk_points += 10
    elif age >= 55: risk_points += 5
    if bmi >= 35: risk_points += 15
    elif bmi >= 30: risk_points += 10
    elif bmi >= 25: risk_points += 5
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
    if admission_count >= 5: risk_points += 20
    elif admission_count >= 3: risk_points += 15
    elif admission_count >= 1: risk_points += 8
    if comorbidity_score >= 6: risk_points += 15
    elif comorbidity_score >= 4: risk_points += 10
    elif comorbidity_score >= 2: risk_points += 5
    if length_of_stay >= 10: risk_points += 10
    elif length_of_stay >= 7: risk_points += 7
    elif length_of_stay >= 5: risk_points += 4
    if emergency_count >= 2: risk_points += 10
    elif emergency_count >= 1: risk_points += 7
    if discharge_type >= 3: risk_points += 5
    elif discharge_type >= 2: risk_points += 3
    elif discharge_type >= 1: risk_points += 2
    if glucose >= 250: risk_points += 10
    elif glucose >= 200: risk_points += 8
    elif glucose >= 140: risk_points += 5
    probability = risk_points / max_points
    scaled_probability = 0.05 + (probability * 0.85)
    return min(scaled_probability, 0.95)

# Routes
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

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
            discharge_types = [v.get('discharge_type', 'Standard Discharge') for v in visits if v.get('discharge_type')]
        
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
        glucose = estimate_glucose_from_diabetes(diabetes_status, diabetes_stage, bmi, age)
        mapped_features = map_comorbidity_to_realistic_values(comorbidities, age, bmi, diabetes_status, diabetes_stage)
        
        has_diabetes_binary = 1 if diabetes_status == "yes" else 0
        prolonged_stay = 1 if length_of_stay >= 7 else 0
        frequent_admissions = 1 if admission_count >= 3 else 0
        high_diagnoses = 1 if mapped_features['num_diagnoses'] >= 10 else 0
        high_medications = 1 if mapped_features['num_medications'] >= 15 else 0
        high_glucose = 1 if glucose >= 200 else 0
        obese = 1 if bmi >= 30 else 0
        
        patient_data = {
            'age': age, 'bmi': bmi, 'bmi_category': bmi_category,
            'glucose': glucose, 'diabetes_status': diabetes_status,
            'diabetes_stage': diabetes_stage, 'discharge_type': discharge_type,
            'admission_count': admission_count, 'length_of_stay': length_of_stay,
            'emergency_count': emergency_count, 'comorbidity_score': comorbidity_score,
            'num_diagnoses': mapped_features['num_diagnoses'],
            'num_medications': mapped_features['num_medications'],
            'visits': visits, 'comorbidities': comorbidities
        }
        sessions[session_id] = patient_data
        
        features = [
            age, length_of_stay, mapped_features['num_diagnoses'],
            mapped_features['num_medications'], admission_count, glucose,
            bmi, has_diabetes_binary, discharge_type, prolonged_stay,
            frequent_admissions, high_diagnoses, high_medications,
            high_glucose, obese
        ]
        
        model_pred, model_prob = predict_readmission(features)
        risk_prob = calculate_risk_score_from_factors(
            age, bmi, diabetes_status, diabetes_stage,
            admission_count, comorbidity_score,
            length_of_stay, emergency_count,
            discharge_type, glucose
        )
        
        final_prob = (0.3 * model_prob) + (0.7 * risk_prob)
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
        # Add risk drivers based on factors
        if age >= 65:
            risk_drivers.append({'factor': f'Elderly patient ({age} years)', 'impact': '+20% risk', 'direction': 'increase'})
        if bmi >= 30:
            risk_drivers.append({'factor': f'Obesity (BMI: {bmi:.1f})', 'impact': '+15% risk', 'direction': 'increase'})
        if diabetes_status == "yes":
            risk_drivers.append({'factor': f'Diabetes present', 'impact': '+20% risk', 'direction': 'increase'})
        if admission_count >= 3:
            risk_drivers.append({'factor': f'Multiple admissions ({admission_count} in 12 months)', 'impact': '+20% risk', 'direction': 'increase'})
        if comorbidity_score >= 4:
            risk_drivers.append({'factor': f'Moderate comorbidity burden (score: {comorbidity_score})', 'impact': '+20% risk', 'direction': 'increase'})
        if length_of_stay >= 7:
            risk_drivers.append({'factor': f'Extended stay ({length_of_stay} days)', 'impact': '+15% risk', 'direction': 'increase'})
        if glucose >= 200:
            risk_drivers.append({'factor': f'High glucose ({glucose:.0f} mg/dL)', 'impact': '+15% risk', 'direction': 'increase'})
        
        if not risk_drivers:
            risk_drivers.append({'factor': 'No major elevated risk drivers identified', 'impact': 'Low risk profile', 'direction': 'decrease'})
        
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
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/summary', methods=['GET'])
def get_summary():
    try:
        session_id = request.args.get('session_id')
        if session_id not in sessions:
            return jsonify({'success': False, 'error': 'Session not found'}), 404
        patient_data = sessions[session_id]
        features = [
            patient_data['age'], patient_data['length_of_stay'],
            patient_data['num_diagnoses'], patient_data['num_medications'],
            patient_data['admission_count'], patient_data['glucose'],
            patient_data['bmi'], 1 if patient_data['diabetes_status'] == "yes" else 0,
            patient_data['discharge_type'], 1 if patient_data['length_of_stay'] >= 7 else 0,
            1 if patient_data['admission_count'] >= 3 else 0,
            1 if patient_data['num_diagnoses'] >= 10 else 0,
            1 if patient_data['num_medications'] >= 15 else 0,
            1 if patient_data['glucose'] >= 200 else 0,
            1 if patient_data['bmi'] >= 30 else 0
        ]
        model_pred, model_prob = predict_readmission(features)
        risk_prob = calculate_risk_score_from_factors(
            patient_data['age'], patient_data['bmi'],
            patient_data['diabetes_status'], patient_data['diabetes_stage'],
            patient_data['admission_count'], patient_data['comorbidity_score'],
            patient_data['length_of_stay'], patient_data['emergency_count'],
            patient_data['discharge_type'], patient_data['glucose']
        )
        final_prob = (0.3 * model_prob) + (0.7 * risk_prob)
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
            'success': True, 'prediction': int(final_prob > 0.35),
            'predicted_outcome': outcome, 'probability': final_prob,
            'probability_score': round(risk_percent, 1),
            'risk_level': risk_level, 'risk_category': risk_level,
            'patient_id': session_id
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
