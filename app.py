from flask import Flask, render_template, request, jsonify
import sqlite3
import os
import re
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

app = Flask(__name__)
DB_PATH = "database.db"

# ── Gemini AI setup ──────────────────────────────────────
genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))


# ── Database init ────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name   TEXT    NOT NULL,
            dob         TEXT    NOT NULL,
            email       TEXT    NOT NULL,
            glucose     REAL    NOT NULL,
            haemoglobin REAL    NOT NULL,
            cholesterol REAL    NOT NULL,
            remarks     TEXT    DEFAULT ''
        )
    ''')
    conn.commit()
    conn.close()

# ── Validation helpers ───────────────────────────────────
def valid_email(email):
    return re.match(r'^[\w\.-]+@[\w\.-]+\.\w{2,}$', email)

def valid_dob(dob):
    try:
        return datetime.strptime(dob, "%Y-%m-%d") < datetime.now()
    except ValueError:
        return False

def valid_number(val):
    try:
        return float(val) > 0
    except (ValueError, TypeError):
        return False

# ── AI prediction ────────────────────────────────────────
def get_ai_remarks(name, glucose, haemoglobin, cholesterol):
    try:
        # Check if API key is configured
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            print("⚠️ WARNING: GEMINI_API_KEY not configured. Using fallback.")
            return generate_fallback_remarks(name, glucose, haemoglobin, cholesterol)
        
        prompt = f"""
You are a medical assistant. Based on the following blood test results,
give a brief 2-3 line health assessment mentioning possible risks or healthy status.
Do NOT give specific diagnoses — only general health observations.

Patient: {name}
Glucose: {glucose} mg/dL  (Normal: 70–99)
Haemoglobin: {haemoglobin} g/dL  (Normal: Men 13.5–17.5, Women 12–15.5)
Cholesterol: {cholesterol} mg/dL  (Normal: below 200)

Reply in simple English, 2-3 sentences only.
"""
        print(f"🔄 Calling Gemini API for patient: {name}")
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt, request_options={"timeout": 10})
        
        if response and response.text:
            result = response.text.strip()
            print(f"✅ AI remarks generated: {result}")
            return result
        else:
            print("❌ Empty response from Gemini API")
            return generate_fallback_remarks(name, glucose, haemoglobin, cholesterol)
            
    except Exception as e:
        error_msg = str(e).lower()
        print(f"❌ API Error: {str(e)}")
        
        # Handle quota/rate limit errors gracefully
        if "quota" in error_msg or "rate" in error_msg or "429" in error_msg:
            print("🔄 Quota exceeded. Using fallback remarks.")
            return generate_fallback_remarks(name, glucose, haemoglobin, cholesterol)
        
        # Use fallback for any other error
        print("🔄 API failed. Using fallback remarks.")
        return generate_fallback_remarks(name, glucose, haemoglobin, cholesterol)

# ── Fallback health assessment (when API is unavailable) ────
def generate_fallback_remarks(name, glucose, haemoglobin, cholesterol):
    """Generate comprehensive health remarks based on blood test values"""
    remarks = []
    
    glucose_val = float(glucose)
    hb_val = float(haemoglobin)
    chol_val = float(cholesterol)
    
    # Glucose assessment
    if glucose_val < 70:
        remarks.append("⚠️ Low glucose (hypoglycemia risk)")
    elif glucose_val > 125:
        remarks.append("⚠️ Elevated glucose levels")
    else:
        remarks.append("✓ Glucose normal")
    
    # Hemoglobin assessment
    if hb_val < 12:
        remarks.append("⚠️ Low hemoglobin (anemia risk)")
    elif hb_val > 17.5:
        remarks.append("⚠️ High hemoglobin")
    else:
        remarks.append("✓ Hemoglobin normal")
    
    # Cholesterol assessment
    if chol_val > 200:
        remarks.append("⚠️ Cholesterol above normal")
    else:
        remarks.append("✓ Cholesterol normal")
    
    return " ".join(remarks)

# ── Routes ───────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

# TEST AI API
@app.route("/api/test-ai", methods=["GET"])
def test_ai():
    """Test if Gemini API is working"""
    try:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            return jsonify({"success": False, "error": "GEMINI_API_KEY not configured in .env"}), 400
        
        print("🧪 Testing Gemini API...")
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content("Say 'MIRA API is working!' in 5 words max.", request_options={"timeout": 10})
        
        if response and response.text:
            print(f"✅ API Test Successful: {response.text}")
            return jsonify({"success": True, "message": response.text}), 200
        else:
            return jsonify({"success": False, "error": "Empty response from API"}), 500
            
    except Exception as e:
        error_msg = str(e)
        print(f"❌ API Test Failed: {error_msg}")
        return jsonify({"success": False, "error": error_msg}), 500

# CREATE
@app.route("/api/patients", methods=["POST"])
def add_patient():
    data = request.json

    # Validation
    errors = []
    if not data.get("full_name", "").strip():
        errors.append("Full name is required.")
    if not valid_email(data.get("email", "")):
        errors.append("Invalid email address.")
    if not valid_dob(data.get("dob", "")):
        errors.append("Date of birth must be a past date.")
    for field in ["glucose", "haemoglobin", "cholesterol"]:
        if not valid_number(data.get(field)):
            errors.append(f"{field.capitalize()} must be a positive number.")
    if errors:
        return jsonify({"success": False, "errors": errors}), 400

    remarks = get_ai_remarks(
        data["full_name"],
        data["glucose"],
        data["haemoglobin"],
        data["cholesterol"]
    )

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO patients (full_name, dob, email, glucose, haemoglobin, cholesterol, remarks)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (data["full_name"].strip(), data["dob"], data["email"].strip(),
          float(data["glucose"]), float(data["haemoglobin"]),
          float(data["cholesterol"]), remarks))
    conn.commit()
    new_id = c.lastrowid
    conn.close()
    return jsonify({"success": True, "id": new_id, "remarks": remarks}), 201

# READ
@app.route("/api/patients", methods=["GET"])
def get_patients():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM patients ORDER BY id DESC").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

# READ single
@app.route("/api/patients/<int:pid>", methods=["GET"])
def get_patient(pid):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM patients WHERE id=?", (pid,)).fetchone()
    conn.close()
    if not row:
        return jsonify({"error": "Not found"}), 404
    return jsonify(dict(row))

# UPDATE
@app.route("/api/patients/<int:pid>", methods=["PUT"])
def update_patient(pid):
    data = request.json

    errors = []
    if not data.get("full_name", "").strip():
        errors.append("Full name is required.")
    if not valid_email(data.get("email", "")):
        errors.append("Invalid email address.")
    if not valid_dob(data.get("dob", "")):
        errors.append("Date of birth must be a past date.")
    for field in ["glucose", "haemoglobin", "cholesterol"]:
        if not valid_number(data.get(field)):
            errors.append(f"{field.capitalize()} must be a positive number.")
    if errors:
        return jsonify({"success": False, "errors": errors}), 400

    remarks = get_ai_remarks(
        data["full_name"],
        data["glucose"],
        data["haemoglobin"],
        data["cholesterol"]
    )

    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        UPDATE patients SET full_name=?, dob=?, email=?, glucose=?, haemoglobin=?,
        cholesterol=?, remarks=? WHERE id=?
    ''', (data["full_name"].strip(), data["dob"], data["email"].strip(),
          float(data["glucose"]), float(data["haemoglobin"]),
          float(data["cholesterol"]), remarks, pid))
    conn.commit()
    conn.close()
    return jsonify({"success": True, "remarks": remarks})

# DELETE
@app.route("/api/patients/<int:pid>", methods=["DELETE"])
def delete_patient(pid):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM patients WHERE id=?", (pid,))
    conn.commit()
    conn.close()
    return jsonify({"success": True})

# ────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
