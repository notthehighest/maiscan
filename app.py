from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import firebase_admin
from firebase_admin import credentials, firestore, auth
import pyrebase
import datetime
import os
from werkzeug.utils import secure_filename
import secrets
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from dotenv import load_dotenv

# ---------------- LOAD ENV ----------------
load_dotenv()

# ---------------- FLASK SETUP ----------------
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", secrets.token_hex(16))

UPLOAD_FOLDER = "static/user_image"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

# ---------------- FIREBASE SETUP ----------------
def initialize_firebase():
    """Initialize Firebase with flexible credential loading"""
    try:
        # Method 1: Try environment variables first (for Render/Production)
        if all([
            os.getenv("FIREBASE_TYPE"),
            os.getenv("FIREBASE_PROJECT_ID"),
            os.getenv("FIREBASE_PRIVATE_KEY_ID"),
            os.getenv("FIREBASE_PRIVATE_KEY"),
            os.getenv("FIREBASE_CLIENT_EMAIL"),
            os.getenv("FIREBASE_CLIENT_ID")
        ]):
            print("Loading Firebase credentials from environment variables...")
            firebase_credentials = {
                "type": os.getenv("FIREBASE_TYPE"),
                "project_id": os.getenv("FIREBASE_PROJECT_ID"),
                "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
                "private_key": os.getenv("FIREBASE_PRIVATE_KEY").replace('\\n', '\n'),
                "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
                "client_id": os.getenv("FIREBASE_CLIENT_ID"),
                "auth_uri": os.getenv("FIREBASE_AUTH_URI", "https://accounts.google.com/o/oauth2/auth"),
                "token_uri": os.getenv("FIREBASE_TOKEN_URI", "https://oauth2.googleapis.com/token"),
                "auth_provider_x509_cert_url": os.getenv("FIREBASE_AUTH_PROVIDER_X509_CERT_URL", "https://www.googleapis.com/oauth2/v1/certs"),
                "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_X509_CERT_URL"),
                "universe_domain": "googleapis.com"
            }
            cred = credentials.Certificate(firebase_credentials)
            print("Firebase credentials loaded from environment variables")
            
        # Method 2: Fall back to JSON file (for Local Development)
        else:
            print("Loading Firebase credentials from JSON file...")
            firebase_credentials_path = os.getenv("FIREBASE_CREDENTIALS", "project-maiscan-firebase-adminsdk-fbsvc-8491da1d45.json")
            
            if not firebase_credentials_path or not os.path.exists(firebase_credentials_path):
                raise FileNotFoundError(f"Firebase credentials file not found: {firebase_credentials_path}")
            
            cred = credentials.Certificate(firebase_credentials_path)
            print(f"Firebase credentials loaded from JSON file: {firebase_credentials_path}")
        
        # Initialize Firebase Admin SDK
        firebase_admin.initialize_app(cred)
        return firestore.client()
        
    except Exception as e:
        print(f"Firebase initialization error: {e}")
        raise e

# Initialize Firebase
db = initialize_firebase()

# ---------------- PYREBASE SETUP ----------------
def initialize_pyrebase():
    """Initialize Pyrebase for client-side Firebase operations"""
    try:
        firebaseConfig = {
            "apiKey": os.getenv("FIREBASE_API_KEY"),
            "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
            "databaseURL": os.getenv("FIREBASE_DATABASE_URL"),
            "projectId": os.getenv("FIREBASE_PROJECT_ID"),
            "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
            "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID"),
            "appId": os.getenv("FIREBASE_APP_ID"),
            "measurementId": os.getenv("FIREBASE_MEASUREMENT_ID")
        }
        
        # Validate required fields
        required_fields = ["apiKey", "authDomain", "projectId"]
        missing_fields = [field for field in required_fields if not firebaseConfig.get(field)]
        
        if missing_fields:
            print(f"Missing Pyrebase config fields: {missing_fields}")
            return None
        
        pb = pyrebase.initialize_app(firebaseConfig)
        pb_auth = pb.auth()
        print("Pyrebase initialized successfully")
        return pb, pb_auth
        
    except Exception as e:
        print(f"Pyrebase initialization warning: {e}")
        return None, None

# Initialize Pyrebase
pb_result = initialize_pyrebase()
if pb_result:
    pb, pb_auth = pb_result
else:
    pb, pb_auth = None, None

# ---------------- FLASK-LOGIN SETUP ----------------
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = "Please log in to access this page."
login_manager.login_message_category = "info"

class User(UserMixin):
    def __init__(self, uid, email, username=None):
        self.id = uid
        self.email = email
        self.username = username

@login_manager.user_loader
def load_user(user_id):
    try:
        user_record = auth.get_user(user_id)
        user_doc = db.collection("Users").document(user_id).get()
        username = None
        if user_doc.exists:
            username = user_doc.to_dict().get("username")
        return User(uid=user_record.uid, email=user_record.email, username=username)
    except Exception as e:
        print("Error loading user:", e)
        return None

# ---------------- MODEL DOWNLOAD AND LOADING ----------------
def download_model_from_mega():
    model_path = "maiscan_disease_model_final.keras"
    
    # Check if model already exists
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path)
        print(f"Model already exists: {model_path} ({file_size} bytes)")
        return model_path
    
    print("Downloading model from Mega...")
    try:
        # Try importing mega - this might fail on some systems
        from mega import Mega
        
        # Initialize Mega client
        mega = Mega()
        m = mega.login()  # Anonymous login
        
        # Your Mega URL
        mega_url = "https://mega.nz/file/eoQTgJaR#maLXsn2jC5kTGnwGpdEi9DGUcbSslRXhs5NgC2iqxU4"
        
        # Download the file
        m.download_url(mega_url, dest_filename=model_path)
        
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            print(f"Model downloaded successfully from Mega ({file_size} bytes)")
            return model_path
        else:
            print("Model download completed but file not found")
            return None
        
    except ImportError:
        print("Mega library not available - cannot download model")
        return None
    except Exception as e:
        print(f"Error downloading model from Mega: {e}")
        return None

def load_ml_model():
    """Load the ML model with proper error handling"""
    print("=" * 50)
    print("INITIALIZING ML MODEL")
    print("=" * 50)
    
    # Debug: Check current directory and files
    print(f"Current working directory: {os.getcwd()}")
    print(f"Files in current directory: {os.listdir('.')}")
    
    # Try to download model from Mega
    model_path = download_model_from_mega()
    
    if not model_path or not os.path.exists(model_path):
        print("MODEL NOT FOUND - ML functionality will be disabled")
        return None
    
    try:
        print(f"Loading model from: {model_path}")
        model = load_model(model_path)
        print("MODEL LOADED SUCCESSFULLY")
        print("=" * 50)
        return model
        
    except Exception as e:
        print(f"ERROR LOADING MODEL: {e}")
        print("ML functionality will be disabled")
        print("=" * 50)
        return None

# Initialize ML Model
model = load_ml_model()

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("base.html")

# -------- REGISTER --------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not email or not password:
            flash("Email and password are required.", "danger")
            return render_template("register.html")

        try:
            # Create user in Firebase Authentication
            user_record = auth.create_user(email=email, password=password)

            # Save extra data in Firestore
            db.collection("Users").document(user_record.uid).set({
                "email": email,
                "created_at": datetime.datetime.utcnow()
            })

            flash("Registration successful! Please log in.", "success")
            return redirect(url_for("login"))

        except Exception as e:
            print("Registration error:", e)
            flash("Registration failed: " + str(e), "danger")

    return render_template("register.html")

# -------- LOGIN --------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not email or not password:
            flash("Email and password are required.", "danger")
            return render_template("login.html")

        try:
            # Check if pyrebase is available
            if pb_auth is None:
                flash("Authentication service not available.", "danger")
                return render_template("login.html")
            
            # Authenticate with Firebase using Pyrebase
            user = pb_auth.sign_in_with_email_and_password(email, password)

            # Get Firebase user record
            user_record = auth.get_user(user["localId"])

            # Flask-Login user
            user_obj = User(uid=user_record.uid, email=user_record.email)
            login_user(user_obj)

            flash("Login successful!", "success")
            return redirect(url_for("maiscan"))

        except Exception as e:
            print("Login error:", e)
            flash("Invalid email or password.", "danger")

    return render_template("login.html")

# -------- FORGOT PASSWORD --------
@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        
        if not email:
            flash("Email is required.", "danger")
            return render_template("forgot_password.html")
        
        try:
            if pb_auth is None:
                flash("Password reset service not available.", "danger")
                return render_template("forgot_password.html")
                
            # Send password reset email
            pb_auth.send_password_reset_email(email)
            flash("Password reset email sent! Check your inbox.", "success")
            return redirect(url_for("login"))
            
        except Exception as e:
            print("Password reset error:", e)
            error_msg = str(e)
            if "INVALID_EMAIL" in error_msg:
                flash("Invalid email address.", "danger")
            elif "MISSING_EMAIL" in error_msg:
                flash("Email is required.", "danger")
            else:
                flash("Error sending reset email. Please try again.", "danger")
    
    return render_template("forgot_password.html")

# -------- RESET PASSWORD --------
@app.route("/reset-password", methods=["GET", "POST"])
def reset_password():
    return render_template("reset_password.html")

# -------- LOGOUT --------
@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for("home"))

# -------- UPDATE ACCOUNT --------
@app.route("/update-account", methods=["POST"])
@login_required
def update_account():
    username = request.form.get("username", "").strip()
    email = request.form.get("email", "").strip()
    password = request.form.get("password", "").strip()

    try:
        updates = {}

        # Update email
        if email and email != current_user.email:
            auth.update_user(current_user.id, email=email)
            updates["email"] = email

        # Update password
        if password:
            auth.update_user(current_user.id, password=password)

        # Update Firestore user profile
        if username:
            updates["username"] = username

        if updates:
            db.collection("Users").document(current_user.id).update(updates)

        flash("Account updated successfully!", "success")
    except Exception as e:
        print("Update error:", e)
        flash("Failed to update account: " + str(e), "danger")

    return redirect(url_for("maiscan"))

# -------- MAISCAN DASHBOARD --------
@app.route("/maiscan")
@login_required
def maiscan():
    try:
        # Fetch user's uploads from Firestore
        uploads_ref = db.collection("UploadedImages").where("user_id", "==", current_user.id)
        uploads = [doc.to_dict() for doc in uploads_ref.stream()]

        # Disease stats
        disease_counts = {}
        for up in uploads:
            disease = up.get("disease_type", "Unknown")
            disease_counts[disease] = disease_counts.get(disease, 0) + 1

        total_images = sum(disease_counts.values())
        disease_count = sum(c for d, c in disease_counts.items() if "healthy" not in d.lower())
        most_common_disease = max(
            (d for d in disease_counts if "healthy" not in d.lower()),
            key=lambda d: disease_counts[d],
            default="None"
        )
        disease_types = list(disease_counts.keys())

    except Exception as e:
        print("Error loading dashboard:", e)
        uploads, disease_counts, total_images, disease_count, most_common_disease, disease_types = [], {}, 0, 0, "None", []

    return render_template(
        "mais.html",
        uploads=uploads,
        disease_counts=disease_counts,
        total_images=total_images,
        disease_count=disease_count,
        most_common_disease=most_common_disease,
        disease_types=disease_types
    )

# -------- PREDICTION FUNCTION --------
def pred_corn_disease(img_path):
    """Predict corn disease from image path"""
    try:
        if model is None:
            return "Model Not Available", "invalid_image.html", 0.0
            
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        pred_class = np.argmax(prediction)
        confidence = float(np.max(prediction))

        CONFIDENCE_THRESHOLD = 0.5
        if confidence < CONFIDENCE_THRESHOLD:
            return "Unknown Class", "invalid_image.html", confidence

        diseases = {
            0: ("Aphids", "aphids.html"),
            1: ("Armyworm", "armyworm.html"),
            2: ("Common Cutworm", "common_cutworm.html"),
            3: ("Common Rust", "common_rust.html"),
            4: ("Common Smut", "common_smut.html"),
            5: ("Corn Borer", "corn_borer.html"),
            6: ("Earwig", "earwig.html"),
            7: ("Fusarium Ear Rot", "fusarium_ear_rot.html"),
            8: ("Gray Leaf Spot", "gray_leaf_spot.html"),
            9: ("Healthy Corn", "healthycorn.html"),
            10: ("Healthy Leaf", "healthyleaf.html"),
            11: ("Leaf Blight", "leaf_blight.html"),
            12: ("Leafhopper", "leafhopper.html"),
        }

        return diseases.get(pred_class, ("Unknown Class", "invalid_image.html")) + (confidence,)

    except Exception as e:
        print("Error in prediction:", e)
        return "Error in Prediction", "invalid_image.html", 0.0

# -------- PREDICTION --------
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    print("=" * 50)
    print("PREDICTION REQUEST RECEIVED")
    print(f"Model status: {'Available' if model is not None else 'NOT AVAILABLE'}")
    
    if model is None:
        print("Model is None - redirecting with error")
        flash("ML model is currently unavailable. Please contact support.", "danger")
        return redirect(url_for("maiscan"))
    
    if "image" not in request.files:
        print("No image in request files")
        flash("No image uploaded.", "danger")
        return redirect(url_for("maiscan"))

    file = request.files["image"]
    print(f"File received: {file.filename}")
    
    if file.filename == "" or not allowed_file(file.filename):
        print("Invalid file")
        flash("Invalid file type. Please upload PNG, JPG, or JPEG files.", "danger")
        return redirect(url_for("maiscan"))

    try:
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_")
        filename = timestamp + filename
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        print(f"File saved to: {file_path}")

        # Predict
        print("Starting prediction...")
        pred, output_page, confidence = pred_corn_disease(file_path)
        print(f"Prediction result: {pred}, Template: {output_page}, Confidence: {confidence}")

        # Save metadata to Firestore (only for valid predictions)
        if pred not in ["Unknown Class", "Model Not Available", "Error in Prediction"]:
            try:
                db.collection("UploadedImages").add({
                    "filename": filename,
                    "user_id": current_user.id,
                    "disease_type": pred,
                    "confidence": confidence,
                    "upload_date": datetime.datetime.utcnow()
                })
                print("Prediction saved to Firestore")
            except Exception as db_error:
                print(f"Error saving to Firestore: {db_error}")

        print(f"Rendering template: {output_page}")
        return render_template(output_page, pred_output=pred, user_image=file_path, confidence=confidence)

    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        flash("Error processing image. Please try again.", "danger")
        return redirect(url_for("maiscan"))

# -------- PREDICTION REALTIME --------
@app.route("/api/predict", methods=['POST'])
@login_required
def api_predict():
    print("API prediction request received")
    print(f"Model status: {'Available' if model is not None else 'NOT AVAILABLE'}")
    
    if model is None:
        return jsonify({
            "valid": False, 
            "error": "ML model is currently unavailable", 
            "disease": "", 
            "confidence": 0
        })
           
    if 'image' not in request.files:
        return jsonify({"valid": False, "error": "No image provided", "disease": "", "confidence": 0})
            
    file = request.files['image']
    if file.filename == '':
        return jsonify({"valid": False, "error": "No file selected", "disease": "", "confidence": 0})
            
    # Save temporary file
    filename = secure_filename(f"temp_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    try:
        # Predict disease
        pred, _, confidence = pred_corn_disease(file_path)
        
        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Check if it's a valid prediction
        is_valid = pred not in ["Model Not Available", "Error in Prediction", "Unknown Class"]
        
        return jsonify({
            "valid": is_valid,
            "disease": pred if is_valid else "",
            "confidence": confidence,
            "error": pred if not is_valid else ""
        })
            
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        print(f"Error in API prediction: {e}")
        return jsonify({"valid": False, "error": "Prediction failed", "disease": "", "confidence": 0}), 500

if __name__ == "__main__":
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)