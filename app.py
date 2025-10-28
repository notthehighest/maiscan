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
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from dotenv import load_dotenv
import gdown
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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

# Check and fix permissions for upload folder
upload_folder = app.config["UPLOAD_FOLDER"]
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)
os.chmod(upload_folder, 0o755)  # Ensure proper permissions

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
            print("🔄 Loading Firebase credentials from environment variables...")
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
            print("✅ Firebase credentials loaded from environment variables")
            
        # Method 2: Fall back to JSON file (for Local Development)
        else:
            print("🔄 Loading Firebase credentials from JSON file...")
            firebase_credentials_path = os.getenv("FIREBASE_CREDENTIALS", "project-maiscan-firebase-adminsdk-fbsvc-8491da1d45.json")
            
            if not firebase_credentials_path or not os.path.exists(firebase_credentials_path):
                raise FileNotFoundError(f"Firebase credentials file not found: {firebase_credentials_path}")
            
            cred = credentials.Certificate(firebase_credentials_path)
            print(f"✅ Firebase credentials loaded from JSON file: {firebase_credentials_path}")
        
        # Initialize Firebase Admin SDK
        firebase_admin.initialize_app(cred)
        return firestore.client()
        
    except Exception as e:
        print(f"❌ Firebase initialization error: {e}")
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
            print(f"⚠️ Missing Pyrebase config fields: {missing_fields}")
            return None
        
        pb = pyrebase.initialize_app(firebaseConfig)
        pb_auth = pb.auth()
        print("✅ Pyrebase initialized successfully")
        return pb, pb_auth
        
    except Exception as e:
        print(f"⚠️ Pyrebase initialization warning: {e}")
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

# ---------------- ML MODEL LOADING ----------------
model = None

def load_ml_model():
    """Load ML model with compatibility for TensorFlow 2.16.1 and Keras 3.2.0"""
    global model
    
    try:
        # Clear any existing TensorFlow session
        import tensorflow as tf
        tf.keras.backend.clear_session()
        
        # Google Drive file ID
        FILE_ID = os.getenv("MODEL_FILE_ID", "1b-n8usXAIBmsBV8TqPz4nkIXqqcBM-fP")
        MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(BASE_DIR, "maiscan_disease_model_final_finetuned.keras")

        # Debug: Check current directory and files
        logger.debug(f"Current directory: {BASE_DIR}")
        logger.debug(f"Files in directory: {os.listdir(BASE_DIR)}")
        logger.debug(f"Model path: {model_path}")

        # Download model if it doesn't already exist
        if not os.path.exists(model_path):
            print("🔄 Downloading model from Google Drive...")
            try:
                gdown.download(MODEL_URL, model_path, quiet=False)
                print("✅ Model downloaded successfully")
                
                # Verify the file was downloaded
                if os.path.exists(model_path):
                    file_size = os.path.getsize(model_path)
                    print(f"✅ Model file size: {file_size} bytes")
                else:
                    print("❌ Model file not found after download")
                    return
                    
            except Exception as download_error:
                print(f"❌ Error downloading model: {download_error}")
                return

        # Verify model file exists and has content
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            print(f"📁 Model file exists, size: {file_size} bytes")
            
            if file_size == 0:
                print("❌ Model file is empty")
                os.remove(model_path)  # Remove empty file
                return
        else:
            print("❌ Model file does not exist")
            return

        # Load the model with improved error handling
        print("🔄 Loading model...")
        try:
            # For TensorFlow 2.16.1 + Keras 3.2.0 compatibility
            model = load_model(model_path, compile=False)
            print("✅ Model loaded successfully")
            
            # Test the model with a simple prediction
            test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
            test_prediction = model.predict(test_input, verbose=0)
            print(f"✅ Model test prediction shape: {test_prediction.shape}")
            
        except Exception as load_error:
            print(f"❌ Error loading model: {load_error}")
            model = None
            # Try to clean up corrupted file
            if os.path.exists(model_path):
                os.remove(model_path)
                print("🗑️ Removed potentially corrupted model file")

    except Exception as e:
        print(f"❌ Unexpected error in model loading: {e}")
        model = None

# Load the model when the app starts
print("🚀 Starting ML model loading...")
load_ml_model()

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("base.html")

@app.route("/debug")
def debug():
    model_status = "Loaded" if model is not None else "Not Loaded"
    model_path_debug = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maiscan_disease_model_final_finetuned.keras")
    model_exists = os.path.exists(model_path_debug)
    file_size = os.path.getsize(model_path_debug) if model_exists else 0
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    files_in_dir = os.listdir(current_dir)
    
    return jsonify({
        "model_status": model_status,
        "model_path": model_path_debug,
        "model_exists": model_exists,
        "model_file_size": file_size,
        "current_directory": current_dir,
        "files_in_dir": files_in_dir,
        "upload_folder_exists": os.path.exists(app.config["UPLOAD_FOLDER"])
    })

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
            # ✅ Create user in Firebase Authentication
            user_record = auth.create_user(email=email, password=password)

            # ✅ Save extra data in Firestore
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
            # ✅ Authenticate with Firebase using Pyrebase
            user = pb_auth.sign_in_with_email_and_password(email, password)

            # ✅ Get Firebase user record
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
        # ✅ Fetch user's uploads from Firestore
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
        disease_types=disease_types,
        model_loaded=model is not None
    )

# -------- PREDICTION --------
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if model is None:
        flash("ML model is not available. Please try again later.", "danger")
        return redirect(url_for("maiscan"))
        
    if "image" not in request.files:
        flash("No image uploaded.", "danger")
        return redirect(url_for("maiscan"))

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        flash("Invalid file type.", "danger")
        return redirect(url_for("maiscan"))

    try:
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_")
        filename = timestamp + filename
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Predict
        pred, output_page, confidence = pred_corn_disease(file_path)

        # ✅ Save metadata to Firestore
        if pred != "Unknown Class" and pred != "Model Error":
            db.collection("UploadedImages").add({
                "filename": filename,
                "user_id": current_user.id,
                "disease_type": pred,
                "confidence": confidence,
                "upload_date": datetime.datetime.utcnow()
            })

        return render_template(output_page, pred_output=pred, user_image=file_path, confidence=confidence)

    except Exception as e:
        print("Prediction error:", e)
        flash("Error processing image.", "danger")
        return redirect(url_for("maiscan"))

# -------- PREDICTION REALTIME --------
@app.route("/api/predict", methods=['POST'])
@login_required
def api_predict():
    if model is None:
        return jsonify({"valid": False, "error": "ML model not available", "disease": "", "confidence": 0})
            
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
        
        # Check if it's an invalid image
        is_valid = not pred.startswith("Invalid Image") and not pred.startswith("Model Error")
        
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

# -------- PREDICTION FUNCTION --------
def pred_corn_disease(img_path):
    if model is None:
        print("❌ Model is not loaded, cannot make prediction")
        return "Model Error", "invalid_image.html", 0.0
    
    try:
        # Verify image file exists
        if not os.path.exists(img_path):
            print(f"❌ Image file not found: {img_path}")
            return "Invalid Image", "invalid_image.html", 0.0
            
        # Load and preprocess image - using tf.keras.utils.load_img for Keras 3.x
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array, verbose=0)
        pred_class = np.argmax(prediction)
        confidence = float(np.max(prediction))

        CONFIDENCE_THRESHOLD = 0.7
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

        disease_name, template_name = diseases.get(pred_class, ("Unknown Class", "invalid_image.html"))
        return disease_name, template_name, confidence

    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        return "Prediction Error", "invalid_image.html", 0.0

# Health check endpoint for Render
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.datetime.utcnow().isoformat()
    })

if __name__ == "__main__":
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get("PORT", 8080))
    
    # Debug information
    print(f"🚀 Starting Flask app on port {port}")
    print(f"📁 Current directory: {os.getcwd()}")
    print(f"📁 Files in directory: {os.listdir('.')}")
    print(f"✅ Model status: {'Loaded' if model is not None else 'Not loaded'}")
    
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)