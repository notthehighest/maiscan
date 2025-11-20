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
from tensorflow.keras.utils import load_img, img_to_array
from dotenv import load_dotenv
import gdown
import logging
import tempfile
import requests
import tensorflow as tf

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
tflite_interpreter = None
input_details = None
output_details = None
is_quantized = False
input_scale = 0
input_zero_point = 0
output_scale = 0
output_zero_point = 0

def load_tflite_model():
    """Load TFLite model from Google Drive"""
    global tflite_interpreter, input_details, output_details, is_quantized
    global input_scale, input_zero_point, output_scale, output_zero_point
    
    try:
        # Clear any existing TensorFlow session
        tf.keras.backend.clear_session()
        
        # Google Drive file ID
        FILE_ID = "1UCXZvjyozzoQKhA74ldc92PpAFsL_7t4"
        MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"

        print("🔄 Loading TFLite model from Google Drive...")
        
        # Create a temporary file to store the model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tflite') as temp_model_file:
            temp_model_path = temp_model_file.name

        try:
            # Download model using gdown
            gdown.download(MODEL_URL, temp_model_path, quiet=False)
            
            # Verify the file was downloaded
            if os.path.exists(temp_model_path):
                file_size = os.path.getsize(temp_model_path)
                print(f"✅ TFLite model downloaded successfully, size: {file_size} bytes")
                
                if file_size == 0:
                    print("❌ Model file is empty")
                    os.remove(temp_model_path)
                    return False
            else:
                print("❌ Model file not found after download")
                return False

            # Load TFLite model and allocate tensors
            print("🔄 Loading TFLite interpreter...")
            tflite_interpreter = tf.lite.Interpreter(model_path=temp_model_path)
            tflite_interpreter.allocate_tensors()
            
            # Get input and output tensors
            input_details = tflite_interpreter.get_input_details()
            output_details = tflite_interpreter.get_output_details()
            
            # Check if model is quantized
            is_quantized = input_details[0]['dtype'] == np.int8
            input_scale = input_details[0]['quantization'][0]
            input_zero_point = input_details[0]['quantization'][1]
            output_scale = output_details[0]['quantization'][0]
            output_zero_point = output_details[0]['quantization'][1]
            
            print(f"✅ TFLite model loaded successfully")
            print(f"📊 Input details: {input_details[0]['shape']}")
            print(f"📊 Output details: {output_details[0]['shape']}")
            print(f"📊 Model is quantized: {is_quantized}")
            if is_quantized:
                print(f"📊 Input scale: {input_scale}, zero point: {input_zero_point}")
                print(f"📊 Output scale: {output_scale}, zero point: {output_zero_point}")
            
            # Test the model with a simple prediction
            test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
            try:
                if is_quantized:
                    # Quantize the test input
                    test_input_quantized = (test_input / input_scale + input_zero_point).astype(np.int8)
                    tflite_interpreter.set_tensor(input_details[0]['index'], test_input_quantized)
                else:
                    tflite_interpreter.set_tensor(input_details[0]['index'], test_input)
                
                tflite_interpreter.invoke()
                test_output = tflite_interpreter.get_tensor(output_details[0]['index'])
                
                if is_quantized:
                    # Dequantize the output
                    test_output = (test_output.astype(np.float32) - output_zero_point) * output_scale
                
                print(f"✅ Model test prediction successful, shape: {test_output.shape}")
                return True
            except Exception as test_error:
                print(f"⚠️ Model test prediction failed: {test_error}")
                return True  # Still return True as model loaded
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)
                print("🗑️ Temporary model file cleaned up")

    except Exception as e:
        print(f"❌ Error loading TFLite model from cloud: {e}")
        return False

# Load the model when the app starts
print("🚀 Starting TFLite model loading from Google Drive...")
print(f"🧪 TensorFlow version: {tf.__version__}")

model_loaded = load_tflite_model()

if not model_loaded:
    print("❌ Failed to load TFLite model")
else:
    print("✅ TFLite model loaded successfully")

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("base.html")

@app.route("/debug")
def debug():
    model_status = "Loaded" if tflite_interpreter is not None else "Not Loaded"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    files_in_dir = os.listdir(current_dir)
    
    return jsonify({
        "model_status": model_status,
        "model_type": "TFLite",
        "model_quantized": is_quantized,
        "model_source": "Google Drive Cloud",
        "tensorflow_version": tf.__version__,
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
        chart_data = [] # Prep for JSON output

        for up in uploads:
            disease = up.get("disease_type", "Unknown")
            disease_counts[disease] = disease_counts.get(disease, 0) + 1
            
            # Process date for chart filtering
            upload_date = up.get("upload_date")
            date_str = ""
            if upload_date:
                # Handle Firestore timestamp (datetime obj)
                if hasattr(upload_date, 'isoformat'):
                    date_str = upload_date.isoformat()
                else:
                    date_str = str(upload_date)
            
            chart_data.append({
                "disease": disease,
                "date": date_str
            })

        total_images = sum(disease_counts.values())
        # Sum of all counts excluding healthy variants
        disease_count = sum(c for d, c in disease_counts.items() if "healthy" not in d.lower())
        
        # Calculate Disease Percentage
        disease_percentage = 0.0
        if total_images > 0:
            disease_percentage = round((disease_count / total_images) * 100, 1)

        most_common_disease = max(
            (d for d in disease_counts if "healthy" not in d.lower()),
            key=lambda d: disease_counts[d],
            default="None"
        )
        disease_types = list(disease_counts.keys())

    except Exception as e:
        print("Error loading dashboard:", e)
        uploads, disease_counts, total_images, disease_count, most_common_disease, disease_types = [], {}, 0, 0, "None", []
        disease_percentage = 0.0
        chart_data = []

    return render_template(
        "mais.html",
        uploads=uploads,
        disease_counts=disease_counts,
        total_images=total_images,
        disease_count=disease_count,
        disease_percentage=disease_percentage,
        chart_data=chart_data,
        most_common_disease=most_common_disease,
        disease_types=disease_types,
        model_loaded=tflite_interpreter is not None
    )

# -------- PREDICTION --------
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if tflite_interpreter is None:
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
    if tflite_interpreter is None:
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
    if tflite_interpreter is None:
        print("❌ TFLite model is not loaded, cannot make prediction")
        return "Model Error", "invalid_image.html", 0.0
    
    try:
        # Verify image file exists
        if not os.path.exists(img_path):
            print(f"❌ Image file not found: {img_path}")
            return "Invalid Image", "invalid_image.html", 0.0
            
        # Load and preprocess image
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

        # Make prediction with TFLite interpreter
        if is_quantized:
            # Quantize the input for quantized models
            img_array_quantized = (img_array / input_scale + input_zero_point).astype(np.int8)
            tflite_interpreter.set_tensor(input_details[0]['index'], img_array_quantized)
        else:
            # Use float32 for non-quantized models
            tflite_interpreter.set_tensor(input_details[0]['index'], img_array)
        
        tflite_interpreter.invoke()
        prediction = tflite_interpreter.get_tensor(output_details[0]['index'])
        
        # Dequantize output if model is quantized
        if is_quantized:
            prediction = (prediction.astype(np.float32) - output_zero_point) * output_scale
        
        pred_class = np.argmax(prediction)
        confidence = float(np.max(prediction))

        CONFIDENCE_THRESHOLD = 0.7
        if confidence < CONFIDENCE_THRESHOLD:
            return "Unknown Class", "invalid_image.html", confidence

        diseases = {
            0: ("Aphids", "aphids.html"),
            1: ("Armyworm", "armyworm.html"),
            2: ("Common Rust", "common_rust.html"),
            3: ("Common Smut", "common_smut.html"),
            4: ("Corn Borer", "corn_borer.html"),
            5: ("Earwig", "earwig.html"),
            6: ("Fusarium Ear Rot", "fusarium_ear_rot.html"),
            7: ("Gray Leaf Spot", "gray_leaf_spot.html"),
            8: ("Healthy Corn", "healthycorn.html"),
            9: ("Healthy Leaf", "healthyleaf.html"),
            10: ("Leaf Blight", "leaf_blight.html"),
            11: ("Leafhopper", "leafhopper.html"),
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
        'model_loaded': tflite_interpreter is not None,
        'model_type': 'TFLite',
        'model_quantized': is_quantized,
        'model_source': 'Google Drive Cloud',
        'timestamp': datetime.datetime.utcnow().isoformat()
    })

if __name__ == "__main__":
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get("PORT", 8080))
    
    # Debug information
    print(f"🚀 Starting Flask app on port {port}")
    print(f"📁 Current directory: {os.getcwd()}")
    print(f"📁 Files in directory: {os.listdir('.')}")
    print(f"✅ Model status: {'Loaded' if tflite_interpreter is not None else 'Not loaded'}")
    print(f"🌐 Model type: TFLite")
    print(f"🌐 Model quantized: {is_quantized}")
    print(f"🌐 Model source: Google Drive Cloud")
    
    app.run(debug=True, host="0.0.0.0", port=port, threaded=True)