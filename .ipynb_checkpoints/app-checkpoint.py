
from flask import Flask, render_template,request
 
import numpy as np
import os

import mysql.connector
 
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

#load model
filepath = 'mais_model.keras'
model =load_model(filepath)
print(model)
print('@@ Model Loaded Successfully')
 
 
def pred_corn_disease(corn_leaf):
  test_image = load_img(corn_leaf, target_size = (150, 150)) # load image 
  print("@@ Got Image for prediction")
   
  test_image = img_to_array(test_image)/255 # convert image to np array and normalize
  test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
   
  result = model.predict(test_image).round(3) # predict diseased palnt or not
  print('@@ Raw result = ', result)
   
  pred = np.argmax(result, axis=1) # get the index of max value
  print(pred)
  if pred == 0:
    return "Blight Disease", 'blight.html'
  elif pred == 1:
      return "Common Rust Disease", 'commonrust.html'
  elif pred == 2:
      return "Leaf Spot Disease", 'leafspot.html'
  elif pred == 3:
    return "Healthy Corn Leaf", 'healthy.html'

def create_connection():
    """ Create a connection to MySQL database """
    return mysql.connector.connect(
        host="localhost",
        user="maiscan",
        password="maiscanMAISCAN2024",
        database="maiscan"
    )

def create_table(conn):
    """ Create a table to store uploaded images """
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS UploadedImages
                     (id INT AUTO_INCREMENT PRIMARY KEY,
                     filename VARCHAR(255) NOT NULL,
                     upload_image BLOB,
                     upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
 
app = Flask(__name__)
 
# render index.html page
@app.route("/",methods=['GET','POST'])
def home():
        return render_template('mais.html')     
  
# get input image from client then predict class and render respective .html page for solution
@app.route("/predict", methods = ['GET','POST'])
def predict():
     if request.method == 'POST':
        file = request.files['image'] # fet input
        filename = file.filename        
        print("@@ Input posted = ", filename)
         
        file_path = os.path.join('static/user_image', filename)
        file.save(file_path)

        # Read the image file as binary data
        with open(file_path, "rb") as f:
            upload_image = f.read()
            
        # Database operations
        conn = create_connection()
        create_table(conn)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO UploadedImages (filename, upload_image) VALUES (%s, %s)", (filename, upload_image))
        conn.commit()
        conn.close()

 
        print("@@ Predicting class......")
        pred, output_page = pred_corn_disease(corn_leaf=file_path)               
        
    
        return render_template(output_page, pred_output = pred, user_image = file_path)
     
# For local system & cloud
if __name__ == "__main__":
    app.run(threaded = False, port=8080) 
