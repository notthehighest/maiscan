from tensorflow import keras

model = keras.models.load_model("maiscan_disease_model_final.keras")
model.save("maiscan_disease_model_final.h5")