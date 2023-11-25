import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Cargar el conjunto de datos
data = pd.read_csv("datos_procesados.csv")
# Separar características y etiquetas
X = data.drop("NObeyesdad", axis=1)
y = data["NObeyesdad"]
# Codificar etiquetas en formato one-hot
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_one_hot = tf.keras.utils.to_categorical(y_encoded)

# Dividir datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=22)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


funcion_activacion = tf.keras.layers.LeakyReLU(alpha=0.025)

# Crear el modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=27, activation=funcion_activacion, input_shape=(27,)),
    tf.keras.layers.Dense(units=128, activation=funcion_activacion), 
    tf.keras.layers.Dense(units=7, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer=(tf.keras.optimizers.Adam(learning_rate=0.0095)) , loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X_train_scaled, y_train, epochs=20, batch_size=64, validation_data=(X_test_scaled, y_test))
# Graficar la precisión durante el entrenamiento y la validación
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
# Graficar la pérdida durante el entrenamiento y la validación
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
# Cargar el modelo
model = tf.keras.models.load_model("modelo_entrenado")

# Hacer predicciones en datos de prueba
predictions = model.predict(X_test_scaled)
predicted_classes = tf.argmax(predictions, axis=1)
true_classes = tf.argmax(y_test, axis=1)
correct_predictions = tf.equal(predicted_classes, true_classes)
# Calcular la precisión en los datos de prueba
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
print(f"Precisión en datos de prueba: {accuracy.numpy()}")
# Guardar el modelo
model.save("modelo_entrenado")

# Obtener índices de instancias clasificadas incorrectamente
incorrect_indices = np.where(np.argmax(y_test, axis=1) != np.argmax(predictions, axis=1))[0]

# Visualizar algunas instancias incorrectamente clasificadas
for idx in incorrect_indices[:]:
    predicted_label = label_encoder.inverse_transform([np.argmax(predictions[idx])])[0]
    true_label = label_encoder.inverse_transform([np.argmax(y_test[idx])])[0]
    
    print(f"Instancia {idx}: Predicho: {predicted_label}, Real: {true_label}")
