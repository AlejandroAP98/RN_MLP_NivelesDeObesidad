import pandas as pd

# Cargar el conjunto de datos
data = pd.read_csv("data.csv")

# Categorización de Datos Categóricos
categorical_columns = ["Gender", "family_history_with_overweight", "FAVC", "CAEC", "SMOKE", "SCC", "MTRANS"]
data_categorical = pd.get_dummies(data[categorical_columns])

# Normalización de Datos Numéricos
numeric_columns = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
data_numeric = (data[numeric_columns] - data[numeric_columns].min()) / (data[numeric_columns].max() - data[numeric_columns].min())

# Mapeo de la Variable Objetivo
target_mapping = {"Insufficient_Weight": 0, "Normal_Weight": 1, "Overweight_Level_I": 2, "Overweight_Level_II": 3, "Obesity_Type_I": 4, "Obesity_Type_II": 5, "Obesity_Type_III": 6}
data["NObeyesdad"] = data["NObeyesdad"].map(target_mapping)

# Concatenar los nuevos conjuntos de datos
processed_data = pd.concat([data_categorical, data_numeric, data["NObeyesdad"]], axis=1)

# Puedes guardar el conjunto de datos procesado si lo deseas
processed_data.to_csv("datos_procesados.csv", index=False)
