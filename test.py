import os

file_path = "trained_plant_disease_model.keras"  # Update if your model is in a different location
print("File exists:", os.path.exists(file_path))

# If it's in a subfolder like 'models/', try:
# file_path = "models/trained_plant_disease_model.keras"
