import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import random

df = pd.read_csv("genetic_disorders_dataset.csv")

# Preprocess data
label_encoders = {}
for column in ["Gender", "Family History", "Symptom1", "Symptom2", "Symptom3", "Genetic Disorder"]:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Train model
X = df.drop(["Genetic Disorder", "Precautions"], axis=1)
y = df["Genetic Disorder"]
model = DecisionTreeClassifier()
model.fit(X, y)

class GeneticDisorderPredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("Genetic Disorder Predictor")
        self.root.geometry("800x600")
        self.root.resizable(False, False)
        
        # Style
        style = ttk.Style()
        style.configure("TFrame", background="#f0f0f0")
        style.configure("TLabel", background="#f0f0f0", font=("Arial", 10))
        style.configure("TButton", font=("Arial", 10), padding=5)
        style.configure("TCombobox", padding=5)
        style.configure("TEntry", padding=5)
        
        # Main Frame
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        self.title_label = ttk.Label(self.main_frame, text="Genetic Disorder Prediction", font=("Arial", 16, "bold"))
        self.title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Patient Details Frame
        self.details_frame = ttk.LabelFrame(self.main_frame, text="Patient Details")
        self.details_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        # Age
        self.age_label = ttk.Label(self.details_frame, text="Age:")
        self.age_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.age_entry = ttk.Entry(self.details_frame)
        self.age_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Gender
        self.gender_label = ttk.Label(self.details_frame, text="Gender:")
        self.gender_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.gender_combo = ttk.Combobox(self.details_frame, values=["Male", "Female"], state="readonly")
        self.gender_combo.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        # Family History
        self.family_label = ttk.Label(self.details_frame, text="Family History of Genetic Disorders:")
        self.family_label.grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.family_combo = ttk.Combobox(self.details_frame, values=["Yes", "No"], state="readonly")
        self.family_combo.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        
        # Symptoms Frame
        self.symptoms_frame = ttk.LabelFrame(self.main_frame, text="Symptoms")
        self.symptoms_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        
        # Symptom 1
        self.symptom1_label = ttk.Label(self.symptoms_frame, text="Primary Symptom:")
        self.symptom1_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.symptom1_combo = ttk.Combobox(self.symptoms_frame, values=[
            "Persistent cough", "Frequent lung infections", "Poor growth", 
            "Pain crises", "Fatigue", "Swelling in hands/feet",
            "Uncontrolled movements", "Cognitive decline", "Emotional disturbances",
            "Flattened facial features", "Small head", "Poor muscle tone",
            "Excessive bleeding", "Easy bruising", "Joint pain",
            "Muscle weakness", "Loss of motor skills", "Seizures",
            "Developmental delays", "Musty body odor", "Skin rashes",
            "Tall slender build", "Long limbs", "Heart murmurs",
            "Difficulty walking", "Learning disabilities", "Facial bone deformities"
        ], state="readonly")
        self.symptom1_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Symptom 2
        self.symptom2_label = ttk.Label(self.symptoms_frame, text="Secondary Symptom:")
        self.symptom2_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.symptom2_combo = ttk.Combobox(self.symptoms_frame, values=[
            "Persistent cough", "Frequent lung infections", "Poor growth", 
            "Pain crises", "Fatigue", "Swelling in hands/feet",
            "Uncontrolled movements", "Cognitive decline", "Emotional disturbances",
            "Flattened facial features", "Small head", "Poor muscle tone",
            "Excessive bleeding", "Easy bruising", "Joint pain",
            "Muscle weakness", "Loss of motor skills", "Seizures",
            "Developmental delays", "Musty body odor", "Skin rashes",
            "Tall slender build", "Long limbs", "Heart murmurs",
            "Difficulty walking", "Learning disabilities", "Facial bone deformities", "None"
        ], state="readonly")
        self.symptom2_combo.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        # Symptom 3
        self.symptom3_label = ttk.Label(self.symptoms_frame, text="Tertiary Symptom:")
        self.symptom3_label.grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.symptom3_combo = ttk.Combobox(self.symptoms_frame, values=[
            "Persistent cough", "Frequent lung infections", "Poor growth", 
            "Pain crises", "Fatigue", "Swelling in hands/feet",
            "Uncontrolled movements", "Cognitive decline", "Emotional disturbances",
            "Flattened facial features", "Small head", "Poor muscle tone",
            "Excessive bleeding", "Easy bruising", "Joint pain",
            "Muscle weakness", "Loss of motor skills", "Seizures",
            "Developmental delays", "Musty body odor", "Skin rashes",
            "Tall slender build", "Long limbs", "Heart murmurs",
            "Difficulty walking", "Learning disabilities", "Facial bone deformities", "None"
        ], state="readonly")
        self.symptom3_combo.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        
        # Buttons
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.grid(row=3, column=0, pady=10)
        
        self.predict_button = ttk.Button(self.button_frame, text="Predict Disorder", command=self.predict_disorder)
        self.predict_button.grid(row=0, column=0, padx=5)
        
        self.clear_button = ttk.Button(self.button_frame, text="Clear", command=self.clear_fields)
        self.clear_button.grid(row=0, column=1, padx=5)
        
        # Results Frame
        self.results_frame = ttk.LabelFrame(self.main_frame, text="Prediction Results")
        self.results_frame.grid(row=1, column=1, rowspan=3, padx=10, pady=10, sticky="nsew")
        
        self.results_text = tk.Text(self.results_frame, wrap=tk.WORD, width=40, height=20, font=("Arial", 10))
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Configure grid weights
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(1, weight=1)
        self.main_frame.rowconfigure(2, weight=1)
        
    def predict_disorder(self):
        try:
            # Get input values
            age = int(self.age_entry.get())
            gender = self.gender_combo.get()
            family_history = self.family_combo.get()
            symptom1 = self.symptom1_combo.get()
            symptom2 = self.symptom2_combo.get() if self.symptom2_combo.get() != "None" else ""
            symptom3 = self.symptom3_combo.get() if self.symptom3_combo.get() != "None" else ""
            
            if not all([age, gender, family_history, symptom1]):
                messagebox.showerror("Error", "Please fill in all required fields")
                return
            
            # Encode inputs
            gender_encoded = label_encoders["Gender"].transform([gender])[0]
            family_encoded = label_encoders["Family History"].transform([family_history])[0]
            symptom1_encoded = label_encoders["Symptom1"].transform([symptom1])[0]
            symptom2_encoded = label_encoders["Symptom2"].transform([symptom2])[0] if symptom2 else 0
            symptom3_encoded = label_encoders["Symptom3"].transform([symptom3])[0] if symptom3 else 0
            
            # Make prediction
            input_data = [[age, gender_encoded, family_encoded, symptom1_encoded, symptom2_encoded, symptom3_encoded]]
            prediction_encoded = model.predict(input_data)[0]
            predicted_disorder = label_encoders["Genetic Disorder"].inverse_transform([prediction_encoded])[0]
            
            # Get precautions
            precautions = df[df["Genetic Disorder"] == prediction_encoded]["Precautions"].iloc[0]
            
            # Display results
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Predicted Genetic Disorder:\n")
            self.results_text.insert(tk.END, f"{predicted_disorder}\n\n", "bold")
            self.results_text.insert(tk.END, f"Precautions:\n")
            self.results_text.insert(tk.END, f"{precautions}\n\n")
            self.results_text.insert(tk.END, f"Patient Details:\n")
            self.results_text.insert(tk.END, f"Age: {age}\n")
            self.results_text.insert(tk.END, f"Gender: {gender}\n")
            self.results_text.insert(tk.END, f"Family History: {family_history}\n")
            self.results_text.insert(tk.END, f"Symptoms: {symptom1}, {symptom2}, {symptom3}\n")
            self.results_text.tag_configure("bold", font=("Arial", 10, "bold"))
            self.results_text.config(state=tk.DISABLED)
            
        except ValueError as e:
            messagebox.showerror("Error", "Please enter valid information in all fields")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def clear_fields(self):
        self.age_entry.delete(0, tk.END)
        self.gender_combo.set('')
        self.family_combo.set('')
        self.symptom1_combo.set('')
        self.symptom2_combo.set('')
        self.symptom3_combo.set('')
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = GeneticDisorderPredictor(root)
    root.mainloop()