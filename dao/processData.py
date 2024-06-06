import pickle
import time
import pandas as pd

def process_input_data(age, gender, symptoms):
    symptom_list = ['Fever', 'Cough', 'Headache', 'Sore Throat', 'Runny Nose', 'Fatigue', 'Chills', 'Body Aches',
                    'Shortness of Breath', 'Chest Pain', 'Nausea', 'Vomiting', 'Diarrhea', 'Rash', 'Joint Pain',
                    'Abdominal Pain', 'Polydipsia', 'Weight Loss', 'Frequent Urination', 'Dizziness',
                    'Difficulty swallowing', 'Hoarse voice']

    symptoms_encoded = [
        1 if symptom in symptoms else 0 for symptom in symptom_list]
    patient_data = symptoms_encoded + [age, gender]

    with open('model.pkl', 'rb') as model_file:
        rf_classifier = pickle.load(model_file)
        onehot_encoder = pickle.load(model_file)
        label_encoders = pickle.load(model_file)
        accuracy = pickle.load(model_file)

    new_patient_data = pd.DataFrame(
        [patient_data], columns=symptom_list + ['Age', 'Gender'])

    for col in ['Gender']:
        new_patient_data[col] = label_encoders[col].transform(
            new_patient_data[col])

    new_patient_encoded = onehot_encoder.transform(new_patient_data)

    start_time = time.time()
    new_patient_prediction = rf_classifier.predict(new_patient_encoded)
    end_time = time.time()

    prediction_time = end_time - start_time
    return new_patient_prediction[0], accuracy, prediction_time
