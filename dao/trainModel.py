from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle

def train_model():
    data = pd.read_csv('./datasets/common-diseases.csv')
    X = data.drop('Disease', axis=1)
    y = data['Disease']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    categorical_cols = ['Gender']
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
        label_encoders[col] = le

    onehot_encoder = OneHotEncoder(handle_unknown='ignore')
    X_train_encoded = onehot_encoder.fit_transform(X_train)
    X_test_encoded = onehot_encoder.transform(X_test)

    rf_classifier = RandomForestClassifier(n_jobs=2, random_state=42)
    rf_classifier.fit(X_train_encoded, y_train)

    y_pred = rf_classifier.predict(X_test_encoded)
    accuracy = accuracy_score(y_test, y_pred) * 100

    with open('model.pkl', 'wb') as model_file:
        pickle.dump(rf_classifier, model_file)
        pickle.dump(onehot_encoder, model_file)
        pickle.dump(label_encoders, model_file)
        pickle.dump(accuracy, model_file)
