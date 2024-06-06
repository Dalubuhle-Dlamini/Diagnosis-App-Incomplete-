from flask import Flask, render_template, request, url_for, redirect
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score
import pickle

from dao.processData import process_input_data

app = Flask(__name__)

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


train_model()

@app.route('/load')
def loading():
    return render_template('loading.html')

@app.route('/')
def index():
    return render_template('index.html')


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/extra")
def extra():
    return render_template("extra.html")


@app.route("/support")
def support():
    return render_template("support.html")


@app.route("/diagnosis")
def diagnose():
    return render_template("what-diagnosis.html")


@app.route("/common-diagnosis")
def common_diagnosis():
    return render_template("common-diagnosis.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        gender = request.form['gender']
        symptoms = request.form.keys()
    else:
        age = int(request.args.get('age'))
        gender = request.args.get('gender')
        symptoms = request.args.keys()


    symptoms = list(symptoms)
    symptoms.remove('age')
    symptoms.remove('gender')

    prediction, accuracy, prediction_time = process_input_data(
        age, gender, symptoms)

    remedies_df = pd.read_csv('./datasets/treatment.csv')
    disease_remedies = remedies_df[remedies_df['Disease'] == prediction]

    details_df = pd.read_csv('./datasets/details.csv')
    disease_details = details_df[details_df['Disease'] == prediction]

    symptoms_df = pd.read_csv('./datasets/symptoms.csv')
    disease_symptoms = symptoms_df[symptoms_df['Disease'] == prediction]

    images_df = pd.read_csv('./datasets/images.csv')
    disease_images = images_df[images_df['Disease'] == prediction]

    return render_template('result.html',
                           prediction=prediction,
                           accuracy=accuracy,
                           prediction_time=prediction_time,
                           symptoms=symptoms,
                           remedies=disease_remedies.to_dict(orient='records'),
                           details=disease_details.to_dict(orient='records'),
                           dsymptoms=disease_symptoms.to_dict(orient='records'),
                           images=disease_images.to_dict(orient='records'))



# error handlers
@app.errorhandler(400)
def bad_upload(e):
    return render_template('./error/400.html'), 400


@app.errorhandler(404)
def page_not_found(e):
    return render_template('./error/404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('./error/500.html'), 500


@app.errorhandler(502)
def bad_gateway(e):
    return render_template('./error/502.html'), 502


@app.errorhandler(503)
def service_unavailable(e):
    return render_template('./error/503.html'), 503

if __name__ == '__main__':
    app.run(debug=True)
