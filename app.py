from flask import Flask, request, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the fitted scaler
with open('scaler.pkl', 'rb') as scaler_file:
    numerical_transformer = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form
    gender = int(request.form['gender'])
    age = float(request.form['age'])
    income = float(request.form['income'])
    time = float(request.form['time'])
    amount = float(request.form['amount'])
    reward = float(request.form['reward'])
    difficulty = float(request.form['difficulty'])
    duration = float(request.form['duration'])
    email = int(request.form['email'])
    mobile = int(request.form['mobile'])
    social = int(request.form['social'])
    web = int(request.form['web'])
    offer_type_bogo = int(request.form['offer_type_bogo'])
    offer_type_discount = int(request.form['offer_type_discount'])
    offer_type_informational = int(request.form['offer_type_informational'])

    # Create a DataFrame for the input
    input_data = pd.DataFrame([[gender, age, income, time, amount, reward, difficulty, duration, email, mobile, social, web, offer_type_bogo, offer_type_discount, offer_type_informational]],
                              columns=['gender', 'age', 'income', 'time', 'amount', 'reward', 'difficulty', 'duration', 'email', 'mobile', 'social', 'web', 'offer_type_bogo', 'offer_type_discount', 'offer_type_informational'])

    # Preprocess the numerical data
    numerical_cols = ['age', 'income', 'time', 'amount', 'reward', 'difficulty', 'duration']
    input_data_numerical = numerical_transformer.transform(input_data[numerical_cols])

    # Convert transformed numerical data back to DataFrame with appropriate column names
    input_data_numerical = pd.DataFrame(input_data_numerical, columns=numerical_cols, index=input_data.index)

    # Combine numerical and categorical data
    input_data_scaled = pd.concat([input_data_numerical, input_data.drop(columns=numerical_cols)], axis=1)

    # Make prediction
    prediction = model.predict(input_data_scaled)[0]

    # Map the prediction to a readable format
    event_mapping = {0: 'offer received', 1: 'offer viewed', 2: 'transaction', 3: 'offer completed'}
    prediction_text = event_mapping[prediction]

    return render_template('index.html', prediction_text=f'Predicted event: {prediction_text}')

if __name__ == '__main__':
    app.run(debug=True)
