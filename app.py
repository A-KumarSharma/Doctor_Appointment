# app.py
from flask import Flask, request, send_file
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
from datetime import datetime
import io

app = Flask(__name__)

# Load and preprocess data
def load_and_preprocess_data():
    # Read data from Excel file
    df = pd.read_excel('dummy_npi_data.xlsx')
    
    # Convert time strings to minutes
    def time_to_minutes(t):
        try:
            if isinstance(t, str):
                h, m = map(int, t.split(':'))
                return h * 60 + m
            elif isinstance(t, datetime):
                return t.hour * 60 + t.minute
            else:
                return 0
        except:
            return 0
    
    # Ensure required columns exist, if not create dummy ones
    required_columns = ['NPI', 'Speciality', 'Region', 'LoginTime', 'LogoutTime', 'TimeSpent', 'CountAttempts']
    for col in required_columns:
        if col not in df.columns:
            if col in ['TimeSpent', 'CountAttempts']:
                df[col] = np.random.randint(1, 10, df.shape[0])  # Random numbers for demo
            elif col in ['LoginTime', 'LogoutTime']:
                df[col] = '09:00' if col == 'LoginTime' else '17:00'  # Default times
            else:
                df[col] = 'Unknown'  # Default text
    
    df['LoginMinutes'] = df['LoginTime'].apply(time_to_minutes)
    df['LogoutMinutes'] = df['LogoutTime'].apply(time_to_minutes)
    
    # Encode categorical variables
    le_speciality = LabelEncoder()
    le_region = LabelEncoder()
    df['Speciality_Encoded'] = le_speciality.fit_transform(df['Speciality'])
    df['Region_Encoded'] = le_region.fit_transform(df['Region'])
    
    return df

# Train model
def train_model(df):
    # Features for prediction
    features = ['Speciality_Encoded', 'Region_Encoded', 'LoginMinutes', 
                'LogoutMinutes', 'TimeSpent', 'CountAttempts']
    
    # For this demo, creating a dummy target variable
    # In real scenario, this would come from actual survey attendance data
    df['WillAttend'] = np.random.randint(0, 2, df.shape[0])
    
    X = df[features]
    y = df['WillAttend']
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, features

# Initialize data and model
df = load_and_preprocess_data()
model, features = train_model(df)

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Doctor Survey Prediction</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 600px; margin: auto; }
            input, button { margin: 10px 0; padding: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Doctor Survey Prediction</h2>
            <form action="/predict" method="post">
                <label>Enter Time (HH:MM):</label><br>
                <input type="text" name="time" placeholder="HH:MM" required><br>
                <button type="submit">Generate CSV</button>
            </form>
        </div>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input time
        input_time = request.form['time']
        input_minutes = sum(int(x) * 60 ** i for i, x in enumerate(reversed(input_time.split(':'))))
        
        # Prepare data for prediction
        df['InputTimeDiff'] = abs(df['LoginMinutes'] - input_minutes)
        prediction_data = df[features]
        
        # Make predictions
        probabilities = model.predict_proba(prediction_data)[:, 1]
        df['AttendanceProbability'] = probabilities
        
        # Select top doctors (e.g., top 50%)
        threshold = np.percentile(probabilities, 50)
        selected_doctors = df[df['AttendanceProbability'] >= threshold][['NPI', 'Speciality', 'Region']]
        
        # Create CSV
        output = io.StringIO()
        selected_doctors.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'doctors_to_contact_{input_time.replace(":", "")}.csv'
        )
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)