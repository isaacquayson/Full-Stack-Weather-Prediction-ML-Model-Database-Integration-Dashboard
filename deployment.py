from flask import Flask, render_template, request, jsonify, session
import pickle
import mysql.connector
import numpy as np

# creating the application
app = Flask(__name__)

# loading the machine learning saved model and preprocessing objects
model = pickle.load(open('xgb_model.sav', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))  # Load the label encoder

# Database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',  # Replace with your MySQL username
    'password': 'Isaac%quayson2580',  # Replace with your MySQL password
    'database': 'weather'
}


# Function to save data to MySQL
def save_to_database(data):
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()

        # Create table if it doesn't exist
        create_table_query = """
        CREATE TABLE IF NOT EXISTS weather_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            temperature FLOAT,
            humidity FLOAT,
            wind_speed FLOAT,
            precipitation FLOAT,
            atmospheric_pressure FLOAT,
            uv_index FLOAT,
            visibility_km FLOAT,
            cloud_cover VARCHAR(20),
            season VARCHAR(20),
            location VARCHAR(20),
            prediction VARCHAR(50),
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        cursor.execute(create_table_query)

        # Insert data
        insert_query = """
        INSERT INTO weather_data 
        (temperature, humidity, wind_speed, precipitation, atmospheric_pressure, 
         uv_index, visibility_km, cloud_cover, season, location, prediction)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        cursor.execute(insert_query, (
            data['temperature'], data['humidity'], data['wind_speed'],
            data['precipitation'], data['atmospheric_pressure'], data['uv_index'],
            data['visibility_km'], data['cloud_cover'], data['season'],
            data['location'], data['prediction']
        ))

        connection.commit()
        cursor.close()
        connection.close()
        return True
    except Exception as e:
        print(f"Database error: {e}")
        return False


# creating the homepage
@app.route('/')
def home():
    result = ''
    return render_template('index.html', **locals())


# function to handle the prediction
@app.route('/predict', methods=['POST'])
def predict():
    # getting raw user inputs
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    wind_speed = float(request.form['wind_speed'])
    precipitation = float(request.form['precipitation'])
    atmospheric_pressure = float(request.form['atmospheric_pressure'])
    uv_index = float(request.form['uv_index'])
    visibility_km = float(request.form['visibility_km'])
    cloud_cover = request.form['cloud_cover'].lower()
    season = request.form['season'].lower()
    location = request.form['location'].lower()

    # Store original values for saving
    original_data = {
        'temperature': temperature,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'precipitation': precipitation,
        'atmospheric_pressure': atmospheric_pressure,
        'uv_index': uv_index,
        'visibility_km': visibility_km,
        'cloud_cover': cloud_cover,
        'season': season,
        'location': location
    }

    # --- Apply same preprocessing as training ---

    # Scale numeric
    numeric_features = [[temperature, humidity, wind_speed, precipitation,
                         atmospheric_pressure, uv_index, visibility_km]]
    scaled_numeric = scaler.transform(numeric_features)

    # Encode categorical
    categorical_features = [[cloud_cover, season, location]]
    encoded_categorical = encoder.transform(categorical_features)

    # Combine
    features = np.hstack([scaled_numeric, encoded_categorical])

    # --- Final prediction ---
    prediction_encoded = model.predict(features)[0]
    result = label_encoder.inverse_transform([prediction_encoded])[0]

    original_data['prediction'] = result

    # Store data in session for potential saving
    session['weather_data'] = original_data

    return render_template('index.html', result=result, form_data=original_data)


# Route to handle saving data
@app.route('/save_data', methods=['POST'])
def save_data():
    try:
        # Get data from session
        data = session.get('weather_data', {})

        if not data:
            return jsonify({'success': False, 'message': 'No data to save'})

        # Save to database
        success = save_to_database(data)

        if success:
            return jsonify({'success': True, 'message': 'Data saved successfully!'})
        else:
            return jsonify({'success': False, 'message': 'Error saving data to database'})

    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})


if __name__ == "__main__":
    app.secret_key = 'your_secret_key_here'  # Needed for session
    app.run(debug=True)