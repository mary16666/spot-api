
from flask import Flask, request, jsonify
import pandas as pd
import joblib
from datetime import datetime

app = Flask(__name__)

# טען את המודלים והפיצ'רים
reg_model = joblib.load('parking_spots_predictor_reg_model.pkl')
clf_model = joblib.load('parking_predictor_model.pkl')
reg_features = joblib.load('parking_spots_reg_model_features_list_corrected.pkl')
clf_features = joblib.load('main_clf_model_features_list.pkl')

# טען את האנקודרים
day_encoder = joblib.load('day_of_week_encoder.pkl')
time_encoder = joblib.load('time_of_day_encoder.pkl')
city_encoder = joblib.load('city_encoder.pkl')
parking_encoder = joblib.load('parking_name_encoder.pkl')

# טען את טבלת פרטי החניונים
static_data = pd.read_csv('parking_static_data.csv')

def get_time_features():
    now = datetime.now()
    hour = now.hour
    weekday = now.weekday()  # 0=Monday, ..., 6=Sunday

    # קידוד זמן לקטגוריה: בוקר, צהריים, ערב
    if 6 <= hour < 12:
        time_of_day = 'morning'
    elif 12 <= hour < 18:
        time_of_day = 'afternoon'
    else:
        time_of_day = 'evening'

    encoded_day = day_encoder.transform([weekday])[0]
    encoded_time = time_encoder.transform([time_of_day])[0]

    return encoded_day, encoded_time

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        city = data.get('city')
        parking_name = data.get('parking_name')

        # שליפת פרטי חניון מתוך טבלת המידע
        row = static_data[(static_data['city'] == city) & (static_data['parking_name'] == parking_name)]
        if row.empty:
            return jsonify({'error': 'חניון לא נמצא בטבלת הנתונים'}), 400

        total_spots = row.iloc[0]['total_spots']
        cost_status = row.iloc[0]['cost_status']
        parking_type = row.iloc[0]['parking_type']

        # שליפת זמן נוכחי
        encoded_day, encoded_time = get_time_features()

        # יצירת מילון עם כל הפיצ'רים לרגרסיה
        reg_input = {
            'encoded_day_of_week': encoded_day,
            'encoded_time_of_day': encoded_time,
            'combined_parking_cost_status': cost_status,
            'encoded_abnormal_parking': 0,  # ברירת מחדל
            'עיר_' + city: 1,
            'שם_חניה_' + parking_name: 1
        }
        for col in reg_features:
            if col not in reg_input:
                reg_input[col] = 0

        reg_df = pd.DataFrame([reg_input])[reg_features]
        predicted_available_spots = max(0, round(reg_model.predict(reg_df)[0]))
        predicted_available_spots = min(predicted_available_spots, total_spots)

        # הכנה לסיווג
        clf_input = reg_input.copy()
        clf_input.update({
            'parking_spots_available_current': predicted_available_spots,
            'duration_minutes': 30,
            'partial_duration_info': 0,
            'is_short_duration_no_spot': 0
        })
        for col in clf_features:
            if col not in clf_input:
                clf_input[col] = 0

        clf_df = pd.DataFrame([clf_input])[clf_features]
        prediction = clf_model.predict(clf_df)[0]
        proba = clf_model.predict_proba(clf_df)[0].tolist()

        return jsonify({
            'predicted_available_spots': int(predicted_available_spots),
            'predicted_is_available': int(prediction),
            'probability_available': round(proba[1], 2),
            'probability_not_available': round(proba[0], 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
