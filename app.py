from flask import Flask, request, jsonify
import pandas as pd
import joblib
from datetime import datetime
import os # נוסיף את זה כדי לוודא נתיבים

app = Flask(__name__)

# נתיב שבו Render מאחסן את הקבצים שלך (התיקייה הנוכחית של הקוד)
MODEL_DIR = os.getcwd() 

try:
    # טען את המודלים והפיצ'רים
    reg_model = joblib.load(os.path.join(MODEL_DIR, 'parking_spots_predictor_reg_model.pkl'))
    clf_model = joblib.load(os.path.join(MODEL_DIR, 'parking_predictor_model.pkl'))
    # חשוב: ודאי ששם הקובץ של רשימת הפיצ'רים של הרגרסיה הוא זהה למה שיצרת בשלב 1!
    # הוא נראה 'parking_spots_reg_model_features_list.pkl' אצלי
    reg_features = joblib.load(os.path.join(MODEL_DIR, 'parking_spots_reg_model_features_list.pkl'))
    clf_features = joblib.load(os.path.join(MODEL_DIR, 'main_clf_model_features_list.pkl'))
    
    # טען את האנקודרים
    day_encoder = joblib.load(os.path.join(MODEL_DIR, 'day_of_week_encoder.pkl'))
    time_encoder = joblib.load(os.path.join(MODEL_DIR, 'time_of_day_encoder.pkl'))
    # אין צורך ב-city_encoder ו-parking_encoder אם את בונה את הפיצ'רים ידנית עם prefix כמו 'עיר_'
    #city_encoder = joblib.load(os.path.join(MODEL_DIR, 'city_encoder.pkl'))
    #parking_encoder = joblib.load(os.path.join(MODEL_DIR, 'parking_name_encoder.pkl'))

    # טען את טבלת פרטי החניונים
    static_data = pd.read_csv(os.path.join(MODEL_DIR, 'parking_static_data.csv'))
    print("Models, features, encoders, and static data loaded successfully.")
except Exception as e:
    print(f"Error loading files: {e}")
    # אם יש שגיאת טעינה, סגור את האפליקציה כדי למנוע טעויות בהמשך
    exit(1) # זה יגרום ל-Render להראות שגיאה, וזה טוב כדי לזהות בעיות


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

    encoded_day = day_encoder.transform([[weekday]])[0][0] # שינוי קל כאן
    encoded_time = time_encoder.transform([[time_of_day]])[0][0] # שינוי קל כאן

    return encoded_day, encoded_time

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print(f"Received data: {data}") # לצורך דיבוג

        city = data.get('city')
        parking_name = data.get('parking_name')

        if not city or not parking_name:
            return jsonify({'error': 'יש לספק עיר ושם חניון.'}), 400

        # שליפת פרטי חניון מתוך טבלת המידע
        row = static_data[(static_data['city'] == city) & (static_data['parking_name'] == parking_name)]
        if row.empty:
            return jsonify({'error': 'חניון לא נמצא בטבלת הנתונים. ודא/י שהשם והעיר מדויקים.'}), 400

        total_spots = row.iloc[0]['total_spots']
        cost_status = row.iloc[0]['cost_status']
        parking_type = row.iloc[0]['parking_type'] # לא בשימוש ישיר במודלים אבל טוב לשלוף

        # שליפת זמן נוכחי
        encoded_day, encoded_time = get_time_features()

        # --- בניית הפיצ'רים למודל הרגרסיה בצורה בטוחה ---
        # אתחול מילון עם כל הפיצ'רים של הרגרסיה וערך 0
        reg_input_dict = {col: 0 for col in reg_features}

        # עדכון הערכים הישירים
        reg_input_dict['encoded_day_of_week'] = encoded_day
        reg_input_dict['encoded_time_of_day'] = encoded_time
        reg_input_dict['combined_parking_cost_status'] = cost_status
        reg_input_dict['encoded_abnormal_parking'] = 0 # ברירת מחדל

        # עדכון פיצ'רי One-Hot Encoding עבור העיר והחניון הנבחרים
        # ודא שהעמודות קיימות ברשימת הפיצ'רים של המודל
        city_col_name = 'עיר_' + city
        if city_col_name in reg_input_dict:
            reg_input_dict[city_col_name] = 1

        parking_col_name = 'שם_חניה_' + parking_name
        if parking_col_name in reg_input_dict:
            reg_input_dict[parking_col_name] = 1
        
        # יצירת DataFrame מהמילון, עם סדר העמודות הנכון
        # חשוב: ודא ש-reg_features מכיל את כל העמודות ושמותיהן תואמים
        reg_df = pd.DataFrame([reg_input_dict], columns=reg_features)
        
        # הדפסה לדיבוג - ודא שזה נראה כמו שצריך
        print(f"Regression input DataFrame: \n{reg_df}")


        predicted_available_spots = max(0, round(reg_model.predict(reg_df)[0]))
        predicted_available_spots = min(predicted_available_spots, total_spots)

        # --- בניית הפיצ'רים למודל הסיווג בצורה בטוחה ---
        # אתחול מילון עם כל הפיצ'רים של הסיווג וערך 0
        clf_input_dict = {col: 0 for col in clf_features}

        # העתקת פיצ'רים קיימים מהרגרסיה או עדכון ישיר
        clf_input_dict.update({
            'encoded_day_of_week': encoded_day,
            'encoded_time_of_day': encoded_time,
            'combined_parking_cost_status': cost_status,
            'encoded_abnormal_parking': 0, # ברירת מחדל
            'parking_spots_available_current': predicted_available_spots, # זהו החיזוי מהרגרסיה
            'duration_minutes': 30, # ערך קבוע כרגע, אפשר להפוך לקלט
            'partial_duration_info': 0, # ערך קבוע כרגע, אפשר להפוך לקלט
            'is_short_duration_no_spot': 0 # ערך קבוע כרגע, אפשר להפוך לקלט
        })

        # עדכון פיצ'רי One-Hot Encoding עבור העיר והחניון הנבחרים
        if city_col_name in clf_input_dict:
            clf_input_dict[city_col_name] = 1
        if parking_col_name in clf_input_dict:
            clf_input_dict[parking_col_name] = 1

        # יצירת DataFrame מהמילון, עם סדר העמודות הנכון
        clf_df = pd.DataFrame([clf_input_dict], columns=clf_features)

        # הדפסה לדיבוג - ודא שזה נראה כמו שצריך
        print(f"Classification input DataFrame: \n{clf_df}")

        prediction = clf_model.predict(clf_df)[0]
        proba = clf_model.predict_proba(clf_df)[0].tolist()

        return jsonify({
            'predicted_available_spots': int(predicted_available_spots),
            'predicted_is_available': int(prediction),
            'probability_available': round(proba[1], 2),
            'probability_not_available': round(proba[0], 2)
        })

    except Exception as e:
        print(f"An error occurred during prediction: {e}") # הדפסת השגיאה המלאה
        return jsonify({'error': str(e), 'message': 'שגיאה פנימית בשרת'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) # חשוב ל-Render
