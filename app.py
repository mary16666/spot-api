from flask import Flask, request, jsonify
import pandas as pd
import joblib
from datetime import datetime
import os
from sklearn.preprocessing import LabelEncoder # חשוב לוודא שזה מיובא אם משתמשים בו


app = Flask(__name__)

# נתיב שבו Render מאחסן את הקבצים שלך (התיקייה הנוכחית של הקוד)
MODEL_DIR = os.getcwd() 

try:
    # טען את המודלים ורשימות הפיצ'רים שלהם
    # ודא ששם קובץ רשימת הפיצ'רים של הרגרסיה הוא 'parking_spots_reg_model_features_list.pkl'
    # ולא הגרסה עם '_corrected' (אלא אם כן זו הגרסה היחידה שנשארת לך)
    reg_model = joblib.load(os.path.join(MODEL_DIR, 'parking_spots_predictor_reg_model.pkl'))
    clf_model = joblib.load(os.path.join(MODEL_DIR, 'parking_predictor_model.pkl'))
    reg_features = joblib.load(os.path.join(MODEL_DIR, 'parking_spots_reg_model_features_list.pkl'))
    clf_features = joblib.load(os.path.join(MODEL_DIR, 'main_clf_model_features_list.pkl'))
    
    # טען את האנקודרים (רק את אלה שאכן שמרת בקולאב)
    # אלו אנקודרים מסוג LabelEncoder עבור פיצ'רים שקודדו כך (יום בשבוע ושעת היום)
    day_encoder = joblib.load(os.path.join(MODEL_DIR, 'day_of_week_encoder.pkl'))
    time_encoder = joblib.load(os.path.join(MODEL_DIR, 'time_of_day_encoder.pkl'))
    
    # אין צורך לטעון city_encoder ו-parking_encoder אם השתמשת ב-pd.get_dummies()
    # עבור קידוד One-Hot של עיר ושם חניה. הם נוצרים באופן דינמי בבניית הפיצ'רים.

    # טען את טבלת פרטי החניונים הסטטיים
    static_data = pd.read_csv(os.path.join(MODEL_DIR, 'parking_static_data.csv'))
    
    print("Models, features, encoders, and static data loaded successfully.")
except Exception as e:
    # הדפס שגיאת טעינה מלאה ללוגים של Render
    print(f"Error loading files: {e}") 
    # אם יש שגיאת טעינה קריטית, צא מהאפליקציה. זה יופיע ככשלון ב-Render logs.
    exit(1)


def get_time_features():
    """
    פונקציה זו מחשבת את הפיצ'רים הקשורים לזמן (יום בשבוע ושעה ביום נוכחיים)
    ומקודדת אותם באמצעות האנקודרים שנטענו.
    """
    now = datetime.now()
    hour = now.hour
    weekday = now.weekday()  # 0=Monday (שני), ..., 6=Sunday (ראשון)

    # קידוד שעת היום לקטגוריה: בוקר, צהריים, ערב
    if 6 <= hour < 12:
        time_of_day_category = 'morning'
    elif 12 <= hour < 18:
        time_of_day_category = 'afternoon'
    else:
        time_of_day_category = 'evening'

    # קידוד יום בשבוע ושעת היום באמצעות LabelEncoders
    # שימו לב לפורמט: transform([[value]])[0][0] עבור ערך בודד
    try:
        encoded_day = day_encoder.transform([[weekday]])[0][0]
    except ValueError as e:
        # טיפול במקרה של ערך שלא נראה ע"י האנקודר באימון
        print(f"Warning: day_encoder could not transform weekday {weekday}. Error: {e}")
        # במקרה כזה, אפשר להחזיר ערך ברירת מחדל או לטפל אחרת (לדוגמה, 0 אם הוא לא קיים)
        encoded_day = 0 # הגדר ערך ברירת מחדל בטוח אם לא נמצא
        
    try:
        encoded_time = time_encoder.transform([[time_of_day_category]])[0][0]
    except ValueError as e:
        # טיפול במקרה של ערך שלא נראה ע"י האנקודר באימון
        print(f"Warning: time_encoder could not transform time_of_day_category {time_of_day_category}. Error: {e}")
        encoded_time = 0 # הגדר ערך ברירת מחדל בטוח אם לא נמצא

    return encoded_day, encoded_time


@app.route('/predict', methods=['POST'])
def predict():
    """
    נקודת קצה ל-API לקבלת קלט מהמשתמש וביצוע חיזוי דו-שלבי.
    """
    try:
        data = request.get_json()
        print(f"Received data: {data}") # הדפסת הקלט ללוגים לצורך דיבוג

        city = data.get('city')
        parking_name = data.get('parking_name')

        if not city or not parking_name:
            return jsonify({'error': 'יש לספק עיר ושם חניון.'}), 400

        # שליפת פרטי חניון מתוך טבלת המידע הסטטי
        # חשוב: ודא ששמות העמודות כאן ('city', 'parking_name') תואמים בדיוק ל-CSV
        row = static_data[(static_data['city'] == city) & (static_data['parking_name'] == parking_name)]
        if row.empty:
            return jsonify({'error': 'חניון לא נמצא בטבלת הנתונים הסטטיים. ודא/י שהשם והעיר מדויקים.'}), 400

        # *** תיקון כאן: שינוי מ-'total_spots' ל-'total_parking_spots' ***
        total_parking_spots = row.iloc[0]['total_parking_spots'] 
        # *** תיקון כאן: שינוי שם העמודה ל-combined_parking_cost_status ***
        cost_status = row.iloc[0]['combined_parking_cost_status'] 
        # parking_type_encoded = row.iloc[0]['parking_type_encoded'] # ניתן לשלוף אם יש צורך בהמשך

        # שליפת פיצ'רי זמן נוכחי
        encoded_day, encoded_time = get_time_features()

        # --- בניית הפיצ'רים למודל הרגרסיה (שלב 1) בצורה בטוחה ---
        # אתחול מילון עם כל הפיצ'רים של הרגרסיה וערך 0.
        # זה מבטיח שכל העמודות הדרושות למודל יהיו קיימות, גם אם לא הועבר להן ערך ספציפי בקלט.
        reg_input_dict = {col: 0 for col in reg_features}

        # עדכון ערכים לפיצ'רים שהם לא One-Hot Encoding
        reg_input_dict['encoded_day_of_week'] = encoded_day
        reg_input_dict['encoded_time_of_day'] = encoded_time
        reg_input_dict['combined_parking_cost_status'] = cost_status
        reg_input_dict['encoded_abnormal_parking'] = 0 # ברירת מחדל, אם זה לא מגיע מהקלט

        # עדכון פיצ'רי One-Hot Encoding עבור העיר והחניון הנבחרים
        # ודא שהעמודות קיימות ברשימת הפיצ'רים של המודל לפני עדכון ערכן.
        city_col_name = 'עיר_' + city
        if city_col_name in reg_input_dict:
            reg_input_dict[city_col_name] = 1

        parking_col_name = 'שם_חניה_' + parking_name
        if parking_col_name in reg_input_dict:
            reg_input_dict[parking_col_name] = 1
        
        # יצירת DataFrame ממילון הפיצ'רים, עם סדר העמודות הנכון כפי שנטען מ-reg_features
        reg_df = pd.DataFrame([reg_input_dict], columns=reg_features)
        
        # הדפסה לדיבוג - ודא שזה נראה כמו שצריך בלוגים של Render
        print(f"Regression input DataFrame (first 5 features): \n{reg_df.iloc[:, :5]}")
        print(f"Regression input DataFrame (last 5 features): \n{reg_df.iloc[:, -5:]}")

        # חיזוי מספר המקומות הפנויים (מודל הרגרסיה)
        predicted_available_spots = reg_model.predict(reg_df)[0]
        predicted_available_spots = max(0, round(predicted_available_spots)) # ודא שהמספר לא שלילי ועגל אותו
        # *** תיקון כאן: שינוי מ-'total_spots' ל-'total_parking_spots' ***
        predicted_available_spots = min(predicted_available_spots, total_parking_spots) 

        # --- בניית הפיצ'רים למודל הסיווג (שלב 2) בצורה בטוחה ---
        # אתחול מילון עם כל הפיצ'רים של הסיווג וערך 0
        clf_input_dict = {col: 0 for col in clf_features}

        # העתקת פיצ'רים קיימים מהרגרסיה או עדכון ישיר
        clf_input_dict.update({
            'encoded_day_of_week': encoded_day,
            'encoded_time_of_day': encoded_time,
            'combined_parking_cost_status': cost_status, # עדכון ה-cost_status המתוקן
            'encoded_abnormal_parking': 0, # ברירת מחדל
            'parking_spots_available_current': predicted_available_spots, # זהו החיזוי מהרגרסיה!
            'duration_minutes': data.get('duration_minutes', 30), # קבל מקלט או ברירת מחדל
            'partial_duration_info': data.get('partial_duration_info', 0), # קבל מקלט או ברירת מחדל
            'is_short_duration_no_spot': data.get('is_short_duration_no_spot', 0) # קבל מקלט או ברירת מחדל
        })

        # עדכון פיצ'רי One-Hot Encoding עבור העיר והחניון הנבחרים
        if city_col_name in clf_input_dict:
            clf_input_dict[city_col_name] = 1
        if parking_col_name in clf_input_dict:
            clf_input_dict[parking_col_name] = 1

        # יצירת DataFrame ממילון הפיצ'רים, עם סדר העמודות הנכון כפי שנטען מ-clf_features
        clf_df = pd.DataFrame([clf_input_dict], columns=clf_features)

        # הדפסה לדיבוג - ודא שזה נראה כמו שצריך בלוגים של Render
        print(f"Classification input DataFrame (first 5 features): \n{clf_df.iloc[:, :5]}")
        print(f"Classification input DataFrame (last 5 features): \n{clf_df.iloc[:, -5:]}")

        # חיזוי זמינות החניה (מודל הסיווג)
        prediction = clf_model.predict(clf_df)[0]
        proba = clf_model.predict_proba(clf_df)[0].tolist()

        # החזרת התוצאות ל-Base44
        return jsonify({
            'predicted_available_spots': int(predicted_available_spots),
            'predicted_is_available': int(prediction),
            'probability_available': round(proba[1], 2),
            'probability_not_available': round(proba[0], 2),
            'message': 'Prediction successful'
        })

    except Exception as e:
        # במקרה של שגיאה, הדפס את השגיאה המלאה ללוגים והחזר הודעה ברורה
        print(f"An error occurred during prediction: {e}")
        return jsonify({'error': str(e), 'message': 'שגיאה פנימית בשרת'}), 500

if __name__ == '__main__':
    # הרצה מקומית לצורך דיבוג. ב-Render, gunicorn מפעיל את האפליקציה
    app.run(debug=True, host='0.0.0.0', port=5000)
