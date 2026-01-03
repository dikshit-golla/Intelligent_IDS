from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import shap
import os
import google.generativeai as genai

app = Flask(__name__)

# Configure Gemini AI
genai.configure(api_key=os.getenv('GEMINI_API_KEY', 'AIzaSyAeZvQHUKMvA4RFYMR2XzWK1ahAtmHbdFI'))  # Replace with your actual API key or set environment variable

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'models', 'xgb_model.pkl')
model = joblib.load(model_path)

# Load the preprocessor for SHAP
preprocessor = model.named_steps['preprocess']
clf = model.named_steps['clf']

# Feature names
categorical_features = ['protocol_type', 'service', 'flag']
numeric_features = [
    'duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]

# Home page with form
@app.route('/', methods=['GET'])
def index():
    # lists for select inputs
    protocols = ['tcp', 'udp', 'icmp']
    services = ['http', 'smtp', 'ftp', 'domain_u', 'auth', 'finger', 'eco_i', 'other']
    flags = ['SF', 'S0', 'REJ', 'RSTR', 'S1', 'S2', 'S3']
    return render_template('index.html', protocols=protocols, services=services, flags=flags)


@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data
    form_data = {k: float(request.form.get(k)) if request.form.get(k).replace('.', '').isdigit() else request.form.get(k) for k in request.form}
    
    # Convert to DataFrame
    input_df = pd.DataFrame([form_data])
    
    # Ensure correct data types
    for col in numeric_features:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(float)
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]
    
    # Preprocess for SHAP
    input_preprocessed = preprocessor.transform(input_df)
    
    # SHAP explainer
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(input_preprocessed)
    
    # Get feature names after preprocessing
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_feature_names = np.concatenate([numeric_features, cat_feature_names])
    
    # For single prediction, shap_values might be 2D for binary classification
    if len(shap_values.shape) == 3:
        shap_values = shap_values[1]  # For class 1 (attack)
    
    # Get top contributing features
    shap_dict = dict(zip(all_feature_names, shap_values[0]))
    top_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    
    # Generate AI summary using Gemini
    prediction_text = "ATTACK" if prediction == 1 else "NORMAL"
    confidence = prediction_proba[prediction] * 100
    top_features_str = ", ".join([f"{feat}: {val:.3f}" for feat, val in top_features])
    
    prompt = f"""
    Based on the intrusion detection analysis:
    - Prediction: {prediction_text}
    - Confidence: {confidence:.2f}%
    - Top contributing features (SHAP values): {top_features_str}
    
    Generate a concise, professional summary explaining the prediction and the role of the key features in the decision.
    """
    
    try:
        model_ai = genai.GenerativeModel('gemini-2.0-flash')
        response = model_ai.generate_content(prompt)
        ai_summary = response.text.strip()
    except Exception as e:
        ai_summary = f"Error generating AI summary: {str(e)}. Fallback: The model predicts {prediction_text} with {confidence:.2f}% confidence. Key features: {top_features_str}"
    
    return render_template('result.html', 
                         data=form_data, 
                         prediction=prediction, 
                         confidence=confidence,
                         shap_values=top_features,
                         ai_summary=ai_summary)


if __name__ == '__main__':
    app.run(debug=True)
