import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

try:
    loaded_pipeline = joblib.load('unsw_rf_model.joblib')
except FileNotFoundError:
    print("!!! ไม่พบไฟล์ 'unsw_rf_model.joblib' !!!")
    exit()

try:
    df_test_full = pd.read_csv("UNSW_NB15_testing-set.csv")
    
    df_normal_samples = df_test_full[df_test_full['label'] == 0]
    df_attack_samples = df_test_full[df_test_full['label'] == 1]
    
except FileNotFoundError:
    print("!!! ไม่พบไฟล์ 'UNSW_NB15_testing-set.csv' !!!")
    exit()

MODEL_COLUMNS = [
    'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes',
    'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt',
    'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt',
    'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len',
    'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm',
    'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd',
    'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports'
]

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    json_data = request.json
    try:
        input_df = pd.DataFrame([json_data], columns=MODEL_COLUMNS)
    except Exception as e:
        return jsonify({'error': f'Input data format error: {e}'})

    try:
        prediction = loaded_pipeline.predict(input_df)
        result = 'Attack' if prediction[0] == 1 else 'Normal'
        return jsonify({'prediction': result, 'status': 'success'})
    except Exception as e:
        return jsonify({'error': f'Prediction error: {e}'})

@app.route('/get_sample/<sample_type>', methods=['GET'])
def get_sample(sample_type):
    if sample_type == 'normal':
        sample_row = df_normal_samples.sample(1).iloc[0]
        actual_label = 'Normal'
    elif sample_type == 'attack':
        sample_row = df_attack_samples.sample(1).iloc[0]
        actual_label = 'Attack'
    else:
        return jsonify({'error': 'Invalid sample type'}), 400
        
    sample_data = sample_row.drop(['id', 'label', 'attack_cat']).to_dict()
    
    return jsonify({
        'data': sample_data,
        'actual_label': actual_label
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)