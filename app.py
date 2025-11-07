import streamlit as st
import pandas as pd
import joblib
import json

MODEL_COLUMNS = [
    'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes',
    'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt',
    'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt',
    'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len',
    'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm',
    'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd',
    'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports'
]

@st.cache_resource
def load_model():
    try:
        model_pipeline = joblib.load('unsw_rf_model.joblib')
        return model_pipeline
    except FileNotFoundError:
        st.error("ไม่พบไฟล์โมเดล 'unsw_rf_model.joblib'!")
        return None

@st.cache_resource
def load_test_data():
    try:
        df = pd.read_csv("UNSW_NB15_testing-set.csv")
        df_normal = df[df['label'] == 0]
        df_attack = df[df['label'] == 1]
        return df_normal, df_attack
    except FileNotFoundError:
        st.error("ไม่พบไฟล์ข้อมูล 'UNSW_NB15_testing-set.csv'!")
        return None, None

pipeline = load_model()
df_normal, df_attack = load_test_data()

st.set_page_config(layout="wide", page_title="UNSW-NB15 Tester")
st.title("UNSW-NB15 Model Tester")
st.markdown("ทดสอบโมเดล Random Forest ด้วยข้อมูลจาก Test Set")

if pipeline is None or df_normal is None:
    st.error("แอปไม่สามารถเริ่มทำงานได้ ขาดไฟล์ที่จำเป็น (Model หรือ CSV)")
    st.stop() 

if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

def run_prediction(sample_type):
    if sample_type == 'normal':
        sample_row = df_normal.sample(1).iloc[0]
        actual_label = 'Normal'
    else:
        sample_row = df_attack.sample(1).iloc[0]
        actual_label = 'Attack'
    
    sample_data_dict = sample_row.drop(['id', 'label', 'attack_cat']).to_dict()
    
    input_df = pd.DataFrame([sample_data_dict], columns=MODEL_COLUMNS)
    
    prediction = pipeline.predict(input_df)
    model_prediction = 'Attack' if prediction[0] == 1 else 'Normal'
    
    st.session_state.last_prediction = {
        'data': sample_data_dict,
        'actual': actual_label,
        'predicted': model_prediction
    }

st.divider()
st.subheader("1. เลือกประเภทข้อมูลที่จะสุ่มทดสอบ")

col1, col2 = st.columns(2)

with col1:
    st.button(
        "ทดสอบด้วยข้อมูล Normal 1 แถว", 
        on_click=run_prediction, 
        args=('normal',), 
        use_container_width=True
    )

with col2:
    st.button(
        "ทดสอบด้วยข้อมูล Attack 1 แถว", 
        on_click=run_prediction, 
        args=('attack',), 
        use_container_width=True
    )

if st.session_state.last_prediction:
    result = st.session_state.last_prediction
    
    st.divider()
    st.subheader("2. ผลการทำนาย")
    
    is_correct = (result['actual'] == result['predicted'])
    
    if is_correct:
        st.success("✅ ผลลัพธ์ถูกต้อง!")
    else:
        st.error("❌ ผลลัพธ์ผิดพลาด!")

    res_col1, res_col2 = st.columns(2)
    res_col1.metric("ข้อมูลจริง (Actual Label)", result['actual'])
    res_col2.metric("โมเดลทำนาย (Model Prediction)", result['predicted'])
    
    with st.expander("คลิกเพื่อดูข้อมูล JSON ที่ส่งไปทำนาย"):
        st.json(result['data'])