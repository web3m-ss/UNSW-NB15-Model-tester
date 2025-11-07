import requests
import pandas as pd
import json

# --- 1. อ่านไฟล์ Test Set ---
try:
    df_test = pd.read_csv("UNSW_NB15_testing-set.csv")
    print("อ่าน UNSW_NB15_testing-set.csv สำเร็จ")
except FileNotFoundError:
    print("!!! ไม่พบไฟล์ 'UNSW_NB15_testing-set.csv' !!!")
    exit()

# --- 2. สุ่มตัวอย่างมา 1 แถวเพื่อทดสอบ ---
# (เราจะใช้ .sample(1) เพื่อสุ่ม หรือ .iloc[0] เพื่อเอาแถวแรก)
test_row_series = df_test.sample(1).iloc[0]

# คำตอบที่แท้จริงของแถวนี้ (ไว้เปรียบเทียบ)
actual_label = "Attack" if test_row_series['label'] == 1 else "Normal"

# --- 3. แปลงแถว (Series) ให้เป็น Dictionary (JSON) ---
# (เราต้องลบ id, label, attack_cat ออกก่อนส่งไป)
test_data_dict = test_row_series.drop(['id', 'label', 'attack_cat']).to_dict()

# แสดงข้อมูลที่จะส่งไป
print("\n--- ข้อมูลที่จะส่งไปทดสอบ (1 แถว) ---")
print(json.dumps(test_data_dict, indent=2))
print(f"คำตอบที่แท้จริง (จาก CSV): {actual_label}")

# --- 4. ส่ง Request ไปยัง API Server ---
# (ต้องมั่นใจว่า app.py กำลังรันอยู่)
url = 'http://127.0.0.1:5000/predict'

print("\n...กำลังส่งข้อมูลไปยังเซิร์ฟเวอร์โมเดล...")

try:
    response = requests.post(url, json=test_data_dict)
    
    if response.status_code == 200:
        # --- 5. รับผลการทำนายกลับมา ---
        result = response.json()
        print("\n--- ผลลัพธ์จากโมเดล ---")
        print(f"การทำนาย: {result.get('prediction')}")
        
        # เปรียบเทียบผล
        if result.get('prediction') == actual_label:
            print("ผลลัพธ์: ถูกต้อง!")
        else:
            print("ผลลัพธ์: ผิดพลาด!")
            
    else:
        print(f"เกิดข้อผิดพลาดจากเซิร์ฟเวอร์ (Code: {response.status_code}):")
        print(response.json())

except requests.exceptions.ConnectionError:
    print("\n!!! ไม่สามารถเชื่อมต่อเซิร์ฟเวอร์ได้ !!!")
    print("กรุณารัน 'python app.py' ในอีก Terminal หนึ่งก่อน")