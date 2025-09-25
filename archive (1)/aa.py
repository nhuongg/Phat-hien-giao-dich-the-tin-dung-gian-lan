# -*- coding: utf-8 -*-
# Trang web dự đoán giao dịch thẻ tín dụng gian lận
# Sử dụng creditcard_balanced_85_15.csv để huấn luyện mô hình (85% hợp lệ, 15% gian lận)

import io, base64
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, render_template_string, jsonify, redirect, url_for, flash
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, precision_score, recall_score, confusion_matrix, 
                           roc_auc_score, accuracy_score, roc_curve, precision_recall_curve, 
                           average_precision_score, ConfusionMatrixDisplay)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import joblib
import os
import tempfile

# ====== cấu hình ======
SEED = 42
MODEL_DIR = Path("models"); MODEL_DIR.mkdir(exist_ok=True)

app = Flask(__name__)

# ====== Global variables ======
trained_models = {}
scaler = None
feature_names = []

def load_and_train_models():
    """Huấn luyện mô hình từ creditcard_balanced_85_15.csv"""
    global trained_models, scaler, feature_names
    
    print("Đang tải dữ liệu từ creditcard_balanced_85_15.csv...")
    df = pd.read_csv("creditcard_balanced_85_15.csv")
    
    # Chuẩn bị dữ liệu
    X = df.drop("Class", axis=1)
    y = df["Class"]
    feature_names = X.columns.tolist()
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=SEED, stratify=y
    )
    
    # Huấn luyện các mô hình
    models = {
        "Naive Bayes": GaussianNB(),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=SEED)
    }
    
    print("Đang huấn luyện các mô hình...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"✓ {name} đã huấn luyện xong")
    
    # Lưu mô hình và scaler
    joblib.dump(trained_models, MODEL_DIR/"trained_models.pkl")
    joblib.dump(scaler, MODEL_DIR/"scaler.pkl")
    joblib.dump(feature_names, MODEL_DIR/"feature_names.pkl")
    
    print("✓ Tất cả mô hình đã sẵn sàng!")
    return True

def load_models():
    """Tải mô hình đã huấn luyện"""
    global trained_models, scaler, feature_names
    
    try:
        trained_models = joblib.load(MODEL_DIR/"trained_models.pkl")
        scaler = joblib.load(MODEL_DIR/"scaler.pkl")
        feature_names = joblib.load(MODEL_DIR/"feature_names.pkl")
        print("✓ Đã tải mô hình từ file")
        return True
    except:
        print("Không tìm thấy mô hình, bắt đầu huấn luyện...")
        return load_and_train_models()

def predict_transaction(transaction_data, model_name="Decision Tree"):
    """Dự đoán giao dịch"""
    if model_name not in trained_models:
        return {"error": "Mô hình không tồn tại"}
    
    try:
        # Chuẩn hóa dữ liệu đầu vào
        X_scaled = scaler.transform([transaction_data])
        
        # Dự đoán
        model = trained_models[model_name]
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0]
        
        return {
            "prediction": int(prediction),
            "probability": {
                "legitimate": float(probability[0]),
                "fraudulent": float(probability[1])
            },
            "confidence": float(max(probability)),
            "result_text": "Gian lận" if prediction == 1 else "Hợp lệ"
        }
    except Exception as e:
        return {"error": str(e)}

def predict_batch_csv(file_path, model_name="Decision Tree"):
    """Dự đoán hàng loạt từ file CSV"""
    try:
        # Đọc file CSV
        df = pd.read_csv(file_path)
        
        # Kiểm tra cột bắt buộc
        required_columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'Amount']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return {"error": f"Thiếu các cột: {', '.join(missing_columns)}"}
        
        # Chuẩn bị dữ liệu
        X = df[required_columns].copy()
        
        # Thêm các cột V6-V28 nếu chưa có (với giá trị 0)
        for i in range(6, 29):
            col_name = f'V{i}'
            if col_name not in X.columns:
                X[col_name] = 0.0
        
        # Sắp xếp lại cột theo thứ tự đúng
        all_columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        X = X.reindex(columns=all_columns, fill_value=0.0)
        
        # Chuẩn hóa dữ liệu
        X_scaled = scaler.transform(X)
        
        # Dự đoán
        model = trained_models[model_name]
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        
        # Tạo kết quả
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                "index": i,
                "prediction": int(pred),
                "probability": {
                    "legitimate": float(prob[0]),
                    "fraudulent": float(prob[1])
                },
                "confidence": float(max(prob)),
                "result_text": "Gian lận" if pred == 1 else "Hợp lệ"
            })
        
        # Thống kê
        total_transactions = len(results)
        fraud_count = sum(1 for r in results if r["prediction"] == 1)
        legitimate_count = total_transactions - fraud_count
        
        return {
            "success": True,
            "total_transactions": total_transactions,
            "fraud_count": fraud_count,
            "legitimate_count": legitimate_count,
            "fraud_rate": fraud_count / total_transactions,
            "results": results,
            "model_used": model_name
        }
        
    except Exception as e:
        return {"error": f"Lỗi xử lý file: {str(e)}"}

# ====== UI Template ======
TEMPLATE = """
<!doctype html><html lang="vi"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dự đoán Giao dịch Thẻ tín dụng Gian lận</title>
<style>
  :root{
    --bg: #0f172a;
    --border: #1f2937;
    --text: #e5e7eb;
    --muted: #94a3b8;
    --primary: #6366f1;
    --primary-2: #8b5cf6;
    --bad: #ef4444;
    --good: #10b981;
    --warn: #f59e0b;
  }
  html,body{height:100%}
  body{
    margin:0; padding:32px;
    font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
    color:var(--text);
    background: radial-gradient(1200px 500px at 10% -10%, rgba(99,102,241,.15), transparent 60%),
                radial-gradient(1000px 400px at 90% -10%, rgba(139,92,246,.12), transparent 50%),
                linear-gradient(180deg, #0b1220 0%, #0f172a 60%, #0b1220 100%);
  }
  .container{max-width:1200px;margin:0 auto}
  .header{display:flex;align-items:center;justify-content:space-between;margin-bottom:20px}
  .title{font-size:22px;font-weight:700;letter-spacing:.2px}
  .subtitle{color:var(--muted);font-size:14px;margin-top:4px}
  .pill{display:inline-block;padding:6px 10px;border-radius:999px;background:rgba(99,102,241,.15);color:#c7d2fe;border:1px solid rgba(99,102,241,.35);font-size:12px}

  .card{
    border:1px solid var(--border);
    border-radius:16px;
    padding:18px;
    margin:16px 0;
    background:linear-gradient(180deg, rgba(255,255,255,.02), rgba(255,255,255,.00));
    box-shadow:0 10px 30px rgba(0,0,0,.25), inset 0 1px 0 rgba(255,255,255,.04);
    backdrop-filter: blur(4px);
  }
  h3{margin:0 0 12px 0;font-size:16px;letter-spacing:.3px}
  .grid{display:grid;gap:14px}
  .grid2{grid-template-columns:1fr 1fr}
  .grid3{grid-template-columns:1fr 1fr 1fr}
  @media (max-width: 960px){.grid2,.grid3{grid-template-columns:1fr}}

  .form-group{margin:15px 0}
  .form-group label{display:block;margin-bottom:5px;font-weight:500;color:var(--text)}
  .form-group input, .form-group select{
    width:100%;padding:10px;border:1px solid var(--border);border-radius:8px;
    background:rgba(255,255,255,.02);color:var(--text);font-size:14px
  }
  .form-group input:focus, .form-group select:focus{
    outline:none;border-color:var(--primary);box-shadow:0 0 0 2px rgba(99,102,241,.2)
  }

  .btn{display:inline-block;padding:12px 20px;border-radius:10px;background:linear-gradient(90deg, var(--primary), var(--primary-2));color:#fff;text-decoration:none;border:none;cursor:pointer;transition:.2s box-shadow;font-size:14px;font-weight:500}
  .btn:hover{box-shadow:0 8px 24px rgba(99,102,241,.35)}
  .btn.secondary{background:transparent;color:var(--text);border:1px solid var(--border)}
  .btn.danger{background:var(--bad)}
  .btn.success{background:var(--good)}

  .result-card{
    border:2px solid var(--primary);
    background:rgba(99,102,241,.05);
    padding:20px;
    border-radius:12px;
    text-align:center;
    margin:20px 0;
  }
  .result-fraud{border-color:var(--bad);background:rgba(239,68,68,.05)}
  .result-legitimate{border-color:var(--good);background:rgba(16,185,129,.05)}

  .probability-bar{
    width:100%;height:20px;background:rgba(255,255,255,.1);border-radius:10px;overflow:hidden;margin:10px 0
  }
  .probability-fill{
    height:100%;transition:width 0.3s ease;border-radius:10px
  }
  .fraud-fill{background:linear-gradient(90deg, #ef4444, #dc2626)}
  .legitimate-fill{background:linear-gradient(90deg, #10b981, #059669)}

  .small{font-size:13px;color:var(--muted)}
  .danger{color:var(--bad)}
  .success{color:var(--good)}
  .warning{color:var(--warn)}

  .loading{display:none;text-align:center;padding:20px}
  .spinner{border:3px solid var(--border);border-top:3px solid var(--primary);border-radius:50%;width:40px;height:40px;animation:spin 1s linear infinite;margin:0 auto 10px}
  @keyframes spin{0%{transform:rotate(0deg)}100%{transform:rotate(360deg)}}
</style></head><body>

<div class="container">
  <div class="header">
    <div>
      <div class="title">Dự đoán Giao dịch Thẻ tín dụng Gian lận <span class="pill">AI Prediction</span></div>
      <div class="subtitle">Nhập thông tin giao dịch hoặc tải lên file CSV để dự đoán</div>
    </div>
  </div>

<div class="card">
  <h3>Dự đoán đơn lẻ</h3>
  <form id="predictionForm">
    <div class="grid grid2">
      <div class="form-group">
        <label for="amount">Số tiền giao dịch (Amount)</label>
        <input type="number" id="amount" name="amount" step="0.01" placeholder="Ví dụ: 25.50" required>
      </div>
      <div class="form-group">
        <label for="time">Thời gian (giây từ giao dịch đầu tiên)</label>
        <input type="number" id="time" name="time" placeholder="Ví dụ: 123456" required>
      </div>
    </div>
    
    <div class="grid grid3">
      <div class="form-group">
        <label for="v1">V1 (PCA Component 1)</label>
        <input type="number" id="v1" name="v1" step="0.000001" placeholder="Ví dụ: -1.359807" required>
      </div>
      <div class="form-group">
        <label for="v2">V2 (PCA Component 2)</label>
        <input type="number" id="v2" name="v2" step="0.000001" placeholder="Ví dụ: 0.843503" required>
      </div>
      <div class="form-group">
        <label for="v3">V3 (PCA Component 3)</label>
        <input type="number" id="v3" name="v3" step="0.000001" placeholder="Ví dụ: -0.316615" required>
      </div>
    </div>
    
    <div class="grid grid3">
      <div class="form-group">
        <label for="v4">V4 (PCA Component 4)</label>
        <input type="number" id="v4" name="v4" step="0.000001" placeholder="Ví dụ: -0.150771" required>
      </div>
      <div class="form-group">
        <label for="v5">V5 (PCA Component 5)</label>
        <input type="number" id="v5" name="v5" step="0.000001" placeholder="Ví dụ: 0.674064" required>
      </div>
      <div class="form-group">
        <label for="model">Chọn mô hình dự đoán</label>
        <select id="model" name="model">
          <option value="Decision Tree">Decision Tree (Khuyến nghị)</option>
          <option value="Naive Bayes">Naive Bayes</option>
          <option value="KNN (k=5)">KNN (k=5)</option>
        </select>
      </div>
    </div>
    
    <button type="submit" class="btn">Dự đoán giao dịch</button>
    <button type="button" class="btn secondary" onclick="fillSampleData()">Dữ liệu mẫu</button>
  </form>
</div>

<div class="card">
  <h3>Dự đoán hàng loạt từ file CSV</h3>
  <form id="csvUploadForm" enctype="multipart/form-data">
    <div class="form-group">
      <label for="csvFile">Chọn file CSV</label>
      <input type="file" id="csvFile" name="csvFile" accept=".csv" required>
      <p class="small">File CSV phải có các cột: Time, V1, V2, V3, V4, V5, Amount</p>
    </div>
    
    <div class="form-group">
      <label for="csvModel">Chọn mô hình dự đoán</label>
      <select id="csvModel" name="csvModel">
        <option value="Decision Tree">Decision Tree (Khuyến nghị)</option>
        <option value="Naive Bayes">Naive Bayes</option>
        <option value="KNN (k=5)">KNN (k=5)</option>
      </select>
    </div>
    
    <button type="submit" class="btn">Dự đoán từ file CSV</button>
    <button type="button" class="btn secondary" onclick="downloadSampleCSV()">Tải file mẫu CSV</button>
  </form>
</div>

<div class="loading" id="loading">
  <div class="spinner"></div>
  <p>Đang phân tích giao dịch...</p>
</div>

<div id="result" style="display:none">
  <!-- Kết quả dự đoán sẽ hiển thị ở đây -->
</div>

<div class="card">
  <h3>Hướng dẫn sử dụng</h3>
  <div class="grid grid2">
    <div>
      <p><strong>Dự đoán đơn lẻ:</strong></p>
      <ul class="small">
        <li><strong>Amount:</strong> Số tiền giao dịch (USD)</li>
        <li><strong>Time:</strong> Thời gian tính bằng giây từ giao dịch đầu tiên</li>
        <li><strong>V1-V5:</strong> Các thành phần PCA chính</li>
      </ul>
    </div>
    <div>
      <p><strong>Dự đoán hàng loạt:</strong></p>
      <ul class="small">
        <li><strong>File CSV:</strong> Chứa nhiều giao dịch cùng lúc</li>
        <li><strong>Cột bắt buộc:</strong> Time, V1, V2, V3, V4, V5, Amount</li>
        <li><strong>Kết quả:</strong> Thống kê tổng quan + chi tiết từng giao dịch</li>
      </ul>
    </div>
  </div>
  <div class="grid grid2">
    <div>
      <p><strong>Mô hình AI:</strong></p>
      <ul class="small">
        <li><strong>Decision Tree:</strong> Chính xác cao, dễ hiểu</li>
        <li><strong>Naive Bayes:</strong> Nhanh, phù hợp dữ liệu lớn</li>
        <li><strong>KNN:</strong> Dựa trên giao dịch tương tự</li>
      </ul>
    </div>
    <div>
      <p><strong>Dataset huấn luyện:</strong></p>
      <ul class="small">
        <li><strong>File:</strong> creditcard_balanced_85_15.csv</li>
        <li><strong>Tỷ lệ:</strong> 85% hợp lệ, 15% gian lận</li>
        <li><strong>Kích thước:</strong> 3,282 giao dịch</li>
      </ul>
    </div>
  </div>
</div>

  <div class="small" style="text-align:center;margin:18px 0;color:var(--muted)">
    © 2025 Dự đoán Giao dịch Thẻ tín dụng Gian lận • AI Powered
  </div>
</div>

<script>
function showLoading() {
    document.getElementById('loading').style.display = 'block';
    document.getElementById('result').style.display = 'none';
}

function hideLoading() {
    document.getElementById('loading').style.display = 'none';
}

function fillSampleData() {
    // Dữ liệu mẫu từ creditcard.csv
    document.getElementById('amount').value = '25.50';
    document.getElementById('time').value = '123456';
    document.getElementById('v1').value = '-1.359807';
    document.getElementById('v2').value = '0.843503';
    document.getElementById('v3').value = '-0.316615';
    document.getElementById('v4').value = '-0.150771';
    document.getElementById('v5').value = '0.674064';
}

function downloadSampleCSV() {
    // Tạo file CSV mẫu
    const sampleData = `Time,V1,V2,V3,V4,V5,Amount
123456,-1.359807,0.843503,-0.316615,-0.150771,0.674064,25.50
234567,1.191857,0.266151,0.166480,0.448154,0.060018,3.25
345678,-1.358354,-1.340163,1.773209,0.379780,-0.503198,123.50
456789,1.158233,0.877737,1.548718,0.403034,-0.407193,69.99
567890,-0.425966,0.960523,0.639412,-0.294886,0.215153,10.00`;
    
    const blob = new Blob([sampleData], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'sample_transactions.csv';
    a.click();
    window.URL.revokeObjectURL(url);
}

document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    showLoading();
    
    const formData = new FormData(this);
    const data = Object.fromEntries(formData);
    
    // Chuyển đổi sang số
    for (let key in data) {
        if (key !== 'model') {
            data[key] = parseFloat(data[key]);
        }
    }
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (result.error) {
            showError(result.error);
        } else {
            showResult(result);
        }
    } catch (error) {
        showError('Lỗi kết nối: ' + error.message);
    } finally {
        hideLoading();
    }
});

// Xử lý upload file CSV
document.getElementById('csvUploadForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    showLoading();
    
    const formData = new FormData();
    const fileInput = document.getElementById('csvFile');
    const modelSelect = document.getElementById('csvModel');
    
    if (!fileInput.files[0]) {
        showError('Vui lòng chọn file CSV');
        hideLoading();
        return;
    }
    
    formData.append('csvFile', fileInput.files[0]);
    formData.append('model', modelSelect.value);
    
    try {
        const response = await fetch('/predict_csv', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.error) {
            showError(result.error);
        } else {
            showBatchResult(result);
        }
    } catch (error) {
        showError('Lỗi kết nối: ' + error.message);
    } finally {
        hideLoading();
    }
});

function showResult(result) {
    const isFraud = result.prediction === 1;
    const fraudProb = result.probability.fraudulent;
    const legitimateProb = result.probability.legitimate;
    
    const resultClass = isFraud ? 'result-fraud' : 'result-legitimate';
    const resultIcon = isFraud ? '🚨' : '✅';
    const resultColor = isFraud ? 'danger' : 'success';
    
    const html = `
        <div class="card">
            <div class="result-card ${resultClass}">
                <h2 style="margin:0 0 10px 0;font-size:24px">
                    Kết quả: <span class="${resultColor}">${result.result_text}</span>
                </h2>
                <p class="small">Độ tin cậy: ${(result.confidence * 100).toFixed(1)}%</p>
                
                <div style="margin:20px 0">
                    <div style="display:flex;justify-content:space-between;margin-bottom:5px">
                        <span>Hợp lệ: ${(legitimateProb * 100).toFixed(1)}%</span>
                        <span>Gian lận: ${(fraudProb * 100).toFixed(1)}%</span>
                    </div>
                    <div class="probability-bar">
                        <div class="legitimate-fill" style="width:${legitimateProb * 100}%"></div>
                    </div>
                </div>
                
                <div style="margin-top:15px;padding:10px;background:rgba(255,255,255,.05);border-radius:8px">
                    <p class="small"><strong>Mô hình sử dụng:</strong> ${result.model || 'Decision Tree'}</p>
                    <p class="small"><strong>Thời gian phân tích:</strong> ${result.processing_time || '< 1s'}</p>
                </div>
            </div>
        </div>
    `;
    
    document.getElementById('result').innerHTML = html;
    document.getElementById('result').style.display = 'block';
}

function showBatchResult(result) {
    const fraudRate = (result.fraud_rate * 100).toFixed(1);
    const fraudColor = result.fraud_rate > 0.15 ? 'danger' : result.fraud_rate > 0.05 ? 'warning' : 'success';
    
    let resultsTable = '';
    result.results.forEach((item, index) => {
        const rowClass = item.prediction === 1 ? 'danger' : 'success';
        const icon = item.prediction === 1 ? '🚨' : '✅';
        resultsTable += `
            <tr class="${rowClass}">
                <td>${index + 1}</td>
                <td>${icon} ${item.result_text}</td>
                <td>${(item.probability.legitimate * 100).toFixed(1)}%</td>
                <td>${(item.probability.fraudulent * 100).toFixed(1)}%</td>
                <td>${(item.confidence * 100).toFixed(1)}%</td>
            </tr>
        `;
    });
    
    const html = `
        <div class="card">
            <div class="result-card">
                <h2 style="margin:0 0 15px 0;font-size:24px">
                    Kết quả dự đoán hàng loạt
                </h2>
                
                <div class="grid grid2" style="margin:20px 0">
                    <div style="text-align:center;padding:15px;background:rgba(255,255,255,.05);border-radius:8px">
                        <h3 style="margin:0;color:var(--text)">${result.total_transactions}</h3>
                        <p class="small">Tổng giao dịch</p>
                    </div>
                    <div style="text-align:center;padding:15px;background:rgba(255,255,255,.05);border-radius:8px">
                        <h3 style="margin:0;color:var(--${fraudColor})">${fraudRate}%</h3>
                        <p class="small">Tỷ lệ gian lận</p>
                    </div>
                </div>
                
                <div class="grid grid2" style="margin:15px 0">
                    <div style="text-align:center;padding:10px;background:rgba(16,185,129,.1);border-radius:8px">
                        <h4 style="margin:0;color:var(--good)">${result.legitimate_count}</h4>
                        <p class="small">Hợp lệ</p>
                    </div>
                    <div style="text-align:center;padding:10px;background:rgba(239,68,68,.1);border-radius:8px">
                        <h4 style="margin:0;color:var(--bad)">${result.fraud_count}</h4>
                        <p class="small">Gian lận</p>
                    </div>
                </div>
                
                <div style="margin:15px 0;padding:10px;background:rgba(255,255,255,.05);border-radius:8px">
                    <p class="small"><strong>Mô hình sử dụng:</strong> ${result.model_used}</p>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h3>Chi tiết từng giao dịch</h3>
            <div style="overflow-x:auto;max-height:400px;overflow-y:auto">
                <table style="width:100%;border-collapse:collapse;font-size:14px">
                    <thead>
                        <tr style="background:rgba(255,255,255,.1);position:sticky;top:0">
                            <th style="padding:8px;text-align:left;border-bottom:1px solid var(--border)">#</th>
                            <th style="padding:8px;text-align:left;border-bottom:1px solid var(--border)">Kết quả</th>
                            <th style="padding:8px;text-align:left;border-bottom:1px solid var(--border)">Hợp lệ</th>
                            <th style="padding:8px;text-align:left;border-bottom:1px solid var(--border)">Gian lận</th>
                            <th style="padding:8px;text-align:left;border-bottom:1px solid var(--border)">Độ tin cậy</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${resultsTable}
                    </tbody>
                </table>
            </div>
        </div>
    `;
    
    document.getElementById('result').innerHTML = html;
    document.getElementById('result').style.display = 'block';
}

function showError(message) {
    document.getElementById('result').innerHTML = `
        <div class="card">
            <div class="danger" style="text-align:center;padding:20px">
                <h3>Lỗi</h3>
                <p>${message}</p>
            </div>
        </div>
    `;
    document.getElementById('result').style.display = 'block';
}
</script>

</body></html>
"""

# ====== routes ======
@app.route("/")
def home():
    return render_template_string(TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint để dự đoán giao dịch"""
    try:
        data = request.get_json()
        
        # Kiểm tra dữ liệu đầu vào
        required_fields = ['amount', 'time', 'v1', 'v2', 'v3', 'v4', 'v5']
        for field in required_fields:
            if field not in data or data[field] is None:
                return jsonify({"error": f"Thiếu thông tin: {field}"})
        
        # Tạo vector đặc trưng (chỉ sử dụng 7 đặc trưng chính)
        transaction_data = [
            data['time'],
            data['v1'], data['v2'], data['v3'], data['v4'], data['v5'],
            data['amount']
        ]
        
        # Thêm các đặc trưng V6-V28 với giá trị mặc định (0)
        for i in range(6, 29):
            transaction_data.append(0.0)
        
        # Dự đoán
        model_name = data.get('model', 'Decision Tree')
        result = predict_transaction(transaction_data, model_name)
        
        if 'error' in result:
            return jsonify(result)
        
        result['model'] = model_name
        result['processing_time'] = '< 1s'
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Lỗi xử lý: {str(e)}"})

@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    """API endpoint để dự đoán từ file CSV"""
    try:
        if 'csvFile' not in request.files:
            return jsonify({"error": "Không có file được tải lên"})
        
        file = request.files['csvFile']
        if file.filename == '':
            return jsonify({"error": "Không có file được chọn"})
        
        if not file.filename.lower().endswith('.csv'):
            return jsonify({"error": "File phải có định dạng CSV"})
        
        # Lưu file tạm thời với phương pháp an toàn hơn
        import tempfile
        import time
        
        # Tạo file tạm thời
        fd, tmp_path = tempfile.mkstemp(suffix='.csv')
        try:
            # Lưu file
            with os.fdopen(fd, 'wb') as tmp_file:
                file.save(tmp_file)
            
            # Lấy mô hình từ form
            model_name = request.form.get('model', 'Decision Tree')
            
            # Dự đoán
            result = predict_batch_csv(tmp_path, model_name)
            
            return jsonify(result)
            
        finally:
            # Xóa file tạm (với xử lý lỗi)
            try:
                os.unlink(tmp_path)
            except (OSError, PermissionError) as e:
                print(f"Không thể xóa file tạm {tmp_path}: {e}")
                # Thử xóa sau một chút
                time.sleep(0.1)
                try:
                    os.unlink(tmp_path)
                except:
                    pass  # Bỏ qua nếu vẫn không xóa được
            
    except Exception as e:
        return jsonify({"error": f"Lỗi xử lý file: {str(e)}"})

@app.route("/status")
def status():
    """Kiểm tra trạng thái mô hình"""
    return jsonify({
        "models_loaded": len(trained_models) > 0,
        "scaler_loaded": scaler is not None,
        "feature_count": len(feature_names) if feature_names else 0
    })

if __name__ == "__main__":
    if load_models():
        app.run(debug=True, host='0.0.0.0', port=5001)
   


