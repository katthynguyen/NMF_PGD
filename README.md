## Dự án: Nhận diện đặc trưng khuôn mặt bằng NMF và KNN với FastAPI

Dự án này triển khai một API sử dụng FastAPI để nhận diện đặc trưng khuôn mặt dựa trên NMF (Non-negative Matrix Factorization) và KNN (K-Nearest Neighbors). Code được đóng gói trong Docker để dễ dàng triển khai trên nhiều môi trường khác nhau.

### Tính năng
- Tiền xử lý ảnh khuôn mặt (grayscale, resize, flatten).
- Trích xuất đặc trưng bằng NMF với gradient descent.
- Phân loại danh tính bằng KNN.
- Trả về đặc trưng, nhãn dự đoán, xác suất và biểu đồ trực quan qua API.
- Hỗ trợ chạy local hoặc trong container Docker.

---

## Yêu cầu

### Phần mềm cần cài đặt
- **Python**: >= 3.9 hoặc cao hơn (khuyến nghị 3.11).
- **Docker**: Để chạy trong container (Docker Desktop trên Windows/Mac hoặc Docker Engine trên Linux).
- **Git**: Để clone repository.

### Thư viện Python
Danh sách thư viện được liệt kê trong `requirements.txt`:
---

## Cấu trúc dự án

```
face-recognition-nmf/
├── app/
│   ├── main.py         # File chính chạy FastAPI
│   ├── models.py       # Định nghĩa NMF, KNN và các lớp hỗ trợ
│   └── visualizer.py   # Tạo biểu đồ xác suất
├── train/              # Thư mục chứa ảnh huấn luyện (tạo thủ công)
│   ├── John/
│   ├── Jane/
│   └── Alex/
├── test/               # Thư mục chứa ảnh thử nghiệm (tạo thủ công)
├── Dockerfile          # File cấu hình Docker
├── requirements.txt    # Danh sách thư viện Python
└── README.md           # Hướng dẫn này
```

---

## Cài đặt và chạy

### 1. Clone repository
```bash
git clone <repository-url>
cd face-recognition-nmf
```

---

### 2. Chuẩn bị dữ liệu
- Tạo thư mục `train/` với cấu trúc như sau:
  ```
  train/
    John/
      john_1.jpg
      john_2.jpg
    Jane/
      jane_1.jpg
      jane_2.jpg
    Alex/
      alex_1.jpg
  ```
- Tạo thư mục `test/` chứa ảnh cần dự đoán (ví dụ: `test/john_test.jpg`).

---

### 3. Chạy trên môi trường Local

#### Bước 1: Tạo virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

#### Bước 2: Cài đặt thư viện
```bash
pip install -r requirements.txt
```

#### Bước 3: Chạy ứng dụng
- Chỉnh sửa `main.py` để sử dụng dữ liệu của bạn (xem phần "Ví dụ sử dụng" bên dưới).
- Chạy server FastAPI:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
- Truy cập `http://localhost:8000/docs` để xem Swagger UI và thử API.

---

### 4. Chạy với Docker
#### Bước 1: Build Docker image
```bash
docker build -t face-recognition-nmf .
```

#### Bước 2: Chạy container

##### Trên Linux/Mac
```bash
docker run -d
```

##### Trên Windows (Command Prompt)
```cmd
docker run -d
```

##### Trên Windows (PowerShell)
```powershell
docker run -d
```

#### Bước 3: Kiểm tra
- Truy cập `http://localhost:8000/docs` để xem API.

---
### 6. Ví dụ sử dụng API
- **Request**:
```bash
curl -X POST "http://localhost:8000/extract-features" \
  -H "Content-Type: application/json" \
  -d '["test/john_test.jpg", "test/jane_test.jpg"]'
```

- **Response** (mẫu):
```json
[
    {
        "image_path": "test/john_test.jpg",
        "features": [0.1, 0.2, ...],
        "predicted_label": "John",
        "probabilities": {"John": 0.8, "Jane": 0.15, "Alex": 0.05},
        "plot_path": "plots/plot_abc123.png"
    },
    ...
]
```