# CS317 Lab 2 – Titanic Model Serving & Deployment

## Overview
Lab 2 hướng dẫn cách serving mô hình Titanic đã huấn luyện (Lab 1) dưới dạng API sử dụng FastAPI, đóng gói với Docker, triển khai bằng docker-compose, và test API.

## Project Structure
```text
CS317-Lab2/
│
├── main.py                # FastAPI app serving model
├── best_rf_model.pkl      # Model file (copy từ Lab 1 sau khi train)
├── requirements.txt       # Thư viện kèm phiên bản
├── Dockerfile             # Đóng gói image
├── docker-compose.yml     # Quản lý service
└── README.md              # Hướng dẫn 
```

## Member

- Từ Minh Phi - 22521080
- Lê Thành Tiến - 22521467
- Dương Thành Trí - 22521516
- Nguyễn Minh Thiện - 22521391
- Nguyễn Quốc Vinh - 22521674

---

## 1. Cài đặt & Chạy thử local

```bash
python -m venv venv
source venv/bin/activate  # Trên Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```
- Truy cập docs: http://localhost:8000/docs  

---

## 2. Build & Run với Docker Compose

```bash
docker-compose up --build
```
- API sẽ chạy tại `http://localhost:8000`.

---

## 3. Chạy container từ image đã build

Nếu bạn đã build image (ví dụ: `group12/titanic-api:latest`), có thể chạy container trực tiếp bằng lệnh:

```bash
docker run -d -p 8000:8000 group12/titanic-api:latest
```
- Truy cập API docs tại: http://localhost:8000/docs

---

## 4. Test API (Swagger UI hoặc curl/Postman)

- **Swagger UI:** Truy cập http://localhost:8000/docs, nhập các trường và nhấn "Execute".

  - Body (JSON) mẫu:
    ```json
    {
      "Pclass": 3,
      "Sex": "male",
      "Age": 22,
      "SibSp": 1,
      "Parch": 0,
      "Fare": 7.25,
      "Embarked": "S"
    }
    ```
  - Kết quả trả về:
    ```json
    {
      "Kết quả": "Không sống sót"
    }
    ```
    hoặc
    ```json
    {
      "Kết quả": "Sống sót"
    }
    ```

- **Hoặc dùng curl:**
    ```bash
    curl -X POST "http://localhost:8000/predict" ^
         -H "Content-Type: application/json" ^
         -d "{\"Pclass\":3,\"Sex\":\"male\",\"Age\":22,\"SibSp\":1,\"Parch\":0,\"Fare\":7.25,\"Embarked\":\"S\"}"
    ```

---

## 5. (Optional) Push Docker Image lên Docker Hub

```bash
docker login
docker build -t <your_dockerhub_username>/titanic-api:latest .
docker push <your_dockerhub_username>/titanic-api:latest
```

---

## 6. (Optional) Deploy trên server khác

- Nếu đã push image lên Docker Hub, chỉ cần tạo file `docker-compose.yml` như sau:
  ```yaml
  version: "3.8"
  services:
    titanic-api:
      image: <your_dockerhub_username>/titanic-api:latest
      ports:
        - "8000:8000"
      restart: always
  ```
  Sau đó chạy:
  ```bash
  docker-compose up -d
  ```

---

## Demo Video

- Quá trình chạy container từ image đã build, call API và trả kết quả:
  [Xem video demo tại đây](https://drive.google.com/file/d/1-q8LkkpOH_XNrUBfWblkxDUVUck4juaE/view?usp=sharing)

**Lưu ý:**  
- Nếu muốn dùng model khác (`best_knn_model.pkl`), sửa lại tên file và code tương ứng.
- Nếu gặp vấn đề, liên hệ @ThienNguyenMinh1311 trên GitHub.
