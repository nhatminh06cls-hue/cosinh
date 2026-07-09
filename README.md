# SCOLOSIS AND HUNCHBACK DIAGNOSTIC SYSTEM- BME HUST

> **Mô tả:** Hệ thống phân tích chuyển động và đánh giá tư thế toàn diện (Intelligent Medical Measurement - AI Scoliosis Screening System). Dự án tận dụng mạng Neural Network với mô hình re-trained YOLOv8m-pose và mạng nội suy không gian 3D MiDaS. Phát triển bởi sinh viên Khoa Kỹ thuật Y sinh (BME) tại Trường Đại học Bách khoa Hà Nội (HUST).

## 🚀 Các Tính Năng Chính
* **1. Phân tích tư thế mặt trước (Front):** Đo góc lệch vai, phát hiện vai cao vai thấp.
* **2. Phân tích tư thế mặt nghiêng (Side):** Đo góc ngã thân người, phát hiện gù lưng hoặc ưỡn cột sống.
* **3. Sàng lọc vẹo cột sống (Adam's Forward Bend Test):** Tạo bản đồ chiều sâu 3D (Depth map) để tính toán độ lồi bướu sườn, thay thế máy quét 3D vật lý đắt tiền.
* **4. Báo cáo Tự động:** Tự động tạo báo cáo PDF y khoa chi tiết với biểu đồ trực quan.

---

## 📊 Báo Cáo Lâm Sàng (Kết Quả Demo Thực Tế)

Dưới đây là dữ liệu trích xuất từ một ca sàng lọc thực tế do hệ thống tự động phân tích và kết xuất.

### 1. Tổng Hợp Kết Quả Sàng Lọc

| Hạng mục Sàng lọc | Chỉ số đo được do AI quét | Ngưỡng cảnh báo y khoa | Kết luận của hệ thống |
| :--- | :--- | :--- | :--- |
| **Đo Mặt Trước (Góc lệch vai)** | **2.0°** | > 4.0° | ✅ Bình thường |
| **Đo Mặt Nghiêng (Góc gù lưng)** | **13.8°** | > 18.0° | ✅ Bình thường |
| **Adam Test (Chỉ số bất đối xứng)** | **Asym Index: 4.0%** | > 10.0% | ✅ Trong giới hạn bình thường |

*Lưu ý: Tất cả chỉ số trên nằm trong giới hạn bình thường. Hệ thống khuyến nghị tiếp tục theo dõi định kỳ.*

### 2. Phương pháp & Nguyên lý tính toán của AI

#### A. Đo Mặt Trước (Đường vai)
* **Cách hoạt động:** Camera ghi lại ảnh người dùng từ phía trước. AI (YOLO) tự động xác định vị trí hai vai và vẽ đường thẳng nối chúng.
* **Đánh giá:** Nếu đường đó bị nghiêng (vai bên cao bên thấp), hệ thống đo góc lệch đó. Góc càng lớn, vai càng bị lệch.

#### B. Đo Mặt Nghiêng (Đường lưng)
* **Cách hoạt động:** Camera ghi lại ảnh người dùng từ phía bên cạnh. AI xác định 3 điểm: cổ, vai và hông.
* **Đánh giá:** Hệ thống đo góc uốn của cột sống tại điểm vai. Người đứng thẳng hoàn toàn góc này = 0°. Góc càng lớn chứng tỏ cột sống càng được cuộn về phía trước (nguy cơ gù lưng).

#### C. Phân tích Cột Sống (Adam Test) bằng MiDaS
* **Bước 1 - Chụp ảnh chiều sâu:** Camera phân tích ảnh người cúi về phía trước để tạo ra bản đồ độ lồi lõm 3D của lưng (Vùng màu cam/vàng = nhô ra gần camera hơn, vùng màu xanh = phẳng hơn hoặc lõm vào).
* **Bước 2 - Xác định cột sống:** AI tìm đường chạy dọc giữa lưng (Spine Line) làm trục chuẩn.
* **Bước 3 - Phát hiện bất đối xứng (Rib Hump Score):** Hệ thống so sánh độ lồi của lưng bên trái và bên phải cột sống trên từng hàng pixel. Nếu một bên nhô cao hơn rõ rệt, đó là dấu hiệu lồng ngực vặn xoắn do vẹo cột sống. Chỉ số Asym Index càng cao, nguy cơ càng lớn.

---

## 🛠️ Hướng Dẫn Cài Đặt

**Điều kiện tiên quyết:**
* Python 3.8+
* Khuyến nghị dùng môi trường ảo (`venv` hoặc `conda`)

**Các bước cài đặt:**

1.  **Clone repository về máy:**
    ```bash
    git clone [https://github.com/nhatminh06cls-hue/cosinh.git](https://github.com/nhatminh06cls-hue/cosinh.git)
    cd cosinh
    ```

2.  **Cài đặt các thư viện cần thiết:**
    ```bash
    pip install -r requirements.txt
    ```

## 💻 Cách Sử Dụng

Quy trình phân tích cơ bản đi qua 2 tập lệnh cốt lõi:

**Bước 1: Phân tích Động học & Quét 3D**
Sử dụng mô hình `yolov8m-pose.pt` và tập lệnh `gk_omni_v4.py` để tính toán các số liệu từ ảnh/video đầu vào.
```bash
python gk_omni_v4.py --input path/to/input.mp4 --model yolov8m-pose.pt --output analyzed_data.json
```
**Bước 2: Tạo Báo cáo PDF Lâm sàng
Sử dụng tập lệnh report_generator.py để chuyển đổi file JSON phân tích thành báo cáo PDF hoàn chỉnh.
```bash
python report_generator.py --data analyzed_data.json --output final_report.pdf
```
## 📁 Cấu trúc Tệp tin
* `gk_omni_v4.py`: Tập lệnh cốt lõi cho phân tích động học và không gian.
* `report_generator.py`: Khối xử lý tạo báo cáo PDF đầu ra.
* `yolov8m-pose.pt`: Mô hình re-trained YOLOv8 cho nhận diện keypoints y khoa.
* `.gitignore`: Tệp cấu hình ẩn các file hệ thống tạm.

## 👥 Tác giả
* **bme-hust** 
* GitHub: [@nhatminh06cls-hue](https://github.com/nhatminh06cls-hue)
* *Disclaimer: Báo cáo tự động từ hệ thống chỉ mang tính chất sàng lọc cộng đồng, không thay thế chẩn đoán y khoa chuyên nghiệp.*
