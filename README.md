# Phát Hiện Rác Thải với YOLOv8

## Mục Tiêu Dự Án

Dự án này nhằm phát triển một hệ thống phát hiện rác thải tự động sử dụng công nghệ YOLOv8 (You Only Look Once version 8). Hệ thống có khả năng:

-   **Phát hiện và phân loại** các loại rác thải khác nhau trong hình ảnh
-   **Định vị chính xác** vị trí rác thải với bounding box
-   **Xử lý thời gian thực** với tốc độ cao
-   **Độ chính xác cao** trong việc nhận diện các đối tượng rác thải
-   **Ứng dụng thực tế** trong việc quản lý môi trường và tái chế

## Cấu Trúc Thư Mục Dự Án

```
yolov11/
├── dataset.v1i.yolov8/          # Dataset phiên bản 1
│   ├── data.yaml                # Cấu hình dataset
│   ├── train/                   # Dữ liệu huấn luyện
│   │   ├── images/             # Hình ảnh huấn luyện
│   │   └── labels/             # Nhãn huấn luyện
│   │
│   ├── valid/                   # Dữ liệu xác thực
│   │   ├── images/             # Hình ảnh xác thực
│   │   └── labels/             # Nhãn xác thực
│   │
│   └── test/                    # Dữ liệu kiểm thử
│       ├── images/              # Hình ảnh kiểm thử
│       └── labels/              # Nhãn kiểm thử
│
├── dataset.v2i.yolov8/          # Dataset phiên bản 2
│   ├── data.yaml                # Cấu hình dataset
│   ├── train/                   # Dữ liệu huấn luyện
│   ├── valid/                   # Dữ liệu xác thực
│   └── test/                    # Dữ liệu kiểm thử
│
├── waste_detection.ipynb        # Notebook gốc từ Google Colab
├── requirements.txt             # Các thư viện Python cần thiết
└── README.md                   # Tài liệu hướng dẫn này
```

## Công Nghệ Sử Dụng

### Core Technologies

-   **YOLOv8**: Framework phát hiện đối tượng mới nhất từ Ultralytics
-   **PyTorch**: Deep learning framework
-   **OpenCV**: Xử lý hình ảnh và computer vision
-   **Roboflow**: Platform quản lý và annotation dataset

### Libraries & Dependencies

-   **ultralytics**: Framework YOLOv8 chính thức
-   **torch**: PyTorch deep learning
-   **torchvision**: Computer vision tools cho PyTorch
-   **opencv-python**: Computer vision library
-   **Pillow**: Image processing
-   **matplotlib**: Visualization
-   **numpy**: Numerical computing
-   **roboflow**: Dataset management

## Kiến Trúc Hệ Thống

### 1. Data Pipeline

```
Raw Images → Annotation → Dataset Preparation → YOLO Format
```

### 2. Training Pipeline

```
Dataset → YOLOv8 Model → Training → Validation → Model Export
```

### 3. Inference Pipeline

```
Input Image → Preprocessing → YOLOv8 Inference → Post-processing → Results
```

### 4. Model Architecture

-   **Backbone**: CSPDarknet (Cross Stage Partial Darknet)
-   **Neck**: PANet (Path Aggregation Network)
-   **Head**: Detection heads với anchor-free approach
-   **Loss Functions**:
    -   Box Loss (CIoU)
    -   Classification Loss (BCE)
    -   DFL Loss (Distribution Focal Loss)

## Các Module Chính

### 1. Data Management Module

-   **Dataset Loading**: Tải dữ liệu từ Roboflow hoặc local
-   **Data Preprocessing**: Chuẩn hóa và augmentation
-   **Data Validation**: Kiểm tra tính toàn vẹn dữ liệu

### 2. Model Training Module

-   **Model Initialization**: Khởi tạo YOLOv8 với pretrained weights
-   **Training Loop**: Quá trình huấn luyện với validation
-   **Model Checkpointing**: Lưu trữ model tốt nhất
-   **Metrics Tracking**: Theo dõi loss và accuracy

### 3. Inference Module

-   **Image Preprocessing**: Chuẩn hóa input images
-   **Object Detection**: Phát hiện đối tượng với confidence scores
-   **Post-processing**: NMS (Non-Maximum Suppression)
-   **Visualization**: Vẽ bounding boxes và labels

### 4. Evaluation Module

-   **Metrics Calculation**: Precision, Recall, mAP
-   **Confusion Matrix**: Ma trận nhầm lẫn
-   **Performance Analysis**: Phân tích hiệu suất model

## Pipeline

### Training Pipeline

```
1. Data Preparation
   ├── Download dataset từ Roboflow
   ├── Validate data format
   └── Split train/valid/test

2. Model Training
   ├── Initialize YOLOv8 model
   ├── Configure hyperparameters
   ├── Train for specified epochs
   ├── Validate on validation set
   └── Save best model

3. Model Evaluation
   ├── Evaluate on test set
   ├── Calculate metrics
   └── Generate reports
```

### Inference Pipeline

```
1. Input Processing
   ├── Load image
   ├── Preprocess image
   └── Resize to model input size

2. Model Inference
   ├── Forward pass through YOLOv8
   ├── Extract predictions
   └── Apply confidence threshold

3. Post-processing
   ├── Apply NMS
   ├── Scale bounding boxes
   └── Filter by class

4. Output Generation
   ├── Draw bounding boxes
   ├── Add labels and confidence
   └── Save results
```

## Hướng Dẫn Cài Đặt

### Yêu Cầu Hệ Thống

-   **Python**: 3.8+
-   **GPU**: NVIDIA GPU với CUDA support (khuyến nghị)
-   **RAM**: Tối thiểu 8GB
-   **Storage**: 10GB trống

### Bước 1: Clone Repository

```bash
git clone <repository-url>
cd yolov11
```

### Bước 2: Cài Đặt Dependencies

```bash
# Cài đặt các thư viện cần thiết
pip install -r requirements.txt

# Hoặc cài đặt thủ công
pip install ultralytics torch torchvision opencv-python pillow matplotlib numpy roboflow
```

### Bước 3: Kiểm Tra Cài Đặt

```bash
# Kiểm tra CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Kiểm tra Ultralytics
python -c "import ultralytics; print('Ultralytics installed successfully')"
```

### Bước 4: Chuẩn Bị Dataset

```bash
# Tùy chọn 1: Sử dụng dataset có sẵn
# Đảm bảo dataset.v1i.yolov8/ hoặc dataset.v2i.yolov8/ đã có

# Tùy chọn 2: Download từ Roboflow
# Cần API key và cấu hình workspace/project từ Roboflow
# - Tạo tài khoản Roboflow: https://roboflow.com
# - Tạo workspace và project mới
# - Upload và annotate dataset
# - Lấy API key từ account settings
```

## Hướng Dẫn Sử Dụng

### 1. Huấn Luyện Model

#### Huấn Luyện Cơ Bản

```bash
# Sử dụng dataset có sẵn
python train.py --data-yaml dataset.v1i.yolov8/data.yaml

# Download và huấn luyện từ Roboflow
python train.py --api-key YOUR_API_KEY --epochs 20 --batch-size 32
```

#### Huấn Luyện Nâng Cao

```bash
python train.py \
    --api-key YOUR_API_KEY \
    --workspace YOUR_WORKSPACE_NAME \
    --project YOUR_PROJECT_NAME \
    --version 1 \
    --epochs 50 \
    --batch-size 16 \
    --model-size m
```

#### Tham Số Huấn Luyện

-   `--api-key`: API key Roboflow để download dataset
-   `--workspace`: Tên workspace Roboflow của bạn (cần cấu hình theo workspace riêng)
-   `--project`: Tên project Roboflow của bạn (cần cấu hình theo project riêng)
-   `--version`: Phiên bản dataset (mặc định: 1)
-   `--epochs`: Số epoch huấn luyện (mặc định: 20)
-   `--batch-size`: Batch size (mặc định: 32)
-   `--model-size`: Kích thước model YOLOv8 - n(nano), s(small), m(medium), l(large), x(xlarge) (mặc định: n)
-   `--data-yaml`: Đường dẫn file data.yaml (nếu dataset đã có sẵn)

### 2. Kiểm Thử Model

#### Đánh Giá Model

```bash
# Đánh giá trên tập test
python test.py --evaluate --data-yaml dataset.v1i.yolov8/data.yaml
```

#### Dự Đoán Hình Ảnh Đơn

```bash
# Dự đoán trên một hình ảnh
python test.py --image path/to/image.jpg

# Dự đoán với visualization
python test.py --image path/to/image.jpg --visualize
```

#### Dự Đoán Hàng Loạt

```bash
# Dự đoán trên nhiều hình ảnh
python test.py --image-dir path/to/images/
```

#### Tham Số Kiểm Thử

-   `--model`: Đường dẫn model đã train (.pt file)
-   `--data-yaml`: Đường dẫn file data.yaml cho evaluation
-   `--image`: Đường dẫn hình ảnh đơn cho prediction
-   `--image-dir`: Đường dẫn thư mục hình ảnh cho batch prediction
-   `--evaluate`: Đánh giá model trên tập test
-   `--visualize`: Hiển thị kết quả với matplotlib
-   `--save-result`: Lưu kết quả prediction (mặc định: True)

### 3. Workflow Hoàn Chỉnh

#### Bước 1: Huấn Luyện

```bash
# Download dataset và huấn luyện
python train.py --api-key YOUR_API_KEY --workspace YOUR_WORKSPACE --project YOUR_PROJECT --epochs 30 --model-size m
```

#### Bước 2: Đánh Giá

```bash
# Đánh giá trên tập test
python test.py --evaluate --data-yaml dataset.v1i.yolov8/data.yaml
```

#### Bước 3: Kiểm Thử

```bash
# Test trên hình ảnh đơn
python test.py --image test_images/waste.jpg --visualize

# Test trên nhiều hình ảnh
python test.py --image-dir test_images/
```

## Kết Quả và Hiệu Suất

### Metrics Đánh Giá

-   **Precision**: Tỷ lệ dự đoán đúng trong tổng số dự đoán
-   **Recall**: Tỷ lệ phát hiện đúng trong tổng số đối tượng thực
-   **mAP@0.5**: Mean Average Precision tại IoU=0.5
-   **mAP@0.5:0.95**: Mean Average Precision qua các ngưỡng IoU

### Cấu Trúc Output

```
models/
└── waste_detection/
    ├── weights/
    │   ├── best.pt      # Model weights tốt nhất
    │   └── last.pt      # Model weights cuối cùng
    ├── results.png      # Kết quả huấn luyện
    └── confusion_matrix.png

results/
├── predictions/         # Dự đoán hình ảnh đơn
└── batch_predictions/  # Dự đoán hàng loạt
```

## Xử Lý Sự Cố

### Các Vấn Đề Thường Gặp

#### 1. CUDA Out of Memory

```bash
# Giảm batch size
python train.py --batch-size 8

# Sử dụng model nhỏ hơn
python train.py --model-size n
```

#### 2. Dataset Không Tìm Thấy

-   Kiểm tra đường dẫn data.yaml
-   Đảm bảo dataset được format đúng
-   Kiểm tra quyền truy cập file

#### 3. Model Không Tìm Thấy

-   Huấn luyện model trước bằng `train.py`
-   Kiểm tra đường dẫn model trong `test.py`

### Mẹo Tối Ưu Hiệu Suất

-   Sử dụng GPU để huấn luyện nhanh hơn (CUDA)
-   Điều chỉnh batch size theo GPU memory
-   Sử dụng model nhỏ (nano/small) để huấn luyện nhanh
-   Sử dụng model lớn (large/xlarge) để độ chính xác cao hơn

## Tài Liệu Tham Khảo

### Official Documentation

-   [YOLOv8 Documentation](https://docs.ultralytics.com/)
-   [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
-   [PyTorch Documentation](https://pytorch.org/docs/)

### Research Papers

-   **YOLOv8 Paper**: "YOLOv8: A State-of-the-Art Real-Time Object Detection Model"
-   **YOLO Evolution**: "YOLO: You Only Look Once - Unified, Real-Time Object Detection"

### Tutorials & Guides

-   [Roboflow YOLOv8 Guide](https://blog.roboflow.com/how-to-train-yolov8/)
-   [Computer Vision Tutorials](https://opencv-python-tutroals.readthedocs.io/)

### Related Projects

-   [YOLOv5](https://github.com/ultralytics/yolov5)
-   [YOLOv7](https://github.com/WongKinYiu/yolov7)
-   [Roboflow Universe](https://universe.roboflow.com/)

### Community Resources

-   [Ultralytics Discord](https://discord.gg/ultralytics)
-   [PyTorch Forums](https://discuss.pytorch.org/)
-   [Computer Vision Stack Exchange](https://datascience.stackexchange.com/questions/tagged/computer-vision)

## Đóng Góp

Dự án này được phát triển cho mục đích giáo dục và nghiên cứu. Mọi đóng góp đều được chào đón:

-   Báo cáo bugs và issues
-   Đề xuất tính năng mới
-   Cải thiện documentation
-   Tối ưu hóa code

## Giấy Phép

Dự án này được phát triển cho mục đích giáo dục và nghiên cứu. Vui lòng tuân thủ các quy định về giấy phép của các thư viện được sử dụng.
