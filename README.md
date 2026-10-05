# Nhận diện tài liệu trên điện thoại

Ứng dụng web xử lý ảnh hoàn toàn trong trình duyệt điện thoại:

1. OpenCV.js tìm và căn chỉnh tài liệu.
2. ONNX Runtime Web chạy model `qbox` ở kích thước 1024 × 1024.
3. Ảnh và kết quả không được gửi tới backend; trang không cần API hoặc cơ sở dữ liệu.

## Tài nguyên

- OpenCV.js và WebAssembly được phục vụ như static assets từ Vercel CDN trong `web/opencv/`.
- Model YOLO được tải trực tiếp từ [Hugging Face](https://huggingface.co/Khanh434/YOLOv8n-doclaynet-fine-tuned-chamai); ứng dụng không đóng gói file model vào repository hoặc Vercel deployment.
- ONNX Runtime Web 1.22.0 và WebAssembly của runtime được tải từ jsDelivr CDN.
- OpenCV.js, ONNX Runtime và model chỉ được tải khi nhấn **Nạp model**. Model được lấy từ repository Hugging Face công khai; ảnh đầu vào vẫn ở trên thiết bị.

Model đang được ghim vào revision `24683b7e898e3b2b03178272e03273efad9bc955` để deployment ổn định. Sau khi cập nhật model trên Hugging Face, đổi revision trong `MODEL_URL` ở `web/static/pipeline-worker.js`.

## Deploy lên Vercel

1. Đưa repository lên GitHub/GitLab hoặc kết nối repository Git hiện có trong Vercel.
2. Import project trong Vercel. Chọn framework **Other**; `vercel.json` đã cấu hình thư mục output là `web`.
3. Không cần build command, biến môi trường, backend hay cấu hình lưu trữ.
4. Deploy. Đường dẫn gốc mở giao diện; Vercel phân phối model và OpenCV từ CDN.

Có thể deploy bằng Vercel CLI:

```powershell
npx vercel
npx vercel --prod
```

## Chạy thử trên máy tính

Phục vụ thư mục `web` bằng HTTP server tĩnh. Ví dụ nếu đã cài Python:

```powershell
python -m http.server 8000 --directory web
```

Mở `http://localhost:8000`. Nhấn **Nạp model** trước khi chụp hoặc chọn ảnh. Thiết bị cần kết nối Internet để tải OpenCV.js, ONNX Runtime và model.

## Model và pipeline

Model có một lớp (`qbox`) với output `[1, 5, 21504]`; ngưỡng confidence là `0.8`, NMS IoU là `0.45`. Ảnh đầu vào giới hạn cạnh dài tối đa 1800 px; OpenCV tìm góc tài liệu trên ảnh tối đa 1000 px, căn phối cảnh, deskew và thử hiệu chỉnh cong nhẹ trên lưới 3 × 3. Nếu không tìm được góc hoặc tín hiệu đường chữ phù hợp, ảnh đã resize vẫn được đưa vào YOLO.

Các script Python trong thư mục gốc và `new/` là công cụ thử nghiệm riêng, không nằm trong output deploy của Vercel.

`requirements.txt` chỉ dành cho tiện ích Python `doc_align.py`; `new/requirements.txt` thuộc pipeline thử nghiệm trong thư mục `new/`.
