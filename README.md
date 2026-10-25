# ChamAI - Ứng dụng Nhận diện Bài thi Trắc nghiệm trên Di động

Ứng dụng web hiện đại nắn phẳng và nhận diện bài thi trắc nghiệm hoàn toàn trong trình duyệt (Client-side) tối ưu cho thiết bị di động, sẵn sàng triển khai trên **Vercel**:

* **Pipeline 4 tầng chuyên sâu**:
  1. **L1 (UVDocNet)**: Nắn phẳng tài liệu cong vênh.
  2. **L2 (YOLO QBox)**: Phát hiện vị trí từng câu hỏi trên tài liệu.
  3. **L3 (YOLO CBox)**: Phát hiện vùng các phương án trả lời trên từng câu hỏi.
  4. **L4 (AnswerChoiceNet)**: Đọc ký tự phương án (A, B, C, D) và trạng thái khoanh (*selected*, *not_selected*, *crossed_out*).
* **0đ chi phí máy chủ (Serverless / Static)**: Toàn bộ quá trình suy luận AI chạy trực tiếp trong Web Worker qua **ONNX Runtime Web**, ảnh chụp không gửi lên bất kỳ máy chủ nào.
* **Giao diện Minimalist, Light Theme, Scroll-Free**: Thiết kế chuẩn `100dvh` không cuộn trang, tối ưu thao tác 1 tay trên điện thoại, hỗ trợ thu phóng 2 ngón tay (**pinch-to-zoom**) và kéo (**pan**) mượt mà.
* **Linh hoạt CDN**: Tải trọng số mô hình từ **Cloudflare R2** hoặc **Hugging Face**.

---

## 1. Xuất Mô hình sang ONNX

Nếu muốn tự xuất lại các file mô hình từ playground `ground-25-pipeline-v3`:

```powershell
python scripts/export_to_onnx.py
```

Kết quả sẽ nằm trong thư mục `output_onnx/`:
* `uvdocnet.onnx` (~30 MB)
* `yolo-qbox.onnx` (~12 MB)
* `yolo-cbox.onnx` (~10 MB)
* `acnet.onnx` (~0.6 MB)

---

## 2. Tải Trọng số lên CDN (Cloudflare R2 hoặc Hugging Face)

Tạo thư mục `models/` trên CDN và tải 4 file `.onnx` lên:

```text
models/
├── uvdocnet.onnx
├── yolo-qbox.onnx
├── yolo-cbox.onnx
└── acnet.onnx
```

### Phương án A: Cloudflare R2 (Khuyên dùng)
* Upload vào R2 bucket (ví dụ bucket `chamai-models`).
* Bật Custom Domain hoặc R2 dev URL, ví dụ: `https://pub-xxxx.r2.dev/models`.
* **Lưu ý CORS**: Bật CORS trên R2 bucket để trình duyệt có thể fetch mô hình:
  ```json
  [
    {
      "AllowedOrigins": ["*"],
      "AllowedMethods": ["GET", "HEAD"],
      "AllowedHeaders": ["*"]
    }
  ]
  ```

### Phương án B: Hugging Face
* Tạo repository dạng Model (Public) trên Hugging Face.
* Upload 4 file vào repo.
* Đường dẫn Base URL dạng:
  `https://huggingface.co/<username>/<repo_name>/resolve/main/models`

---

## 3. Cấu hình Đường dẫn CDN trong Ứng dụng

Bạn có 2 cách rất đơn giản:

1. **Cách 1 (Ngay trên giao diện web)**:
   * Mở web, bấm vào biểu tượng **⚙ Cài đặt CDN** ở góc trên bên phải.
   * Chọn nhà cung cấp (**Cloudflare R2**, **Hugging Face**, hoặc **Tùy biến URL**).
   * Dán Base URL của bạn vào và bấm **Lưu cấu hình**. Trình duyệt sẽ tự động ghi nhớ vào `localStorage`.

2. **Cách 2 (Sửa file mặc định)**:
   * Mở file [web/config.js](file:///E:/repositories/ChamAI-test/web/config.js) và cập nhật `r2BaseUrl` hoặc `hfBaseUrl`.

---

## 4. Chạy Thử nghiệm Cục bộ

Chạy static server phục vụ thư mục `web`:

```powershell
python -m http.server 8000 --directory web
```

Mở trình duyệt tại `http://localhost:8000`.

---

## 5. Triển khai lên Vercel

Dự án đã được cấu hình sẵn trong [vercel.json](file:///E:/repositories/ChamAI-test/vercel.json):

1. Đẩy code lên GitHub:
   ```powershell
   git add .
   git commit -m "Complete replacement with Pipeline v3 and Scroll-free Light UI"
   git push origin main
   ```
2. Kết nối repo với **Vercel**:
   * Framework Preset: **Other**.
   * Output Directory: `web` (đã được cấu hình tự động qua `vercel.json`).
3. Deploy hoàn tất. Người dùng truy cập trang web trên điện thoại có thể quét bài thi ngay lập tức.
