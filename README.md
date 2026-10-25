# ChamAI - Ứng dụng Nhận diện Bài thi Trắc nghiệm trên Di động

Ứng dụng web nắn phẳng và nhận diện bài thi trắc nghiệm hoàn toàn trong trình duyệt (Client-side) tối ưu cho thiết bị di động, sẵn sàng triển khai trên **Vercel**:

* **Pipeline 4 tầng chuyên sâu**:
  1. **L1 (UVDocNet)**: Nắn phẳng tài liệu cong vênh.
  2. **L2 (YOLO QBox)**: Phát hiện vị trí từng câu hỏi trên tài liệu.
  3. **L3 (YOLO CBox)**: Phát hiện vùng các phương án trả lời trên từng câu hỏi.
  4. **L4 (AnswerChoiceNet)**: Đọc ký tự phương án (A, B, C, D) và trạng thái khoanh (*selected*, *not_selected*, *crossed_out*).
* **0đ chi phí máy chủ (Serverless / Static)**: Toàn bộ quá trình suy luận AI chạy trực tiếp trong Web Worker qua **ONNX Runtime Web**, ảnh chụp không gửi lên bất kỳ máy chủ nào.
* **Mặc định tải từ Hugging Face Storage Bucket**:
  `https://huggingface.co/buckets/Khanh434/cham-models`
  - `uvdocnet.onnx` (~30 MB)
  - `yolo-qbox.onnx` (~12 MB)
  - `yolo-cbox.onnx` (~10 MB)
  - `acnet.onnx` (~0.6 MB)
* **Giao diện Minimalist, Light Theme, Scroll-Free**: Thiết kế chuẩn `100dvh` không cuộn trang, tối ưu thao tác 1 tay trên điện thoại, hỗ trợ thu phóng 2 ngón tay (**pinch-to-zoom**) và kéo (**pan**) mượt mà.

---

## 1. Xuất Mô hình sang ONNX (Tùy chọn)

Script chuyển đổi mô hình từ `ground-25-pipeline-v3`:

```powershell
python scripts/export_to_onnx.py
```

Kết quả nằm trong thư mục `output_onnx/` sẵn sàng tải lên bucket.

---

## 2. Chạy Thử nghiệm Cục bộ

Phục vụ thư mục `web`:

```powershell
python -m http.server 8000 --directory web
```

Mở trình duyệt tại `http://localhost:8000`. Bấm **Nạp model** hoặc chụp/chọn ảnh để ứng dụng tự động tải mô hình từ bucket `Khanh434/cham-models` và thực thi nhận diện.

---

## 3. Triển khai lên Vercel

Dự án đã được cấu hình trong [vercel.json](file:///E:/repositories/ChamAI-test/vercel.json):

1. Đẩy code lên GitHub:
   ```powershell
   git add .
   git commit -m "Configure default Hugging Face Bucket Khanh434/cham-models"
   git push origin main
   ```
2. Import repo vào **Vercel**:
   * Framework Preset: **Other**.
   * Output Directory: `web`.
3. Deploy hoàn tất. Trang web hoạt động ngay lập tức mà không cần cấu hình thêm bất kỳ biến môi trường nào.
