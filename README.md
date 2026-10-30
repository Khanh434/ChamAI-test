## ChamAI - Ứng dụng Nhận diện Bài thi Trắc nghiệm trên Di động

Ứng dụng web nắn phẳng và nhận diện bài thi trắc nghiệm hoàn toàn trong trình duyệt tối ưu cho thiết bị di động:

* **Pipeline 4 tầng chuyên sâu**:
  1. **L1 (UVDocNet)**: Nắn phẳng tài liệu cong vênh.
  2. **L2 (YOLO-qbox)**: Phát hiện vị trí từng câu hỏi trên tài liệu.
  3. **L3 (YOLO-cbox)**: Phát hiện vùng các phương án trả lời trên từng câu hỏi.
  4. **L4 (ACNet)**: Đọc ký tự phương án (A, B, C, D) và trạng thái khoanh (*selected*, *not_selected*, *crossed_out*).