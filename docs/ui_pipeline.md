# UI Integration Pipeline: Kiến trúc & Kế hoạch Triển khai
*Tài liệu hoạch định chi tiết (Technical Roadmap) quy trình tích hợp Frontend vào Law Chatbot RAG.*

---

## 1. Triết Lý Thiết Kế (Design Philosophy)
- **Tương thích chuẩn (Protocol Compatibility):** Ép Backend FastAPI của chúng ta tuân thủ 100% chuẩn giao tiếp API của OpenAI (`OpenAI-Compatible API`). Điều này là chìa khóa để có thể "cắm" bất kỳ UI mã nguồn mở xịn xò nào (ChatGPT Next Web, Lobe Chat, OpenWebUI) vào hệ thống mà không cần đụng/sửa code phức tạp của phía Frontend.
- **Tách biệt rõ ràng (Separation of Concerns):** UI chỉ làm duy nhất một việc: Hiển thị và cung cấp công cụ chat. Toàn bộ chất xám về quy trình tìm kiếm Luật (RAG), xây dựng Prompt, Sinh câu trả lời sẽ nằm chặt chẽ ở Backend.

## 2. Kiến Trúc Luồng Dữ Liệu (Data-Flow Architecture)

Quy trình chuẩn khi một câu hỏi được gửi xuất phát từ UI:

1. **[Client UI - Giao diện]**
   - Đóng gói câu hỏi của User (và lịch sử hội thoại) vào khối JSON chuẩn: 
     `{"model": "law-model", "messages": [...], "stream": true}`
   - Gửi **POST** Request đến `http://127.0.0.1:8000/v1/chat/completions`.

2. **[Backend Gateway - API Router]** (Cần xây dựng)
   - Tiếp nhận JSON bằng Pydantic model (`ChatCompletionRequest`).
   - Lọc lấy `user_message` (câu hỏi cuối cùng).

3. **[RAG Core - Xử lý não bộ]** (Đã có sẵn, cần tinh chỉnh)
   - **Retriever:** Tìm chunk luật tương ứng từ Qdrant (`retriever.py`).
   - **Generator:** Tạo một `System Schema` bí mật chứa tài liệu RAG, tiêm vào Prompt cùng với câu lệnh của người dùng (`generator.py`).
   - Gọi mô hình LLM (Gemini, OpenAI, Local model) để sinh đáp án.

4. **[Response Formatter - Đóng gói trả về]** (Cần xây dựng)
   - Chuyển hóa đáp án trả về thành cấu trúc `choices[0].message.content` (Non-stream).
   - Hoặc chia nhỏ thành Server-Sent Events (SSE) `delta.content` nếu dùng chế độ **Streaming** để UI hiện chữ mượt mà.

---

## 3. Các Bước Triển Khai Thực Tế (Execution Pipeline)

Để đảm bảo an toàn tuyệt đối và không gây lỗi (không ảo giác), chúng ta sẽ làm từng bước chậm rãi và có kiểm chứng:

### Phase 1: Mở rộng Backend (OpenAI Wrapper)
- **Tạo Pydantic Schema:** Viết các Class (schema) chứa định dạng Request/Response của OpenAI tại thư mục API.
- **Xây dựng Endpoint Router:** Mở file `main.py`, tạo endpoint `@app.post("/v1/chat/completions")`.
- **Hợp nhất Generator:** Nối kết quả đầu ra của `generator.py` với Response Schema vừa tạo. Xử lý thử JSON tĩnh (chưa có Stream) trước.

### Phase 2: Setup Frontend (UI Setup)
- Clone mã nguồn (như ChatGPT Next Web) bằng Git vào thư mục Frontend (cùng cấp hoặc trong Chatbot-UI).
- Thiết lập biến môi trường (Environment Variables) `.env.local` trỏ `BASE_URL` về `http://127.0.0.1:8000` của cổng FastAPI.
- Chạy môi trường Node.js.

### Phase 3: Hoàn thiện Streaming (Trải nghiệm cao cấp)
- Chat văn bản Luật khá dài, nếu không có stream UI sẽ đơ (blocking).
- Cải tiến tính năng StreamingResponse của FastAPI trong endpoint `/v1/chat/completions` để truyền từng chunk token về cho UI ngay lập tức. Đây là phần khó và quyết định 90% cảm giác "mượt mà" của Chatbot.

### Phase 4: System Promping nâng cao cho UI
- Dọn dẹp lại đoạn đóng gói RAG Prompt để tránh việc gửi lịch sử chat rác lên Qdrant, đảm bảo mô hình nhận diện chuẩn xác ý định của người hỏi luật.

---
## 4. Quyết định bước tiếp theo
Dựa trên pipeline này, bước tiếp theo an toàn và chuẩn khoa học nhất là **Phase 1: Chuẩn bị các Pydantic Models mô phỏng cấu trúc OpenAI ở Backend**. Nền móng vững chắc thì ghép UI nào vào cũng chạy ngay lập tức.
