# Kế hoạch v2 triển khai Trích dẫn nguồn cho Legal RAG
*[Bản sửa đổi — xem changelog cuối file]*

## 1. Kết luận ngắn
- Triển khai cơ chế trích dẫn nguồn theo mô hình **single-pass generation**: chỉ 1 lượt gọi LLM để sinh câu trả lời kèm chỉ mục trích dẫn. **Không có retry hay LLM call bổ sung ở bất kỳ nhánh xử lý nào**, kể cả khi validator phát hiện lỗi.
- Bổ sung Citation Validator deterministic ở backend để **phát hiện và chặn citation sai định dạng/chỉ mục** mà không cần thêm LLM call.
- Giữ nguyên contract hiện có với `DocumentSource`/`ChatResponse`, chỉ mở rộng dữ liệu runtime theo hướng backward-compatible.

## 2. Mục tiêu, phạm vi, và ràng buộc

### 2.1 Mục tiêu
- Mọi claim pháp lý trong câu trả lời đều có nguồn dạng `[n]`.
- Người dùng bấm được từ `[n]` đến đúng tài liệu tham chiếu ở UI.
- Đảm bảo traceability: từ câu trả lời quay ngược về chunk nguồn cụ thể.

### 2.2 Phạm vi
- Backend: `retriever.py`, `generator.py`, stream formatter, response contract.
- Frontend NextChat: parser inline citation và panel nguồn.
- Evaluation: thêm bộ đo và ngưỡng chấp nhận cho Citation Accuracy.

### 2.3 Ràng buộc kỹ thuật bắt buộc
- Không phát sinh LLM/API calls không cần thiết ở runtime. **Validator không trigger retry.**
- Tuân thủ async/await và không block event loop.
- Không trộn prompt formatting và stream formatting vào cùng một khối logic (modularity).
- **Citation index được gán sau rerank, trước khi build context. K trong prompt = K trong `sources` event. Đây là invariant cứng, phải đảm bảo bằng assertion.**

## 3. Thiết kế dữ liệu trích dẫn (không phá contract)

### 3.1 Quy tắc ID
- **Không ghi đè `DocumentSource.id`** từ retriever.
- Thêm `citation_index` runtime **sau rerank**, theo thứ tự 1..K để mô hình và UI dùng chung.

Ví dụ object nội bộ khi vào generator:
```python
{
  "id": "chunk-uuid-or-cid",        # giữ nguyên id gốc
  "citation_index": 1,               # dùng cho [1] — gán sau rerank
  "title": "Điều 6 Luật ...",
  "url": "...",
  "score": 0.88,
  "content": "..."
}
```

> **Invariant:** `len(sources_after_rerank) == K` phải nhất quán giữa prompt và SSE payload. Nếu retriever trả về ít hơn k documents, K được cập nhật theo số thực tế. Thêm assertion tại call-site của context builder.

### 3.2 Contract trả về
- Vẫn trả `sources: List[DocumentSource]` như hiện tại.
- Thêm trường tùy chọn `citation_index` trong payload stream event `sources` (không bắt buộc thay schema Pydantic nếu chỉ dùng trong SSE metadata nội bộ).
- Thêm trường tùy chọn `snippet` (chuỗi rút gọn, không phải full content) trong payload stream event `sources` để frontend render hover preview mà không cần bật `include_content`.

## 4. Kiến trúc pipeline v2

### 4.1 Retriever stage
- Lấy top-k theo intent (**các giá trị này là điểm khởi đầu thực nghiệm, cần điều chỉnh sau eval pha C**):
  - `LEGAL_LOOKUP/DEFINITION`: 4-6
  - `PROCEDURE/SUMMARIZE`: 6-8
  - `COMPARE/MULTI_HOP`: 8-12
- Dùng kết quả rerank hiện có.
- **Sau rerank:** gán `citation_index` 1..K theo thứ tự score giảm dần, chỉ với K documents cuối cùng được đưa vào context. K thực tế có thể thấp hơn range trên nếu retriever không đủ kết quả — đây là hành vi bình thường, không phải lỗi.

### 4.2 Citation Context Builder (module riêng)
- Build context theo format ổn định **sau khi đã gán citation_index**:
```text
[1] title=... id=... score=...
content=...

[2] title=... id=... score=...
content=...
```
- Tối ưu cost:
  - Cắt độ dài content theo ngân sách token.
  - Ưu tiên giữ nguyên câu chứa thực thể pháp lý và điều khoản.

### 4.3 Generator Prompt (single-pass)
- Áp dụng kỹ thuật **System Instruction with Constraint Examples** (instructed prompting kết hợp positive/negative exemplars — không phải few-shot theo nghĩa kỹ thuật) bằng cách chèn khối lệnh khắt khe này vào System Prompt:
  ```text
  KHI TRÍCH DẪN NGUỒN, BẠN BẮT BUỘC PHẢI TUÂN THỦ CÁC QUY TẮC SAU:
  1. Mỗi claim pháp lý đều phải có trích dẫn `[n]`, với `n` chỉ được nằm trong danh sách tài liệu [1..K] đã cung cấp. Tuyệt đối không bịa ra số ngoài danh sách.
  2. Đặt trích dẫn NGAY TRƯỚC dấu câu (dấu chấm, dấu phẩy).
  3. Khi trích dẫn nhiều nguồn cùng lúc, viết liền nhau (VD: `[1][2]`). Tuyệt đối không dùng định dạng có dấu phẩy hay khoảng trắng (VD: `[1, 2]`, `[1 - 2]`, `[1-3]`).
  4. Không được dồn trích dẫn cho cả đoạn. Mỗi câu chứa claim pháp lý phải tự mang ít nhất một trích dẫn của chính câu đó.
  5. Nếu dữ liệu không đủ để trả lời, CHỈ ĐƯỢC PHÉP đáp đúng một câu: "Dữ liệu hiện tại không đề cập đến quy định này." Tuyệt đối không suy đoán, không xin lỗi hay giải thích vòng vo.

  VÍ DỤ SAI:
  - Người điều khiển bị phạt tiền từ 2-3 triệu. [1] (Lỗi: Đặt trích dẫn sai vị trí, đứng sau dấu chấm)
  - Tốc độ tối đa trong khu đông dân cư là 50km/h [1, 2]. (Lỗi: Dùng dấu phẩy giữa các thẻ trích dẫn)
  - Xin lỗi, theo dữ liệu tôi nhận được thì không có thông tin. (Lỗi: Trả lời vòng vo, có từ xin lỗi)

  VÍ DỤ ĐÚNG:
  - Theo quy định mới, vi phạm nồng độ cồn sẽ bị tước bằng lái [1][2].
  - Mức phạt này được áp dụng nghiêm ngặt đối với cả xe máy và xe ô tô [3].
  - Dữ liệu hiện tại không đề cập đến quy định này.
  ```
- Không thêm post-LLM call để sửa đáp án.

> **Lý do dùng "Instructed Prompting with Exemplars" thay vì few-shot thuần:** Few-shot thuần cung cấp input/output pairs trong user turn để model học pattern ngầm. Kỹ thuật ở đây là explicit rule enforcement + positive/negative examples trong system turn — mục đích là constraint, không phải pattern learning. Gọi đúng tên giúp tránh implement sai (chèn examples vào user turn làm tốn token và giảm stability).

### 4.4 Citation Validator deterministic (không LLM)
- Parse tất cả pattern `[n]` trong output.
- Check hợp lệ:
  - `n` có thuộc `1..K` không.
  - Nếu câu có claim pháp lý mà thiếu citation, gắn cảnh báo chất lượng.
- **Heuristic phát hiện "claim pháp lý"** (phải document rõ để unit test cover được):
  - **Keyword trigger:** "theo quy định", "theo điều", "bị phạt", "có hiệu lực", "nghị định", "thông tư", "luật", "bộ luật", "quyết định số", và các biến thể viết hoa/thường.
  - **Pattern trigger:** câu chứa số tiền (VD: `\d+[\.,]?\d*\s*(triệu|nghìn|đồng|VND)`), thời hạn (VD: `\d+\s*(ngày|tháng|năm)`), mức phạt, điều khoản dạng `[Dd]iều\s+\d+`.
  - **Exclusion:** câu trong fenced code block (` ``` `) hoặc inline code (`` ` ``) bị bỏ qua hoàn toàn.
  - Heuristic này là **best-effort** — false positive/negative là chấp nhận được ở v2, cần tinh chỉnh dựa trên eval kết quả pha C.
- **Cơ chế xử lý khi phát hiện lỗi (không có retry, không có LLM call bổ sung):**
  - **Safe mode (mặc định):** Chuẩn hóa định dạng citation về dạng hợp lệ (`[1][2]`), loại bỏ citation ngoài range, giữ nguyên nội dung text còn lại, hạ `confidence` xuống mức cảnh báo, gắn `validation_warning` vào response metadata. Log server-side.
  - **Hard mode (opt-in):** Nếu phát hiện citation ngoài range hoặc thiếu citation ở câu claim pháp lý, thay toàn bộ answer bằng câu cố định: "Dữ liệu hiện tại không đề cập đến quy định này.", gắn `validation_warning` và log server-side.
  - Cả hai mode: **answer vẫn được trả về client** — validator không trigger retry và không gọi thêm LLM.

### 4.5 Stream Formatter (module riêng)
- **Ordering guarantee:** Phải `yield sources_event` đồng bộ (explicit await) trước khi bắt đầu yield bất kỳ token nào của answer. Không dùng `asyncio.gather` hay `background_task` để song song hai bước này — ordering không được đảm bảo trong concurrency context.
- `sources` payload gồm `citation_index`, `id`, `title`, `url`, `score`, `snippet`.
- `snippet` được cắt phía server từ chunk đã rerank (VD 180-300 ký tự quanh câu có thông tin chính), không gửi full `content` để tránh tăng băng thông.
- Giữ format event ổn định để frontend không cần đoán schema.

## 5. Frontend hiển thị theo phong cách NotebookLM-like (tham khảo UX)
- Inline badges `[1] [2]` trong tin nhắn bot.
- Click badge sẽ scroll đến card nguồn tương ứng.
- Hover badge hiển thị preview snippet 1-2 câu.
- Dưới câu trả lời có mục "Tài liệu liên quan" theo thứ tự `citation_index`.

**Edge cases bắt buộc phải xử lý:**
- **Citation trong code block:** Parser phải nhận biết fenced code (` ``` ... ``` `) và inline code (`` `...` ``). Bất kỳ `[n]` nào trong code block không được render thành badge, giữ nguyên dạng text.
- **`url` null hoặc rỗng:** Source card vẫn hiển thị `title` và `snippet` (lấy từ payload `sources.snippet`). Không render link/button nếu url rỗng. Không crash.
- **Badge lặp lại (cùng citation_index xuất hiện nhiều lần):** Tất cả badges cùng index đều scroll đến cùng một card. Card không bị duplicate trong source panel.

Lưu ý:
- Đây là tham chiếu phong cách UX NotebookLM (source-grounded answering và clickable references), không sao chép UI/asset.

## 6. Chiến lược tiết kiệm chi phí

### 6.1 Runtime
- Không thêm LLM call cho bước verify citation.
- Giảm context theo budget động:
  - budget token cho context tách theo intent.
  - dừng nạp chunk khi đạt ngưỡng ngân sách.
- Giới hạn số citation tối đa trong output để tránh kéo dài không cần thiết.

### 6.2 Retrieval và rerank
- Duy trì hybrid dense+sparse + fusion hiện có.
- Chỉ rerank top-N hợp lý (thực nghiệm N=20/30 tùy intent), không rerank toàn bộ.

### 6.3 Caching
- Cache kết quả retrieval theo `(normalized_query, intent, top_k)` trong TTL ngắn để giảm truy vấn lặp.
- Cache prompt template đã compile cho từng intent để giảm xử lý string runtime.

## 7. Chống ảo giác và guardrails
- Rule 1: Câu có nội dung pháp lý kết luận phải có citation.
- Rule 2: Citation phải thuộc tập nguồn đã cấp.
- Rule 3: Nếu không đủ bằng chứng thì trả thiếu dữ liệu.
- Rule 4: Với COMPARE/MULTI_HOP, ưu tiên trích ít nhất 2 nguồn khác nhau khi có thể.

## 8. Kế hoạch triển khai theo pha

### Pha 0 (nửa ngày, trước Pha A): Đo baseline
- **Bắt buộc:** Chạy evaluation trên hệ thống hiện tại (chưa có citation feature) và ghi số vào `evaluation/baseline.json`.
- Metrics cần đo: Faithfulness, Answer Relevancy, Latency p95.
- Commit kết quả vào repo trước khi bắt đầu bất kỳ code change nào.

### Pha A (1-2 ngày): Backend core
- Tạo `citation_context_builder`.
- Tạo `citation_validator` kèm heuristic đã định nghĩa ở mục 4.4.
- Chuẩn hóa stream event `sources` với ordering guarantee.
- Thêm assertion `len(sources) == K` tại call-site.

### Pha B (1-2 ngày): Frontend
- Parse `[n]` thành badge có tương tác (markdown-aware, bỏ qua code block).
- Render panel tài liệu liên quan + preview snippet.
- Xử lý đủ 3 edge cases đã liệt kê ở mục 5.

### Pha C (1-2 ngày): Eval và hardening
- Bổ sung test parser/validator.
- Chạy evaluation so sánh với baseline (Pha 0).
- Chốt ngưỡng release và điều chỉnh top-k nếu cần.

## 9. Test plan và tiêu chí chấp nhận

### 9.1 Unit tests
- Parser trích `[n]` đúng với các trường hợp liền nhau, lặp lại, malformed.
- Parser **bỏ qua** `[n]` trong fenced code và inline code.
- Validator phát hiện citation ngoài range.
- Validator phát hiện câu claim thiếu citation theo heuristic đã định nghĩa ở mục 4.4.
- Assertion `len(sources) == K` hoạt động đúng khi retriever trả về ít hơn k.

### 9.2 Contract tests
- SSE event `sources` **luôn** có trước token đầu tiên của answer (integration test kiểm tra ordering).
- Client parse được `citation_index` và mapping đúng sang nguồn.

### 9.3 Evaluation metrics
- Citation Accuracy >= 0.95 trên bộ eval nội bộ.
- Faithfulness >= `max(0.85, baseline_faithfulness - 0.01)` với `baseline_faithfulness` lấy từ `evaluation/baseline.json`.
- Latency p95 tăng không quá 10%.
- Chi phí token/response không tăng quá 5%.

> **Lưu ý quan trọng:** Không được merge feature nếu chưa có số baseline từ Pha 0. "Faithfulness không giảm so với baseline" là tiêu chí không đo được nếu baseline không được pin cụ thể.

## 10. Rủi ro và cách giảm thiểu
- Rủi ro: model vẫn có thể bỏ sót citation ở câu nối.
  - Giảm thiểu: validator + cảnh báo + test case tiếng Việt pháp lý.
- Rủi ro: top-k thấp gây thiếu nguồn cho câu hỏi đa bước.
  - Giảm thiểu: top-k theo intent + fallback web search theo confidence hiện có.
- Rủi ro: frontend parse nhầm `[n]` trong code block.
  - Giảm thiểu: parser markdown-aware, bỏ qua fenced code và inline code (spec rõ tại mục 5).
- **Rủi ro (thêm mới):** Citation index lệch giữa prompt và SSE payload do K thực tế khác K dự kiến.
  - Giảm thiểu: Gán citation_index sau rerank, assertion tại call-site, contract test ordering.

## 11. Danh sách thay đổi đề xuất theo file
- `backend/app/services/generator.py`
  - Tách context builder, stream formatter, validator call-site.
- `backend/app/services/retriever.py`
  - Chuẩn hóa output metadata cho citation mapping; gán `citation_index` sau rerank.
- `backend/app/schemas.py`
  - Giữ nguyên contract chính; nếu cần thì mở rộng optional field theo hướng tương thích ngược.
- `chatbot-ui/NextChat/...`
  - Thêm citation badge renderer (markdown-aware) + source panel component + edge case handling.
- `evaluation/legal_rag_evaluator.py`
  - Bổ sung report tách riêng lỗi citation ngoài range và lỗi thiếu citation.
- **`evaluation/baseline.json` (thêm mới)**
  - File lưu kết quả baseline từ Pha 0, commit vào repo.

## 12. Tài liệu tham khảo

### 12.1 Nguồn nội bộ dự án
- `claude.md` (kiến trúc hệ thống, yêu cầu async/modularity, citation accuracy metric).
- `rules.md` (nguyên tắc không suy đoán, ưu tiên đúng đắn và thực dụng).
- `backend/app/schemas.py` (`DocumentSource`, `ChatResponse`).

### 12.2 Nguồn bên ngoài
- Qdrant docs: Hybrid and Multi-Stage Queries (prefetch, fusion RRF, dense+sparse).
  - https://qdrant.tech/documentation/search/hybrid-queries/
- Anthropic Engineering: Introducing Contextual Retrieval (hybrid retrieval, reranking, cost/latency trade-off, eval-first).
  - https://www.anthropic.com/engineering/contextual-retrieval
- Pinecone Learn: Retrieval-Augmented Generation (grounding, trust, source citations, cost trade-offs).
  - https://www.pinecone.io/learn/retrieval-augmented-generation/

---

## Changelog (so với bản gốc)

| # | Vị trí | Thay đổi | Lý do |
|---|--------|----------|-------|
| 1 | Mục 1, 2.3, 4.4 | Làm rõ validator không trigger retry, answer luôn được trả | Giải quyết mâu thuẫn "single-pass" vs "Hard mode" |
| 2 | Mục 4.3 | Đổi tên kỹ thuật từ "Few-Shot Prompting" → "System Instruction with Constraint Examples" + giải thích | Thuật ngữ sai có thể dẫn đến implement sai |
| 3 | Mục 4.4 | Thêm định nghĩa heuristic cụ thể cho "claim pháp lý" | Heuristic bỏ ngỏ không thể test, không thể đo metric |
| 4 | Mục 3.1, 4.1, 2.3 | Thêm invariant K, gán citation_index sau rerank, assertion requirement | Phòng bug race condition citation index lệch |
| 5 | Mục 4.5 | Thêm ordering guarantee specification (explicit yield, không dùng gather) | Async context không đảm bảo ordering mà không có explicit barrier |
| 6 | Mục 5 | Thêm subsection "Edge cases" với 3 rule cụ thể | Spec frontend thiếu edge case dẫn đến implement không nhất quán |
| 7 | Mục 8, 9.3 | Thêm Pha 0 (đo baseline) và gate Faithfulness dựa trên baseline | Tiêu chí "không giảm so với baseline" phải đo được và có file baseline đi kèm |
| 8 | Mục 4.3 | Loại bỏ rule "dồn citation cuối đoạn", thay bằng rule claim-level citation | Xóa mâu thuẫn với mục tiêu "mỗi claim đều có citation" |
| 9 | Mục 4.4 | Định nghĩa lại Safe/Hard mode để cả hai đều chặn citation sai | Xóa mâu thuẫn "chặn lỗi" nhưng hard mode vẫn giữ lỗi |
| 10 | Mục 3.2, 4.5, 5 | Bổ sung `sources.snippet` và chuẩn hóa hiển thị preview theo snippet | Tránh vênh giữa yêu cầu UI preview và payload không có dữ liệu |
| 11 | Mục 8, 9.3, 11 | Chuẩn hóa đường dẫn baseline thành `evaluation/baseline.json` và bỏ placeholder [X]% | Đảm bảo tiêu chí release đo được và khớp cấu trúc repo |