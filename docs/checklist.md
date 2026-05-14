# Checklist: Khôi phục độ phản hồi của chatbot trên máy 20 GB RAM

## Mục tiêu
- [ ] Phân biệt rõ lỗi contract streaming giữa frontend/backend với lỗi quá tải do dataset mới.
- [ ] Xác định bottleneck thật sự trước khi sửa.
- [ ] Giữ hệ thống hoạt động ổn định trên máy 20 GB RAM, không OOM và không treo UI.

## Checklist thực thi
- [ ] Reproduce và lấy baseline trên máy 20 GB RAM với 3 nhóm query: chitchat, legal lookup, compare/multi-hop.s
- [ ] Ghi nhận TTFB, time-to-first-chunk, total response time, status code, response headers, peak RAM và CPU.
- [ ] Kiểm tra request từ NextChat đi đúng BASE_URL, DEFAULT_MODEL và stream=true.
- [ ] Xác minh backend trả text/event-stream khi stream bật và kết thúc bằng [DONE].
- [ ] Gắn timing probe vào các phase: QueryReflector.reflect, route_retrieval, LocalReranker.rerank, GraphRAGRetriever.search, _load_relationships, generator streaming, memory.add_turn.
- [ ] Đo footprint bộ nhớ của GraphRAG trên dataset mới.
- [ ] So sánh khi bật/tắt GRAPHRAG_ENABLED.
- [ ] So sánh khi bật/tắt local reranker.
- [ ] Kiểm tra độ mới và kích thước của entities.parquet, relationships.parquet, lancedb.
- [ ] Chọn nhánh khắc phục theo dữ liệu đo được.
- [ ] Nếu contract sai, sửa streaming/config trước.
- [ ] Nếu reranker là bottleneck, giảm tải hoặc đẩy nó ra khỏi hot path.
- [ ] Nếu GraphRAG load gây áp lực RAM, tránh nạp toàn bộ relationships vào RAM, ưu tiên lazy-load hoặc giới hạn theo intent.
- [ ] Nếu artifact GraphRAG bị stale/mismatch, rebuild pipeline trước rồi mới tinh chỉnh runtime.
- [ ] Chạy lại baseline sau chỉnh sửa.
- [ ] Chốt ngưỡng vận hành an toàn cho máy 20 GB RAM.
- [ ] Xác nhận UI không còn treo khi gửi truy vấn dài hoặc truy vấn GraphRAG.

## Relevant files
- [backend/app/main.py](backend/app/main.py) - `/v1/chat/completions`, `/api/chat`, startup wiring cho reranker và GraphRAG.
- [backend/app/api/routes.py](backend/app/api/routes.py) - `handle_chat` và `handle_retrieve`.
- [backend/app/services/query_router.py](backend/app/services/query_router.py) - route selection và nhánh `both`.
- [backend/app/services/local_reranker.py](backend/app/services/local_reranker.py) - reranker sync chạy qua executor.
- [backend/app/services/graph_retriever.py](backend/app/services/graph_retriever.py) - `_load_relationships` nạp parquet vào RAM.
- [backend/app/core/config.py](backend/app/core/config.py) - path GraphRAG và các knob latency/memory.
- [chatbot-ui/NextChat/app/client/platforms/openai.ts](chatbot-ui/NextChat/app/client/platforms/openai.ts) - payload request và cờ stream.
- [chatbot-ui/NextChat/app/utils/chat.ts](chatbot-ui/NextChat/app/utils/chat.ts) - SSE, timeout 60s, và hành vi chờ [DONE].
- [chatbot-ui/NextChat/app/config/server.ts](chatbot-ui/NextChat/app/config/server.ts) - BASE_URL, CUSTOM_MODELS, DEFAULT_MODEL.
- [data_pipelines/graph_rag/scripts/03_kaggle_build_relationships.py](data_pipelines/graph_rag/scripts/03_kaggle_build_relationships.py) - tạo artifact relationships lớn.
- [data_pipelines/graph_rag/scripts/04_kaggle_build_communities.py](data_pipelines/graph_rag/scripts/04_kaggle_build_communities.py) - clustering có rủi ro RAM cao.
- [data_pipelines/graph_rag/scripts/06_build_embeddings_lancedb.py](data_pipelines/graph_rag/scripts/06_build_embeddings_lancedb.py) - build index GraphRAG.

## Verification
- [ ] Chạy lại một COMPARE/MULTI_HOP query và một LEGAL_LOOKUP query, rồi so sánh TTFB, first chunk, total time và peak RAM.
- [ ] Xác nhận response là text/event-stream khi stream=true và stream kết thúc bằng [DONE].
- [ ] Chạy backend trong 3 biến thể: đầy đủ, tắt GraphRAG, tắt local reranker.
- [ ] Ghi nhận delta latency và RAM để xác định đúng nút nghẽn.
- [ ] Kiểm tra /metrics và log phase timing.
- [ ] Đặt ngưỡng vận hành an toàn cho máy 20 GB RAM.

## Decisions
- [ ] Ưu tiên khôi phục phản hồi ổn định trước, sau đó mới tối ưu chất lượng GraphRAG.
- [ ] Giữ thay đổi nhỏ, có thể hoàn nguyên.
- [ ] Xem rebuild artifact dataset là một nhánh riêng nếu freshness check không khớp với dataset mới.
