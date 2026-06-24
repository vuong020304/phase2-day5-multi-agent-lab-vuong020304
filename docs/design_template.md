# Design Template - Multi-Agent Research System

## Problem

Hệ thống cần xử lý các yêu cầu nghiên cứu thông tin chuyên sâu (ví dụ: nghiên cứu công nghệ GraphRAG), tự động thực hiện tìm kiếm trên Web, tổng hợp dữ liệu thô, phân tích các quan điểm trái chiều, viết báo cáo học thuật hoàn chỉnh có đính kèm trích dẫn (citations) nguồn và kiểm tra lại tính xác thực trước khi trả kết quả cho người dùng.

## Why multi-agent?

Phương pháp Single-agent (chỉ dùng 1 prompt dài) gặp nhiều giới hạn khi giải quyết các tác vụ phức tạp:
1. **Lỗi Hallucination**: Agent dễ đưa ra thông tin giả mạo khi phải vừa tìm kiếm, vừa tự đánh giá và tự tổng hợp trong cùng một ngữ cảnh.
2. **Context Limits**: Việc nhồi nhét quá nhiều nguồn tài liệu tìm kiếm dễ làm loãng prompt.
3. **Thiếu khả năng tự sửa lỗi (Self-Correction)**: Single-agent không thể tự phản biện (critique) bản thảo của mình một cách khách quan.

*Giải pháp Multi-agent*: Chia nhỏ tác vụ thành các agent chuyên biệt (Researcher, Analyst, Writer, Critic) giúp giảm tải nhận thức (cognitive load), cho phép thiết lập chu trình phản biện - sửa lỗi (critique loop) lặp đi lặp lại để tối ưu chất lượng.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode & Mitigation |
|---|---|---|---|---|
| **Supervisor** | Điều phối luồng làm việc, định tuyến đi tiếp hay kết thúc | `ResearchState` | Cập nhật `route_history` với Agent tiếp theo hoặc `__end__` | Lặp vô hạn (Xử lý: Giới hạn cứng `max_iterations`). |
| **Researcher** | Tối ưu hóa từ khóa tìm kiếm, thu thập kết quả và tổng hợp ghi chú thô | Câu hỏi gốc + Feedback của Critic | `sources` (danh sách tài liệu) + `research_notes` | Lỗi mạng/API tìm kiếm (Xử lý: Tự động fallback sang Mock Search). |
| **Analyst** | Trích xuất các ý chính, đối chiếu các quan điểm phản biện và đánh giá độ tin cậy của nguồn | `research_notes` | `analysis_notes` | Phân tích hời hợt (Xử lý: Dùng System Prompt bắt buộc định dạng rõ ràng). |
| **Writer** | Tổng hợp báo cáo hoàn chỉnh dựa trên ghi chú và phân tích, định dạng markdown và trích dẫn nguồn | Ghi chú từ Researcher & Analyst + Danh sách URLs | `final_answer` (Bản thảo báo cáo) | Quên trích dẫn (Xử lý: Ràng buộc prompt bắt trích dẫn số `[1]`, `[2]`). |
| **Critic** | Đối chiếu bản thảo với nguồn thô để phát hiện lỗi factual, check độ phủ của trích dẫn | `final_answer` + `sources` | Đánh giá chấp thuận (Approved) hoặc Phản hồi chỉnh sửa | Đánh giá quá lỏng lẻo (Xử lý: System Prompt khắt khe định rõ tiêu chuẩn phê duyệt). |

## Shared State

Shared state (`ResearchState`) đóng vai trò là nguồn chân lý duy nhất (Single Source of Truth) truyền qua các node của LangGraph:
- `request`: Lưu thông tin đầu vào (câu hỏi, giới hạn nguồn, đối tượng độc giả).
- `sources`: Danh sách nguồn tài liệu tham khảo (title, url, snippet) được Researcher thu thập để Writer trích dẫn và Critic kiểm chứng.
- `research_notes`: Ghi chú nghiên cứu thô để Analyst và Writer sử dụng.
- `analysis_notes`: Kết quả phân tích sâu của Analyst để Writer tăng chất lượng chuyên môn cho bài viết.
- `final_answer`: Nơi chứa bản draft báo cáo (được cập nhật bởi Writer) và bản phê duyệt cuối cùng.
- `agent_results`: Lưu lịch sử chạy của từng agent bao gồm token usage và chi phí ước tính (USD) phục vụ benchmark.
- `route_history`: Lưu vết các bước đi để Supervisor theo dõi tiến trình và đưa ra quyết định định tuyến.

## Routing Policy

Sơ đồ di chuyển (Graph Workflow):
```text
[START] -> Supervisor (Quyết định)
             |
             +--> (Không có notes) -------> Researcher -> (Quay lại) -> Supervisor
             |
             +--> (Có notes, chưa phân tích) -> Analyst -> (Quay lại) -> Supervisor
             |
             +--> (Chưa có bản viết) ------> Writer     -> (Quay lại) -> Supervisor
             |
             +--> (Chưa kiểm duyệt) -------> Critic     -> (Quay lại) -> Supervisor
             |
             +--> (Critic yêu cầu sửa) -----> Researcher/Writer -> (Quay lại) -> Supervisor
             |
             +--> (Approved hoặc Max Iter) -> [__end__]
```

## Guardrails

- **Max iterations**: Giới hạn cứng số chu trình lặp (mặc định tối đa 6 chu kỳ trong `MAX_ITERATIONS` cấu hình ở file `.env`) để ngăn chặn việc lặp vòng tròn vô hạn.
- **Timeout**: Thiết lập thời gian chờ tối đa 60 giây (`TIMEOUT_SECONDS`) cho cả phiên chạy.
- **Retry**: Áp dụng cơ chế Retry lũy thừa (Exponential Backoff) thông qua thư viện `tenacity` cho LLM Client để khắc phục lỗi nghẽn API tạm thời.
- **Fallback**: Tự động chuyển đổi sang cơ chế tìm kiếm Mock cục bộ nếu API Tavily gặp lỗi hoặc không được cấu hình API key.
- **Validation**: Sử dụng Pydantic schemas để kiểm duyệt cấu trúc dữ liệu đầu vào và đầu ra ở tất cả các Agent.

## Benchmark Plan

- **Query**: `"Research GraphRAG state-of-the-art and write a 500-word summary"`
- **Metrics**: 
  - *Latency*: Thời gian hoàn thành tính bằng giây (giây).
  - *Cost*: Chi phí token ước tính bằng USD (quy đổi theo bảng giá gpt-4o-mini).
  - *Quality*: Điểm chất lượng từ 0-10 do LLM-as-a-judge tự động đánh giá dựa trên tiêu chí: Trả lời đủ ý, Trích dẫn chính xác, Cấu trúc rõ ràng.
  - *Execution Steps*: Số bước luân chuyển công việc (handoff steps) thực tế.
- **Expected Outcome**: Single-agent chạy nhanh hơn và rẻ hơn, nhưng Multi-agent cho điểm chất lượng cao hơn nhờ có vòng phản biện (critic) kiểm chứng thông tin và bổ sung nguồn trích dẫn đầy đủ.

