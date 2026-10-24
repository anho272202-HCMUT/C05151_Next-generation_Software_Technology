# Hướng dẫn gán nhãn người (kiểm tra độ tin cậy của LLM-as-judge)

File cần điền: `eval/human_labels.csv` (30 dòng). Mở bằng Numbers hoặc Excel, điền xong thì **lưu lại đúng định dạng CSV**. Thời gian khoảng 20–25 phút.

Mỗi dòng là một critique mà hệ thống đã sinh cho một task (bài báo × trục). File không hiện quyết định của LLM để bạn chấm độc lập ("mù").

| Cột | Nội dung |
|---|---|
| `critique`, `debate_question` | Điểm yếu và câu hỏi tranh luận mà hệ thống viết |
| `cited_sentences` | Các câu hệ thống trích từ PDF (đã kiểm chứng là có thật trong bài) |
| `ground_truth_weaknesses` | Các điểm yếu nhóm đã gán nhãn cho task này, kèm số trang |

**Hai cột cần điền, mỗi ô ghi `yes` hoặc `no`:**

1. `q1_on_axis_and_supported`: ghi `yes` nếu critique **đúng trục** (assumptions / fairness / data_reliability / generalizability / reproducibility) **và** các câu trích **thực sự ủng hộ** điểm yếu đó. Ghi `no` nếu lệch trục, quá chung chung, hoặc câu trích không chứng minh được.
   → Dùng để kiểm tra Evaluator chạy trong hệ thống (bước quyết định accept/reject).
2. `q2_matches_ground_truth`: ghi `yes` nếu critique nêu **cùng một vấn đề** với ít nhất một điểm yếu trong `ground_truth_weaknesses` (diễn đạt khác vẫn được). Ghi `no` nếu là vấn đề khác hoặc chung chung hơn.
   → Dùng để kiểm tra judge chấm điểm trong harness đánh giá.

Cột `notes` không bắt buộc, dùng khi bạn phân vân.

Điền xong, chạy:

```bash
uv run python -m critic_agent.evaluation agreement
uv run python -m critic_agent.evaluation summarize
```

Lệnh đầu tính độ chính xác và Cohen's kappa giữa judge và người, ghi vào `results/judge_agreement.json`. Báo cáo D2 lấy số liệu từ file này.
