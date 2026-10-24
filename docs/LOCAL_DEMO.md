# Chạy demo Critic-Agent Explorer trên máy (local)

## Lần đầu cài đặt (khoảng 10 phút)

Cần: macOS hoặc Linux, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com).

```bash
git clone https://github.com/anho272202-HCMUT/C05151_Next-generation_Software_Technology.git
cd C05151_Next-generation_Software_Technology
cp .env.example .env
uv sync --frozen                      # cài đúng các phiên bản đã pin
./scripts/download_papers.sh          # tải 8 bài báo từ arXiv vào data/papers/
ollama pull qwen2.5:3b                # model LLM local (~1.9 GB)
ollama pull nomic-embed-text          # model embedding (~0.3 GB)
uv run python -m critic_agent.graph_builder   # dựng đồ thị tri thức
```

## Mỗi lần demo

```bash
ollama serve &                        # nếu Ollama chưa chạy
uv run critic-agent serve             # mở http://localhost:8000
```

| Tab | Cách dùng khi trình bày |
|---|---|
| Tổng quan | Giới thiệu bài toán và 5 trục phản biện |
| Kiến trúc | Bấm từng khối để giải thích vai trò, tool, mức quyền |
| **Chạy demo** | Chọn một lượt đã ghi rồi bấm **▶ Phát** (có thể lùi / tiến từng bước), hoặc chọn bài báo rồi bấm **▶ Chạy LIVE** để hệ thống phân tích thật trên máy (khoảng 2–3 phút/bài với qwen2.5:3b) |
| Đồ thị tri thức | Bấm vào Agentless để thấy cạnh CONTRADICTS và hồ sơ nguồn trích dẫn |
| Kết quả đánh giá | Số liệu Full so với baseline, sau khi chạy đánh giá |
| An ninh | Dán một đoạn văn có chỉ thị độc hại vào ô thử guardrail |

**Phương án dự phòng:** nếu chạy live bị lỗi (Ollama chưa chạy, máy chậm), chuyển sang phát lại lượt đã ghi. Lượt ghi sẵn
không cần mạng và không cần model. Chỉ cần mở trang tĩnh: `python3 -m http.server 8765 --directory web` rồi vào http://localhost:8765.

## Cập nhật dữ liệu hiển thị trên web

Sau khi chạy thêm review hoặc đánh giá:

```bash
uv run python scripts/export_web_data.py   # chép trace mới, đồ thị, kết quả vào web/data/
```

## Lưu ý trên macOS

Nếu thư mục nằm trên Desktop có đồng bộ iCloud, giữ `.venv`, `data/papers` và `runs` trong các thư mục đuôi `.nosync`
(xem README) để iCloud không đẩy file lên cloud giữa lúc chạy. Nếu `uv run` báo `No module named critic_agent`, chạy:
`chflags nohidden .venv/lib/python3.12/site-packages/*.pth`.
