# AI Financial Project Guidelines & Rules

## 1. Core Workflow: Spec-First / Documentation-Driven Development (BẮT BUỘC)
- **Quy tắc tối thượng**: **PHẢI VIẾT DOCS THIẾT KẾ TRƯỚC KHI CODE**.
- Mọi tính năng, pipeline ETL, schema database, hay API mới đều phải có tài liệu đặc tả kiến trúc tại `docs/` trước khi tiến hành code.
- Tài liệu đóng vai trò là "Bản thiết kế mục tiêu" (Target Ground Truth).
- Sau khi code/rebuild database xong, luôn dùng subagent `system-evaluator` (skill `system-evaluator`) để đối chiếu chéo giữa Docs vs Database thật vs Code vs Benchmark.
- Nếu phát hiện tri thức thực tế mới trong quá trình làm, phải cập nhật đồng bộ ngược lại vào Docs ngay lập tức, không để Docs bị trôi (Drift).

## 2. Engineering Standards (Ponytail & Zero-External-Deps)
- Tối giản hóa tối đa mã nguồn (Standard Library first).
- CSDL: SQLite Native (WAL mode, B-Tree Indexes, FTS5 Virtual Table).
- Đảm bảo 100% tính truy vết kiểm toán (Provenance & Traceability): mọi số liệu trích xuất phải gắn liền với `(source_doc, page_no, section_title, raw_value)`.
