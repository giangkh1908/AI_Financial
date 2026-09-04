---
name: doc-driven-development
description: Enforces Spec-First / Documentation-Driven Development before any implementation.
trigger: always_on
---

# BẮT BUỘC: QUY TRÌNH DOC-DRIVEN DEVELOPMENT (SPEC-FIRST WORKFLOW)

Mọi tác vụ kỹ thuật trong dự án này (viết tính năng mới, tạo bảng CSDL mới, xây pipeline ETL, tạo API endpoint, thiết lập mô hình AI) ĐỀU PHẢI TUÂN THỦ NGHIÊM NGẶT QUY TRÌNH 4 BƯỚC:

```
┌─────────────────────────┐
│ Bước 1: SPEC-FIRST      │ -> Viết hoặc cập nhật Docs thiết kế trước khi code
│ (Tài liệu là Chân lý)  │    (Kiến trúc, DDL Schema, API Contract, Tiêu chí nghiệm thu)
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Bước 2: IMPLEMENTATION  │ -> Triển khai code, pipeline, database đúng 100%
│ (Thực thi theo bản vẽ)  │    theo tài liệu đặc tả ở Bước 1
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Bước 3: CROSS-AUDIT     │ -> Dùng system-evaluator đối chiếu chéo 4 đỉnh:
│ (Thẩm định chéo)        │    Docs vs CSDL thật vs Code logic vs Benchmark
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Bước 4: RECONCILIATION  │ -> Sửa code nếu lệch docs. Cập nhật lại docs nếu
│ (Đồng bộ tuyệt đối)     │    phát hiện tri thức/dữ liệu thực tế mới.
└─────────────────────────┘
```

---

## CÁC NGUYÊN TẮC BẤT BIẾN (MANDATORY INVARIANTS)

1. **NO CODE WITHOUT SPEC (Tuyệt đối không code trước khi có Docs)**:
   - Nghiêm cấm việc nhảy vào viết code hoặc tạo database khi chưa có tài liệu đặc tả trong `docs/` được người dùng duyệt.
   - Docs đóng vai trò là "Bản vẽ kỹ thuật" và "Hợp đồng kiểm thử" (Target Ground Truth).

2. **DOCS LÀ THƯỚC ĐO ĐỂ ĐỐI CHIẾU (Ground Truth Baseline)**:
   - Khi hoàn thành code hoặc khi chạy CSDL, tài liệu ở Bước 1 chính là cơ sở để Agent thẩm định (`system-evaluator`) soi chiếu từng trường hợp:
     * Cột trong bảng CSDL có đúng tên, kiểu, nullability như docs đã viết?
     * Số lượng bản ghi, tỷ lệ dữ liệu thực tế có khớp bảng thống kê trong docs?
     * Mẫu truy vấn có chạy trơn tru trên dữ liệu thật mà không dính ngoại lệ?

3. **CONTINUOUS RECONCILIATION (Luôn cập nhật docs khi có tri thức thực tế mới)**:
   - Trong quá trình implement, nếu phát hiện dữ liệu thực tế có phát sinh mới (ví dụ định dạng ngày tháng mới, trường hợp ngoại lệ, mã mẫu biểu bổ sung), PHẢI cập nhật ngược lại tài liệu ngay lập tức.
   - Tuyệt đối không để xảy ra hiện tượng "Code đi đường code, Docs đi đường docs" (Documentation Drift).
