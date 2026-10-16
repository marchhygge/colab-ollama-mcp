# State Handoff Template (Conductor -> Worker)
> **Target Size:** < 1,500 tokens | **Prefill Latency:** < 0.3s | **VRAM Overhead:** < 0.15 GB

---

## 1. Task Objective
- **Goal:** [Tóm tắt 1-2 câu về mục tiêu cụ thể cần đạt được]
- **Target File(s):**
  - `path/to/file1.py` (Mô tả thay đổi)
  - `path/to/file2.py` (Mô tả thay đổi)

## 2. Technical Contracts & Specifications
- **Input Parameters:** [Định nghĩa kiểu dữ liệu & tham số]
- **Output Return:** [Định nghĩa giá trị trả về mong đợi]
- **Constraints / Edge-Cases:**
  - [Ràng buộc 1: ví dụ không dùng thư viện ngoài, dùng pure stdlib]
  - [Ràng buộc 2: xử lý exception timeout / connection error]

## 3. Step-by-Step Execution Checklist
- [ ] **Step 1:** [Viết hoặc cập nhật hàm X trong file Y]
- [ ] **Step 2:** [Thêm unit test kiểm thử logic trong tests/]
- [ ] **Step 3:** [Xác thực cú pháp và kiểu dữ liệu với py_compile]

## 4. Expected Code Signature / Skeleton (Optional)
```python
def example_function(param1: str, param2: int) -> bool:
    """Docstring explaining the exact contract."""
    pass
```
