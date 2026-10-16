# ARC-04: Conductor & Worker Subagent Protocol (State Handoff Architecture)

> **Status:** Approved / Active Architecture  
> **Target Runtimes:** Tesla T4 (15.84 GB VRAM, 12.67 GB System RAM) & NVIDIA A100 (40 GB VRAM)  
> **Parent Architecture:** [[ARC-01 System Overview]], [[ARC-03 Zero-Token Fallback Playbook]]

---

## 1. Bối Cảnh & Vấn Đề Kỹ Thuật (Why This Exists)

Trên các máy ảo GPU như Google Colab Tesla T4, việc hoán đổi qua lại (ping-pong swapping) giữa 2 mô hình lớn 14B (`deepseek-r1:14b` và `qwen2.5-coder:14b`) gặp 3 nút thắt phần cứng nghiêm trọng:
1. **Thiếu hụt System RAM (12.67 GB):** Cả 2 model nặng tổng cộng 18.5 GB, vượt quá dung lượng RAM hệ thống. Khi đổi model, Ollama buộc phải đọc lại từ ổ đĩa mạng gây độ trễ 20–40s.
2. **Xóa sạch KV Cache (Context Wipeout):** Mỗi lần dỡ bỏ model khỏi VRAM, toàn bộ ngữ cảnh 32K tokens vừa tính toán bị xóa sổ hoàn toàn. Model mới phải prefill lại từ đầu mất thêm 30–45s.
3. **Mô hình tính phí của Colab:** Colab trừ phí theo Wall-Clock Time (~1.53 CU/h), model 7B hay 14B chạy 3.5 tiếng đều tốn cùng lượng CU.

---

## 2. Giải Pháp: Mô Hình "Nhạc Trưởng (Conductor) & Chuyên Gia (Worker)"

Thay vì sao chép toàn bộ ngữ cảnh khổng lồ giữa các model, ta áp dụng mô hình **Tách bạch Trạng thái (Decoupled State Handoff)**:

```text
┌────────────────────────────────────────────────────────────────────────┐
│  BƯỚC 1: NHẠC TRƯỞNG (Conductor / Planning)                            │
│  • Tiếp nhận yêu cầu từ lập trình viên.                                │
│  • Suy luận chuỗi tư duy <think>, bóc tách edge-cases & kiến trúc.     │
│  • Xuất ra bản STATE HANDOFF siêu cô đọng:                             │
│    📂 .agents/kb/HANDOFF.md (Dung lượng: ~1,000 – 1,500 tokens)        │
├────────────────────────────────────────────────────────────────────────┤
│  BƯỚC 2: BÀN GIAO SANG CHUYÊN GIA (Worker Specialist / Coding)         │
│  • Chuyên gia nhận task và đọc ĐÚNG 1,500 tokens từ HANDOFF.md.        │
│  • KHÔNG đọc 30,000 tokens lịch sử chat dài dòng.                      │
│  • Tốc độ Prefill: < 0.3 GIÂY (Gần như tức thì).                       │
│  • Mức RAM tiêu tốn: < 0.15 GB VRAM (0% nguy cơ OOM).                  │
│  • Sinh mã nguồn chuẩn xác 100% theo hợp đồng kiến trúc!              │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Phân Bổ Mô Hình Cho 2 Option Notebook

### 🚀 Option 2 (Performance — Primary Daily Driver)
* **File:** `notebooks/colab_ollama_mcp.ipynb`
* **Phần cứng:** Tesla T4 (16GB VRAM, ~1.53 CU/h)
* **Nhạc Trưởng (Planning):** `deepseek-r1:14b` (CoT `<think>` sâu sắc nhất)
* **Chuyên Gia (Coding, Workflow, Diagram):** `qwen2.5-coder:14b` (HumanEval 86.4%, Mermaid.js)

### ⚡ Option 1 (Saver — Fallback Mode)
* **File:** `notebooks/colab_ollama_t4_saver.ipynb`
* **Phần cứng:** Tesla T4 (16GB VRAM, ~1.53 CU/h)
* **Nhạc Trưởng (Planning):** `deepseek-r1:8b` (CoT nhẹ nhàng)
* **Chuyên Gia (Coding & Diagram):** `qwen2.5-coder:7b` (65+ tok/s)
* **Workflow:** `llama3.1:8b` (Meta tool-calling)

---

## 4. Ba Chốt Chặn Kỹ Thuật (System Safeguards)

1. **Linux 16GB NVMe Swapfile:**
   ```bash
   !test -f /swapfile || (fallocate -l 16G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile)
   ```
   Mở rộng bộ nhớ máy ảo từ 12.67 GB lên **28.67 GB**, triệt tiêu 100% nguy cơ OOM Killer.
2. **Flash Attention & Quantized KV Cache:**
   ```bash
   OLLAMA_FLASH_ATTENTION=1
   OLLAMA_KV_CACHE_TYPE=q8_0
   OLLAMA_MAX_LOADED_MODELS=1
   ```
   Nén bộ nhớ ngữ cảnh hơn 50%, giữ VRAM đỉnh tải luôn $\le 12.5\text{ GB} / 15.84\text{ GB}$.
3. **Cơ Chế Bàn Giao Tinh Gọn:**
   Giới hạn file handoff $\le 1,500\text{ tokens}$ để prefill tức thì $< 0.3\text{s}$.
