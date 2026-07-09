# iSE Challenge 2026 Multimodal QA Baseline

Baseline này nhận data lake trong `input/raw`, đọc câu hỏi từ `input/sample_questions.xlsx`, build index đa phương thức, retrieve evidence, gọi OpenRouter để sinh đáp án, rồi xuất `output/submission.csv` đúng format:

```csv
id,answer,evidences
1,2026,"[""file_1.csv""]"
```

## Recommended Workflow

Không nên upload cả project code lên Drive mỗi lần sửa. Luồng gọn hơn:

1. Code nằm trên GitHub.
2. Data lake nằm trên Google Drive, ví dụ `MyDrive/MMDLQA_data/input/raw`.
3. Colab clone/pull code từ GitHub vào `/content/MMDLQA`.
4. Colab đọc input và ghi output qua biến môi trường trỏ về Drive.
5. Khi sửa code ở máy local: `git add`, `git commit`, `git push`; qua Colab chạy lại cell clone/pull.

Drive chỉ cần chứa:

```text
MMDLQA_data/
  input/
    sample_questions.xlsx
    raw/
      ...
  output/
```

Code repo không cần nằm trong Drive.

## Colab Setup

Nếu bạn mới dùng Colab, cách dễ nhất là mở notebook [colab_quickstart.ipynb](colab_quickstart.ipynb), sửa `REPO_URL` và `DRIVE_DATA_DIR`, rồi chạy lần lượt từng cell.

```bash
!apt-get update -y
!apt-get install -y ffmpeg tesseract-ocr tesseract-ocr-vie tesseract-ocr-chi-sim libreoffice
!pip install -r requirements.txt
```

Mount Drive hoặc upload data lake vào:

```text
input/raw/
input/sample_questions.xlsx
```

Set OpenRouter key:

```python
import os
os.environ["OPENROUTER_API_KEY"] = "YOUR_KEY"
os.environ["OPENROUTER_MODEL"] = "gemma-4-26b-a4b-it"
```

Nếu OpenRouter yêu cầu model slug có prefix provider, chỉ cần đổi `OPENROUTER_MODEL`, không cần sửa code.

## Run

Dry run không gọi LLM, hữu ích để kiểm tra ingest/retrieve:

```bash
MMDLQA_USE_LLM=0 python script/run_pipeline.py --rebuild-index --limit 5
```

Chạy đầy đủ:

```bash
python script/run_pipeline.py --rebuild-index
```

Chạy workflow agentic shell mới, hiện vẫn dùng reasoner baseline ở phía sau nhưng RAG boundary đã là một `sentence`:

```bash
MMDLQA_USE_LLM=0 python script/run_agentic.py --limit 5
```

Đánh giá nhanh các câu `exact_match` trong sample có groundtruth:

```bash
python script/evaluate_sample.py
```

Kết quả chính:

```text
output/submission.csv
```

File debug:

```text
output/diagnostics.jsonl
output/run_summary.json
output/cache/files.jsonl
output/cache/chunks.jsonl
```

## Pipeline

1. Walk `input/raw` và phân loại file: table, text, document, image, audio, video.
2. Extract nội dung:
   - CSV/XLSX: cột, sheet, preview rows.
   - TXT/MD/HTML/SQL/code: text sạch.
   - PDF/DOCX/PPTX: text theo page/slide khi thư viện hỗ trợ.
   - Image: kích thước, OCR bằng Tesseract, color stats, optional vision caption bằng OpenRouter khi bật `MMDLQA_USE_LLM_SUMMARIES=1`.
   - Audio: duration + transcript bằng Whisper nếu bật.
   - Video: audio transcript + sample frames + OCR/caption.
3. Chunk và cache vào JSONL.
4. Retrieve bằng BM25 nhẹ + path/folder mention + modality hints.
5. Optional LLM rerank top chunks.
6. Answer JSON bằng OpenRouter, ép evidence chỉ lấy từ file đã retrieve.
7. Normalize exact-match answer và xuất CSV.

## Agentic Refactor

Code đã được tách theo ranh giới phát triển thay vì gom trong một package baseline:

- `mmdlqa_core/`: schema, config, question loader, OpenRouter client, utility và contract chung như `RagQuery`, `ReasoningStep`, `AgentState`.
- `mmdlqa_preprocess/`: xử lý offline data lake, extract file, chunk, và ghi knowledge library/cache.
- `mmdlqa_retrieval/`: search/RAG; `SentenceRAG` nhận một câu truy vấn tự nhiên rồi trả về chunks liên quan.
- `mmdlqa_agents/`: multi-agent workflow `Planner -> RAG -> Tool/Coder -> MoE Reasoners -> Aggregator -> Critic`.
  - `planner.py`: tách câu hỏi thành các `ReasoningStep`; mỗi step là một sentence có thể đưa thẳng vào RAG.
  - `reasoners.py`: chạy deterministic tool trước, rồi các LLM experts như exact-answer và synthesis khi bật MoE.
  - `critic.py`: kiểm tra answer/evidence, có thể yêu cầu retrieve bổ sung bằng `missing_queries`.
  - `workflow.py`: điều phối loop nhiều round và ghi diagnostics chi tiết.
- `mmdlqa_orchestration/`: runner pipeline nối preprocess, retrieval và agent để xuất submission.
- `script/run_agentic.py`: entrypoint chạy song song với baseline runner.

## Cost Controls

- `MMDLQA_USE_LLM=0`: không gọi OpenRouter.
- `MMDLQA_USE_LLM_RERANK=0`: bỏ bước rerank.
- `MMDLQA_USE_VISION_LLM=0`: không gọi vision LLM cho câu hỏi ảnh.
- `MMDLQA_USE_LLM_SUMMARIES=1`: caption mọi ảnh lúc build index, tốn quota hơn nhưng retrieve tốt hơn.
- `MMDLQA_USE_WHISPER=0`: không transcribe audio/video.
- `MMDLQA_USE_AGENTIC_PLANNER=0`: dùng rule-based planner thay vì LLM planner.
- `MMDLQA_USE_AGENTIC_MOE=0`: chỉ dùng deterministic tools + fallback, không gọi MoE reasoners.
- `MMDLQA_USE_AGENTIC_CRITIC=0`: chỉ dùng static critic, không gọi LLM critic.
- `MMDLQA_AGENTIC_MAX_STEPS=5`, `MMDLQA_AGENTIC_MAX_ROUNDS=2`: giới hạn planning/retry.
- `MMDLQA_AGENTIC_MOE_MODELS=model_a,model_b`: optional, dùng model riêng cho các expert.
- `MMDLQA_RETRIEVE_TOP_K=8`, `MMDLQA_RERANK_TOP_K=5`, `MMDLQA_MAX_CONTEXT_CHARS=16000`: giảm token.

## Notes

- `sample_questions.xlsx` hiện có cột `STT`, `Question`, `Groundtruth`, `Data Sources`, `Answer Type`; pipeline chỉ dùng `STT`, `Question`, `Answer Type` khi suy luận.
- `Groundtruth` và `Data Sources` chỉ được đọc để debug/dev, không dùng làm evidence trong submission.
- Các câu định lượng đơn giản có hook deterministic, ví dụ Pearson correlation trên CSV/XLSX nếu `pandas` có sẵn.
