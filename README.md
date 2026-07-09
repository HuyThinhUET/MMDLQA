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
    text_cleaning_output/
      by_file/
      text_cleaning_manifest.csv
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
input/text_cleaning_output/
input/sample_questions.xlsx
```

`input/text_cleaning_output` is now treated as the preferred preprocessed knowledge source.
The index loader reads `by_file/*/clean.txt` plus `metadata.json` first, then extracts raw files
from `input/raw` only when they are not already covered by the cleaned manifest. With the current
local data this gives 60 preprocessed files and 26 raw fallback files, mostly CSV/XLSX tables.

Set OpenRouter key:

```python
import os
os.environ["OPENROUTER_API_KEY"] = "YOUR_KEY"
os.environ["MMDLQA_USE_MODEL_ROUTER"] = "1"
os.environ["MMDLQA_AGENTIC_MAX_ROUNDS"] = "4"
os.environ["MMDLQA_MAX_QUESTION_COST_USD"] = "0.1"
```

Mặc định pipeline dùng strong model routing theo vai trò: planner dùng GPT-5.6 Terra, rerank/scan nhẹ dùng GPT-5.6 Luna,
exact/critic/document/vision dùng Claude Sonnet 5, synthesis dùng GPT-5.6 Sol Pro, coder/table scan dùng Grok 4.5.
Nếu muốn ép toàn bộ workflow về một model duy nhất, set `MMDLQA_USE_MODEL_ROUTER=0`
và đổi `OPENROUTER_MODEL`.

## Run

Dry run không gọi LLM, hữu ích để kiểm tra ingest/retrieve:

```bash
MMDLQA_USE_LLM=0 python script/run_pipeline.py --rebuild-index --limit 5 --text-cleaning-output input/text_cleaning_output
```

Chạy đầy đủ:

```bash
python script/run_pipeline.py --rebuild-index --text-cleaning-output input/text_cleaning_output
```

Chạy workflow agentic shell mới, hiện vẫn dùng reasoner baseline ở phía sau nhưng RAG boundary đã là một `sentence`:

```bash
MMDLQA_USE_LLM=0 python script/run_agentic.py --rebuild-index --limit 5 --text-cleaning-output input/text_cleaning_output
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

1. Load `input/text_cleaning_output/by_file/*/clean.txt` plus metadata as the preferred offline preprocess output.
2. Walk `input/raw` and extract only files not covered by the cleaned output, usually tables/audio/legacy docs.
3. Extract nội dung:
   - CSV/XLSX: cột, sheet, preview rows.
   - TXT/MD/HTML/SQL/code: text sạch.
   - PDF/DOCX/PPTX: text theo page/slide khi thư viện hỗ trợ.
   - Image: kích thước, OCR bằng Tesseract, color stats, optional vision caption bằng OpenRouter khi bật `MMDLQA_USE_LLM_SUMMARIES=1`.
   - Audio: duration + transcript bằng Whisper nếu bật.
   - Video: audio transcript + sample frames + OCR/caption.
4. Chunk và cache vào JSONL.
5. Retrieve bằng BM25 nhẹ + path/folder mention + fuzzy source match + modality hints + optional semantic search.
6. Optional LLM rerank top chunks.
7. Answer JSON bằng OpenRouter, ép evidence chỉ lấy từ file đã retrieve.
8. Normalize exact-match answer và xuất CSV.

## Agentic Refactor

Code đã được tách theo ranh giới phát triển thay vì gom trong một package baseline:

- `mmdlqa_core/`: schema, config, question loader, OpenRouter client, utility và contract chung như `RagQuery`, `ReasoningStep`, `AgentState`.
- `mmdlqa_preprocess/`: xử lý offline data lake, load cleaned preprocess output, extract raw fallback, chunk, và ghi knowledge library/cache.
- `mmdlqa_retrieval/`: search/RAG; `SentenceRAG` nhận một câu truy vấn tự nhiên rồi trả về chunks liên quan, có BM25/path/fuzzy/source/optional semantic scoring.
- `mmdlqa_agents/`: multi-agent workflow `Planner -> RAG -> Tool/Coder -> MoE Reasoners -> Aggregator -> Critic`.
  - `planner.py`: tách câu hỏi thành các `ReasoningStep`; mỗi step là một sentence có thể đưa thẳng vào RAG.
  - `tool_agents.py`: tách `CoderAgent` cho table/SQL/calculation plan + safe executor và `ToolAgent` cho vision/deterministic tools.
- `reasoners.py`: điều phối Coder/Tool trước, rồi các LLM experts như exact-answer và synthesis khi bật MoE.
- `evidence_scanner.py`: quét sâu các file/chunk đã retrieve, dùng model theo modality để trích xuất direct/partial evidence và dừng khi gặp đủ file không liên quan liên tiếp.
- `evidence.py`: tạo evidence ledger dạng claim -> file/chunk/quote cho từng candidate.
  - `structured.py`: validate JSON output của planner/rerank/reasoner/critic/coder và repair một lần nếu sai schema.
  - `critic.py`: kiểm tra answer/evidence, có thể yêu cầu retrieve bổ sung bằng `missing_queries`.
  - `workflow.py`: điều phối loop nhiều round và ghi diagnostics chi tiết.
- `mmdlqa_orchestration/`: runner pipeline nối preprocess, retrieval và agent để xuất submission.
- `script/run_agentic.py`: entrypoint chạy song song với baseline runner.

## Cost Controls

- `MMDLQA_USE_LLM=0`: không gọi OpenRouter.
- `MMDLQA_TEXT_CLEANING_OUTPUT_DIR=input/text_cleaning_output`: folder output preprocess offline.
- `MMDLQA_USE_TEXT_CLEANING_OUTPUT=1`: index cleaned output first.
- `MMDLQA_INCLUDE_RAW_FALLBACK=1`: also extract raw files not covered by cleaned output.
- `MMDLQA_USE_LLM_RERANK=0`: bỏ bước rerank.
- `MMDLQA_USE_VISION_LLM=0`: không gọi vision LLM cho câu hỏi ảnh.
- `MMDLQA_USE_LLM_SUMMARIES=1`: caption mọi ảnh lúc build index, tốn quota hơn nhưng retrieve tốt hơn.
- `MMDLQA_USE_WHISPER=0`: không transcribe audio/video.
- `MMDLQA_USE_AGENTIC_PLANNER=0`: dùng rule-based planner thay vì LLM planner.
- `MMDLQA_USE_AGENTIC_MOE=0`: chỉ dùng deterministic tools + fallback, không gọi MoE reasoners.
- `MMDLQA_USE_AGENTIC_CRITIC=0`: chỉ dùng static critic, không gọi LLM critic.
- `MMDLQA_USE_AGENTIC_TOOLS=1`: bật ToolAgent cho vision/deterministic tools.
- `MMDLQA_USE_AGENTIC_CODER=1`: bật CoderAgent cho table/SQL/calculation tasks.
- `MMDLQA_USE_EVIDENCE_SCANNER=1`: bật LLM evidence scanner để đánh giá file direct/partial/irrelevant trước khi reason.
- `MMDLQA_EVIDENCE_SCAN_MAX_FILES=24`, `MMDLQA_EVIDENCE_SCAN_IRRELEVANT_PATIENCE=6`: scan sâu theo file và dừng khi gặp 6 file không liên quan liên tiếp.
- `MMDLQA_RERANK_CANDIDATE_K=36`, `MMDLQA_RERANK_TOP_K=12`: cho reranker nhìn nhiều candidate hơn sau scanner.
- `MMDLQA_USE_CODER_PLANNER=0`: bật LLM coder planner khi set `1`; mặc định tắt để không tăng cost.
- `MMDLQA_USE_MODEL_ROUTER=1`: bật chọn model theo vai trò thay vì một model chung.
- `MMDLQA_PLANNER_MODEL=openai/gpt-5.6-terra`: model tách câu hỏi thành steps.
- `MMDLQA_RERANK_MODEL=openai/gpt-5.6-luna`: model rerank chunks.
- `MMDLQA_EXACT_MODEL=anthropic/claude-sonnet-5`: expert exact-match mạnh hơn.
- `MMDLQA_SYNTHESIS_MODEL=openai/gpt-5.6-sol-pro`: expert tổng hợp multi-hop mạnh nhất trong profile.
- `MMDLQA_CRITIC_MODEL=anthropic/claude-sonnet-5`: agent phản biện evidence.
- `MMDLQA_CODER_MODEL=x-ai/grok-4.5`: model dành cho coder/calculation agent khi mở rộng.
- `MMDLQA_VISION_MODEL=anthropic/claude-sonnet-5`: model nhìn ảnh/video frame.
- `MMDLQA_SCAN_TEXT_MODEL=openai/gpt-5.6-luna`, `MMDLQA_SCAN_TABLE_MODEL=x-ai/grok-4.5`, `MMDLQA_SCAN_DOCUMENT_MODEL=anthropic/claude-sonnet-5`, `MMDLQA_SCAN_IMAGE_MODEL=anthropic/claude-sonnet-5`: model trích xuất evidence theo modality.
- `MMDLQA_AGENTIC_MAX_STEPS=5`, `MMDLQA_AGENTIC_MAX_ROUNDS=4`: giới hạn planning/retry, mặc định loop khoảng 3-5 vòng.
- `MMDLQA_AGENTIC_MOE_MODELS=model_a,model_b`: optional, override riêng exact/synthesis expert.
- `MMDLQA_MAX_QUESTION_SECONDS=0`: giới hạn thời gian mỗi câu; `0` là không giới hạn.
- `MMDLQA_MAX_QUESTION_LLM_CALLS=0`: giới hạn số LLM calls mỗi câu; `0` là không giới hạn.
- `MMDLQA_MAX_QUESTION_COST_USD=0.1`: giới hạn cost estimate mỗi câu; `0` là không giới hạn.
- `MMDLQA_MAX_QUESTION_RAG_QUERIES=0`: giới hạn số RAG query mỗi câu; `0` là không giới hạn.
- `MMDLQA_LLM_INPUT_COST_PER_MILLION_TOKENS=0`, `MMDLQA_LLM_OUTPUT_COST_PER_MILLION_TOKENS=0`: giá fallback nếu provider không trả cost và model không có trong bảng giá local.
- `MMDLQA_RETRIEVE_TOP_K=8`, `MMDLQA_RERANK_TOP_K=5`, `MMDLQA_MAX_CONTEXT_CHARS=16000`: giảm token.
- `MMDLQA_PRINT_QUESTION_METRICS=1`: in live progress từng câu gồm elapsed time, LLM calls, estimated cost và answer preview.

Mỗi câu hỏi ghi metrics vào `diagnostics.jsonl` trong `answer.diagnostics.metrics`: elapsed time, stage timings, LLM calls, token usage, estimated cost, và trạng thái limit. `run_summary.json` có phần tổng hợp `metrics` cho toàn run.

## Notes

- `sample_questions.xlsx` hiện có cột `STT`, `Question`, `Groundtruth`, `Data Sources`, `Answer Type`; pipeline chỉ dùng `STT`, `Question`, `Answer Type` khi suy luận.
- `Groundtruth` và `Data Sources` chỉ được đọc để debug/dev, không dùng làm evidence trong submission.
- Các câu định lượng đơn giản có hook deterministic, ví dụ Pearson correlation trên CSV/XLSX nếu `pandas` có sẵn.
