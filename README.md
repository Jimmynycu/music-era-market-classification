# Homework 1：音樂年代與發行市場分類

劉昌昀，D15921B14。正式英文報告在 [reports/report_en.pdf](reports/report_en.pdf)。中文閱讀版僅保留於本機 `reports/report_zh-TW_reading.pdf`，不作為正式提交檔。

本次先用凍結的 MERT 特徵訓練分類器，再微調 MERT 最後四層，另外跑 Qwen2.5-Omni 作比較。全部在本機 i7-4790K、16GB RAM、GTX 1060 6GB 上完成，沒有加入額外標註音訊。

本次選用作業書面說明中的預訓練音訊特徵路徑，沒有另外訓練建議方法之一的短片段 CNN。

## 結果

以下是官方 validation 成績。A 有 132 筆，B 有 102 筆；test 沒有公開答案。

| 方法 | A Top-1 | A Top-3 | B Top-1 | B Top-3 |
| --- | ---: | ---: | ---: | ---: |
| 凍結 MERT＋分類器 | 40.15% | 83.33% | 41.18% | 75.49% |
| 微調 MERT 最後四層 | 39.39% | 79.55% | 26.47% | 53.92% |
| Qwen2.5-Omni | 21.21% | 62.88% | 31.37% | 61.76% |

這次凍結方法的成績較好。微調仍保留為提交版本，`inference.py` 預設使用 `finetuned`。`results/finetune_predictions.json` 是微調預測；`results/predictions.json` 是凍結方法預測，兩份各有 234 筆。

## 如何執行推論

已使用 Python 3.12 測試。先在本資料夾安裝套件：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

微調權重放在 `checkpoints/dataset_A_finetuned.pt` 和 `checkpoints/dataset_B_finetuned.pt`。完整上傳包已包含這兩個檔案；從 GitHub clone 時要另外下載：

```bash
gh release download v1.0.0 --repo Jimmynycu/music-era-market-classification --pattern '*_finetuned.pt' --dir checkpoints
sha256sum -c checkpoints/SHA256SUMS
```

也可從 [GitHub Release](https://github.com/Jimmynycu/music-era-market-classification/releases/tag/v1.0.0) 用瀏覽器下載。

將下面 A、B 路徑改成自己的資料集根目錄，各自需有 `manifest.csv` 和 `audio/`。程式只讀取 manifest 中的 test 音訊，不讀 test 標籤。也接受只有 test WAV 的資料夾，但不要直接傳入混有三種 split 的 `audio/`。

```bash
python inference.py --method finetuned --dataset-a /your/path/data/dataset_A --dataset-b /your/path/data/dataset_B --output D15921B14.json
python verify_submission.py D15921B14.json --data-dir /your/path/data --reproduced results/finetune_predictions.json
```

若要跑凍結方法：

```bash
python inference.py --method frozen --dataset-a /your/path/data/dataset_A --dataset-b /your/path/data/dataset_B --output predictions_frozen.json
python verify_submission.py predictions_frozen.json --data-dir /your/path/data --reproduced results/predictions.json
```

兩種方法第一次執行都會下載約 380MB 的 [MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M/tree/12af15fef9d0ac838c3f475bfbbf26d2060dd4f5)，固定 revision 為 `12af15fef9d0ac838c3f475bfbbf26d2060dd4f5`，並使用模型的自訂程式碼。微調 checkpoint 只存有修改過的四個區塊與分類頭，其餘部分從此基底載入。離線時可用 `--model-dir /your/path/MERT-v1-95M` 指定同版本的本機資料夾。

預設有 CUDA 就用 GPU，否則用 CPU；可指定 `--device cpu`。GTX 1060 使用的是 PyTorch 2.6.0／CUDA 12.4：

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

## 程式怎麼分工

| 檔案 | 用途 |
| --- | --- |
| `features.py` | 讀取音訊，擷取 MERT 與聲學特徵 |
| `train.py` | 在 train 內做三折 CV，訓練凍結特徵分類器 |
| `finetune_model.py` | 定義微調模型、分類頭及 checkpoint 載入方式 |
| `finetune.py` | 以 train 內部 holdout 選 epoch，再用全部 train 重訓 |
| `inference.py` | 對 test 音訊輸出前三個標籤 |
| `alm_eval.py` | 呼叫本機音訊語言模型，解析答案並計算 validation 結果 |
| `verify_submission.py` | 檢查 JSON 的樣本、標籤與重現結果 |

凍結方法把 30 秒音訊切成六段 5 秒，以平均與標準差池化 MERT 特徵，接 StandardScaler、L2 正規化和 Ridge。A 使用分層 MERT 加 244 維聲學特徵，B 使用 MERT 中間層；兩個 Ridge 的 alpha 都是 1。

微調更新第 8 至 11 個 transformer block 和分類頭，共更新 28,351,488 個預訓練參數。每輪每首隨機取 5 秒，推論平均六段的 logits。AdamW 學習率為主幹 `1e-5`、分類頭 `1e-3`，有效 batch 16。train 內保留 20% 選輪數後，從原始基底重新訓練；A 跑 6 輪，B 跑 3 輪。

重訓前，將固定版本 MERT 的 `config.json`、`configuration_MERT.py`、`modeling_MERT.py`、`preprocessor_config.json`、`pytorch_model.bin` 放入同一資料夾：

```bash
python features.py --data-dir /your/path/data --model-dir /your/path/MERT-v1-95M
python train.py --data-dir /your/path/data --cache-dir cache --checkpoint-dir checkpoints --results-dir results
python finetune.py --data-dir /your/path/data --model-dir /your/path/MERT-v1-95M --device cuda --micro-batch-size 4 --max-epochs 6
```

微調會從 `tmp/finetune/` 已完成的 epoch 續跑。凍結分類器使用單一 BLAS 執行緒，因本機四執行緒的 OpenBLAS 曾在擬合時崩潰。

## 音訊語言模型

使用 Qwen2.5-Omni-3B 的 Q4_K_M 模型、Q8_0 音訊投影器與 llama.cpp `b11049` Vulkan。A 的 132 筆和 B 的 102 筆 validation 都跑完，只用一種固定提示，讓模型排列六個類別，再取前三名。解析失敗會重試一次，再失敗才用預設排序；此次沒有發生。提示原文與解析在 `alm_eval.py`，分數與原始回答在 `results/alm_metrics.json`、`results/alm_predictions.jsonl`。

交付包內的 `results/alm_predictions.jsonl` 是已完成的歷史結果。若只要重新核算分數，不需啟動模型：

```bash
python3 alm_eval.py --score-only --data-dir /your/path/data --output-dir results
```

這個指令逐筆核對 manifest、原始回答、排名、完整樣本數與保存的 metrics，只輸出核算結果，不修改舊紀錄，也不宣稱已重新執行或驗證歷史模型身分。

若要真正重新執行 ALM，可使用以下固定版本：

```bash
mkdir -p alm_runtime
cd alm_runtime
curl -L --fail -o llama.tar.gz https://github.com/ggml-org/llama.cpp/releases/download/b11049/llama-b11049-bin-ubuntu-vulkan-x64.tar.gz
tar -xzf llama.tar.gz
curl -L --fail -o Qwen2.5-Omni-3B-Q4_K_M.gguf https://huggingface.co/ggml-org/Qwen2.5-Omni-3B-GGUF/resolve/75f1b73b657a50f5092502799457ccb4a4a1f9df/Qwen2.5-Omni-3B-Q4_K_M.gguf
curl -L --fail -o mmproj-Qwen2.5-Omni-3B-Q8_0.gguf https://huggingface.co/ggml-org/Qwen2.5-Omni-3B-GGUF/resolve/75f1b73b657a50f5092502799457ccb4a4a1f9df/mmproj-Qwen2.5-Omni-3B-Q8_0.gguf
./llama-b11049/llama-server -m Qwen2.5-Omni-3B-Q4_K_M.gguf --mmproj mmproj-Qwen2.5-Omni-3B-Q8_0.gguf -c 2048 -np 1 -t 2 -tb 2 -ngl 99 --device Vulkan0 --host 127.0.0.1 --port 8091 --alias qwen2.5-omni-3b --no-webui -b 256 -ub 256 -fa off --cache-ram 0
```

等server載入完成，在另一個終端回到倉庫根目錄執行：

```bash
python3 alm_eval.py --self-check
python3 alm_eval.py --data-dir /your/path/data --output-dir results/alm_fresh_run --compute-device 'NVIDIA GeForce GTX 1060 6GB, Vulkan; 2 CPU threads; all model layers and audio projector GPU-offloaded'
```

`--output-dir` 必須不存在或是空目錄；上例的新結果寫入 `results/alm_fresh_run/`，不會沿用交付包的歷史回答。程式刻意不提供續跑，避免混入無法核實模型／音訊來源的舊結果：中斷後請換一個新目錄從頭執行，已有任何檔案的目錄都會被拒絕。舊紀錄不回填模型身分證據。可執行 `python3 test_alm_eval.py` 檢查這些行為，無需模型或網路。

CPU可改用同一release的`llama-b11049-bin-ubuntu-x64.tar.gz`，並將`-ngl 99 --device Vulkan0`替換為`-ngl 0 --no-mmproj-offload`。執行時間不含下載、載入或中斷；同期本機工作與GPU時脈會影響速度。

## 檢查與限制

兩種分類方法都重新從全部 234 筆 test WAV 推論，結果與保存的 JSON 相同。可執行 `python test_submission.py` 和 `python finetune.py --self-check` 做小型 CPU 檢查，不需要重訓。

微調是在看過凍結方法的 validation 成績後才追加；輪數選擇仍只使用 train。兩種方法的 pooling、分類頭及 A 的聲學特徵不同，因此不能把分數差異全歸因於解凍。微調 B 對三個市場的召回率為零，報告保留這些結果。ALM 的 llama.cpp 音訊功能仍屬實驗性，尚未確認與原始 Transformers 的處理完全相同。

## 報告與繳交

```bash
sudo apt-get install fonts-droid-fallback fonts-dejavu-core
pip install -r requirements-report.txt
python make_report.py --student-id D15921B14 --student-name '劉昌昀 (LIU, CHANG-YUN)' --cloud-url 'https://drive.google.com/drive/folders/1mpd0TqWgGYK-VmeExzmrSfgONGZU-20i' --data-dir /your/path/data --output reports/report_en.pdf
```

正式繳交檔名為 `D15921B14_report.pdf`（英文報告副本）和 `D15921B14.json`。[課程雲端資料夾](https://drive.google.com/drive/folders/1mpd0TqWgGYK-VmeExzmrSfgONGZU-20i) 提供程式、checkpoint、requirements、README 與提交檔案，網址已放入報告首頁。將 PDF、JSON 上傳 NTU COOL，並依作業要求留言同一個資料夾網址。資料夾不包含課程音檔、預訓練模型快取、optimizer 暫存或中文閱讀版。

模型授權與修改說明見 [NOTICE.md](NOTICE.md)。使用的來源如下：

模型來源：[MERT 論文](https://arxiv.org/abs/2306.00107)、[MERT 模型](https://huggingface.co/m-a-p/MERT-v1-95M/tree/12af15fef9d0ac838c3f475bfbbf26d2060dd4f5)；[Qwen 論文](https://arxiv.org/abs/2503.20215)、[原始模型](https://huggingface.co/Qwen/Qwen2.5-Omni-3B)、[GGUF 模型](https://huggingface.co/ggml-org/Qwen2.5-Omni-3B-GGUF/tree/75f1b73b657a50f5092502799457ccb4a4a1f9df)；[llama.cpp](https://github.com/ggml-org/llama.cpp/tree/efa28e950)（[音訊說明](https://github.com/ggml-org/llama.cpp/discussions/13759)）。

程式工具：[PyTorch](https://github.com/pytorch/pytorch)、[Transformers](https://github.com/huggingface/transformers)、[librosa](https://github.com/librosa/librosa)、[scikit-learn](https://scikit-learn.org/stable/about.html)、[NumPy](https://numpy.org/citing-numpy/)、[SciPy](https://github.com/scipy/scipy)、[SoundFile](https://github.com/bastibe/python-soundfile)、[joblib](https://github.com/joblib/joblib)、[threadpoolctl](https://github.com/joblib/threadpoolctl)。

圖表與報告：[Matplotlib](https://matplotlib.org/stable/project/citing.html)、[ReportLab](https://www.reportlab.com/opensource/)。資料：[課程資料與說明](https://drive.google.com/drive/folders/1C8RymiLbr-EGmkxh2Ap5TIybnJYqNsb4)。
