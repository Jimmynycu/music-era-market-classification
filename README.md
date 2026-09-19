# 音樂年代與發行市場分類

使用課程提供的30秒音訊，分別預測美國發行年代（Dataset A）與1980年代唱片的發行市場（Dataset B）。所有特徵擷取、訓練與推論都在本機CPU／GPU完成；沒有使用雲端運算或額外標註音訊。

[繁體中文實驗報告（10頁）](reports/report_zh-TW.pdf) · [公開程式庫](https://github.com/Jimmynycu/music-era-market-classification) · [模型下載](https://github.com/Jimmynycu/music-era-market-classification/releases/tag/v1.0.0)

## 結果：先看表現較好的凍結特徵方法

下表是**官方驗證集**成績，A共132筆、B共102筆。測試集標籤未公開，因此不能把這些數字稱為測試準確率。Top-3表示正確答案出現在前三名，並非單一答案的正確率。

| 方法 | A：Top-1 | A：Top-3 | B：Top-1 | B：Top-3 |
| --- | ---: | ---: | ---: | ---: |
| **凍結MERT特徵＋Ridge分類器** | **40.15%** | **83.33%** | **41.18%** | **75.49%** |
| MERT最後四層真正微調 | 39.39% | 79.55% | 26.47% | 53.92% |
| Qwen2.5-Omni音訊語言模型 | 21.21% | 62.88% | 31.37% | 61.76% |

凍結方法在本次兩個任務的Top-1與Top-3均優於微調方法，因此報告先介紹它。**這項呈現順序沒有變更推論程式：`inference.py`預設仍是`finetuned`。要使用凍結方法，必須明確加上`--method frozen`。**

| 版本 | 已保存的測試預測 | 權重位置 |
| --- | --- | --- |
| 凍結特徵分類器 | [results/predictions.json](results/predictions.json) | `checkpoints/dataset_A.joblib`、`dataset_B.joblib`，已包含於Git |
| 微調模型 | [results/finetune_predictions.json](results/finetune_predictions.json) | `dataset_A_finetuned.pt`、`dataset_B_finetuned.pt`，由GitHub Release下載至`checkpoints/` |

兩份JSON各自包含全部234筆測試樣本，每筆都是三個不同標籤，依模型分數排序。微調版本仍保留為先前指定的提交版本；本次公開整理沒有覆寫任一版本。完整數值與混淆矩陣見[凍結結果](results/metrics.json)、[微調結果](results/finetune_metrics.json)及[音訊語言模型結果](results/alm_metrics.json)。

## 安裝與重現預測

已測試Python 3.12；本機硬體為Intel Core i7-4790K、16GB RAM、NVIDIA GTX 1060 6GB。

```bash
git clone https://github.com/Jimmynycu/music-era-market-classification.git
cd music-era-market-classification
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

資料集不隨倉庫提供，請自行取得課程發布的Dataset A與B。下列路徑各自指向含`manifest.csv`及`audio/`的資料集根目錄；也可以改用僅包含該資料集測試WAV的目錄。**不要把混有train／validation／test的`audio/`目錄當作測試目錄。**有manifest時，程式只讀取`split=test`的音訊，不使用標籤。

### 1. 重現凍結特徵方法

分類器已在Git中，不需要下載微調權重：

```bash
python inference.py --method frozen --dataset-a /your/path/data/dataset_A --dataset-b /your/path/data/dataset_B --output predictions_frozen.json
python verify_submission.py predictions_frozen.json --data-dir /your/path/data --reproduced results/predictions.json
```

### 2. 重現微調方法

兩個微調checkpoint各超過100MiB，存放於Release而非Git。可使用GitHub CLI下載：

```bash
gh release download v1.0.0 --repo Jimmynycu/music-era-market-classification --pattern '*_finetuned.pt' --dir checkpoints
sha256sum -c checkpoints/SHA256SUMS
```

也可用瀏覽器下載[Dataset A微調權重](https://github.com/Jimmynycu/music-era-market-classification/releases/download/v1.0.0/dataset_A_finetuned.pt)與[Dataset B微調權重](https://github.com/Jimmynycu/music-era-market-classification/releases/download/v1.0.0/dataset_B_finetuned.pt)，放進`checkpoints/`。

```bash
python inference.py --method finetuned --dataset-a /your/path/data/dataset_A --dataset-b /your/path/data/dataset_B --output predictions_finetuned.json
python verify_submission.py predictions_finetuned.json --data-dir /your/path/data --reproduced results/finetune_predictions.json
```

省略`--method`也會執行微調版本。只載入可信任的checkpoint。

兩種方法首次執行都會下載約380MB的公開基底模型[MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M/tree/12af15fef9d0ac838c3f475bfbbf26d2060dd4f5)，固定revision為`12af15fef9d0ac838c3f475bfbbf26d2060dd4f5`。微調checkpoint保存更新過的編碼器區塊與分類頭；其餘凍結權重由此基底重建，並比對雜湊。程式使用原模型自訂程式碼（`trust_remote_code=True`），只轉換舊版positional-convolution權重名稱，不改變其值，且拒絕缺失或多餘權重。

若需離線推論，先將該revision的`config.json`、`configuration_MERT.py`、`modeling_MERT.py`、`preprocessor_config.json`及`pytorch_model.bin`放入同一目錄，再加上`--model-dir /your/path/MERT-v1-95M`。資料集、基底模型、特徵cache與訓練暫存均不放進公開倉庫。

預設有CUDA就使用GPU，否則使用CPU；可指定`--device cpu`或`--device cuda`。本次GTX 1060使用保留Pascal支援的PyTorch 2.6.0／CUDA 12.4：

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

## 方法一：凍結MERT特徵＋分類器

MERT保持凍結，沒有微調。每首30秒音訊切成六段不重疊的5秒片段，各自做波形零均值與方差正規化；13個hidden-state層級分別合併所有片段的時間影格，計算768維平均與768維標準差。另從完整原始音訊計算244維聲學統計，包含log-mel、MFCC與差分、chroma、頻譜及能量等特徵。

- **Dataset A**：平均第1–4、5–8、9–12層的特徵後串接，再加聲學統計，共4852維；最後使用`RidgeClassifier(alpha=1)`。
- **Dataset B**：平均第5–8層特徵，共1536維；最後使用`RidgeClassifier(alpha=1)`。

每個任務預先設定40個特徵／Ridge／RBF-SVC候選，只在官方train內做分層三折交叉驗證，種子2026。選擇分數為`Top-1 + 0.5 × Top-3`；同分先比Top-1，再依固定候選順序。`StandardScaler`每折只在該折訓練資料擬合，接著逐首做L2正規化；選定後以全部官方train重新擬合，官方validation僅做最後評估。

若要重跑，先準備課程資料與上述固定版本的本機基底模型：

```bash
python features.py --data-dir /your/path/data --model-dir /your/path/MERT-v1-95M
python train.py --data-dir /your/path/data --cache-dir cache --checkpoint-dir checkpoints --results-dir results
```

`train.py`預設使用一個數值運算執行緒：本機Haswell CPU的SciPy／OpenBLAS單精度Cholesky在四執行緒時曾崩潰，最終擬合使用單執行緒完成。其他系統可自行設定`--threads`。

## 方法二：真正微調MERT最後四層

此追加實驗是在看過凍結方法的官方validation成績後提出，因此不能稱為完全未見的確認性評估。微調設定在執行前固定，梯度更新及epoch選擇只使用官方train；沒有根據微調的官方validation成績更換最終模型。

真正更新的是MERT第8–11個transformer block（從0起算），共**28,351,488個預訓練參數**，以及新分類頭。更早的區塊與前端保持凍結。最後一層的影格平均及`sqrt(variance + 1e-5)`串接為1536維，送入`LayerNorm → Dropout(0.1) → Linear(6)`。

使用FP32交叉熵與AdamW：編碼器學習率`1e-5`、分類頭`1e-3`、weight decay `0.01`、梯度裁切`1.0`、固定學習率、microbatch 4、梯度累積至有效batch 16。每個epoch每首隨機取一段5秒；推論則完整使用六段固定片段並平均logits。除隨機裁切外，不做人聲分離、變調或變速。

官方train內再分出20%分層holdout（種子2026；A為820／206筆，B為638／160筆），以同一加權分數從第1–6個epoch選擇，平手先比Top-1，再選較早epoch。A選到6個epoch，B選到3個epoch；最後都從原始預訓練權重與全新分類頭開始，用全部官方train重訓。

```bash
python finetune.py --self-check
python finetune.py --data-dir /your/path/data --model-dir /your/path/MERT-v1-95M --device cuda --micro-batch-size 4 --max-epochs 6
```

相同指令會從`tmp/finetune/`已完成的epoch續跑；設定或基底雜湊不同時拒絕不相容的續跑。`--dataset A`或`--dataset B`可只跑一組，預設兩組。`--self-check`使用小型CPU模型，不會下載或訓練MERT。若只想重現預測，不必重訓。

[微調紀錄](results/finetune_metrics.json)保存切分ID、epoch歷程、各區塊非零梯度及權重差異；[獨立檢查](results/finetune_integrity_check.json)確認四個預訓練區塊都有改變，凍結權重雜湊不變。這些是編碼器確實更新的證據，不只是分類頭訓練。

**限制不能省略**：微調A的2000s召回率為0／22；微調B對Brazil、Spain、Germany的召回率均為0。凍結與微調版本也使用不同的pooling及分類頭，且凍結A額外使用聲學特徵，所以本比較無法單獨證明「解凍編碼器」本身造成成績變化。

## 音訊語言模型對照

另外以凍結的Qwen2.5-Omni-3B，對A全部132筆、B全部102筆validation執行一次固定提示設計，沒有訓練LLM或把它混入上述分類器。採單一設計的原因是控制本機運算成本並降低輸出歧義；原始英文提示詞與理由保存在[alm_eval.py](alm_eval.py)及[結果metadata](results/alm_metrics.json)。

模型為Q4_K_M、音訊投影器為Q8_0，使用llama.cpp `b11049`（`efa28e950`）Vulkan後端。GBNF限制輸出為A–F六個代碼的排列，再依固定對照表轉回標籤。無效輸出會重試一次，仍失敗才使用預定順序並註記；本次234筆均有效、沒有fallback。分數是排序，不是校準機率。

此版本llama.cpp的音訊功能仍屬實驗性：完整30秒WAV會被重採樣為16kHz，內部補30秒尾端零值，再處理兩個3000-frame mel區塊，共1500個音訊token。沒有驗證它與原始Transformers實作的前處理及數值完全等價；實際輸出有偏向少數類別的現象。檢查與來源見[ALM稽核](results/alm_integrity_check.json)。

需要重跑時，下載固定版本的公開模型與官方Linux執行檔：

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
python3 alm_eval.py --data-dir /your/path/data --compute-device 'NVIDIA GeForce GTX 1060 6GB, Vulkan; 2 CPU threads; all model layers and audio projector GPU-offloaded'
```

逐筆原始輸出存於`results/alm_predictions.jsonl`，相同設定重跑會續接；不要在續跑時更換模型或提示詞。CPU可改用同一release的`llama-b11049-bin-ubuntu-x64.tar.gz`，並將`-ngl 99 --device Vulkan0`替換為`-ngl 0 --no-mmproj-offload`。執行時間不含下載、載入或中斷；同期本機工作與GPU時脈會影響速度。

## 驗證紀錄與重建報告

兩套方法都從原始測試WAV重新推論全部234筆，與各自保存的JSON完全一致：

- [凍結方法完整重現檢查](results/reproduction_check.json)
- [微調方法完整重現檢查](results/finetune_reproduction_check.json)
- [資料集SHA256與音訊格式檢查](results/dataset_audit.json)
- [執行環境](results/run_environment.json)

[CPU相容性檢查](results/cpu_portability_check.json)只檢查凍結方法的四個接近分類邊界樣本，不能視為兩套方法都已在CPU全量驗證。可執行`python test_submission.py`檢查格式解析、資料路徑與分類器序列化。

重建繁體中文10頁報告：

```bash
sudo apt-get install fonts-droid-fallback fonts-dejavu-core
pip install -r requirements-report.txt
python make_report.py --self-check
python make_report.py --repo-url https://github.com/Jimmynycu/music-era-market-classification --data-dir /your/path/data --output output/pdf/report_zh-TW.pdf
```

報告使用倉庫中的結果JSON與課程manifest，不需要音訊或模型cache。上述Ubuntu字型套件提供DroidSansFallbackFull及DejaVu Sans，PDF會嵌入中文字型；執行分類推論不需要這些字型。加入`--student-id YOUR_STUDENT_ID --cloud-url 'https://your-public-cloud-folder-url'`可補齊提交資訊；公開版可先顯示待補欄位。發布的PDF位於[reports/report_zh-TW.pdf](reports/report_zh-TW.pdf)，以此繁體中文版為準，先前英文草稿不是目前報告。

## 課程提交尚需補齊的資料

公開GitHub倉庫與Release提供程式及權重，但**不自動等同課程指定的公開cloud-drive folder**。正式繳交仍需真實學號、符合課程要求的公開雲端資料夾連結、報告首頁連結，以及`<studentID>_report.pdf`與`<studentID>.json`檔名。還須在NTU COOL上傳報告／預測，並依作業要求於`HW1_report`留言放置雲端連結。

正式上傳的JSON須與提供的推論方法及checkpoint相符。程式目前預設及先前指定的提交版本是微調，較高驗證成績的凍結JSON則另外保留；公開整理沒有切換或提交檔案。課程資料、模型cache、optimizer續跑暫存與無關checkpoint均不應上傳。

## 來源與授權

微調權重衍生自MERT，不能當作無限制商用模型；詳見[NOTICE.md](NOTICE.md)。公開預訓練資料是否與匿名課程錄音重疊無法確認。

- [MERT模型與自訂程式碼（CC BY-NC 4.0）](https://huggingface.co/m-a-p/MERT-v1-95M)、[MERT論文](https://arxiv.org/abs/2306.00107)、[研究程式](https://github.com/yizhilll/MERT)
- [PyTorch](https://pytorch.org/)、[Hugging Face Transformers](https://github.com/huggingface/transformers)、[scikit-learn](https://scikit-learn.org/)
- [librosa](https://librosa.org/)、[NumPy](https://numpy.org/)、[SciPy](https://scipy.org/)、[SoundFile](https://python-soundfile.readthedocs.io/)、[joblib](https://joblib.readthedocs.io/)
- [Qwen2.5-Omni-3B（Qwen Research License）](https://huggingface.co/Qwen/Qwen2.5-Omni-3B)、[技術報告](https://arxiv.org/abs/2503.20215)、[GGUF模型轉換](https://huggingface.co/ggml-org/Qwen2.5-Omni-3B-GGUF)
- [llama.cpp（MIT）](https://github.com/ggml-org/llama.cpp)、[Matplotlib](https://matplotlib.org/)、[ReportLab](https://www.reportlab.com/opensource/)
