# 模型來源與授權說明

本專案使用 m-a-p 團隊的 [MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M)，
對應 Li 等人的 [MERT 論文](https://arxiv.org/abs/2306.00107)。
固定模型版本為 `12af15fef9d0ac838c3f475bfbbf26d2060dd4f5`；
[該版本模型卡](https://huggingface.co/m-a-p/MERT-v1-95M/blob/12af15fef9d0ac838c3f475bfbbf26d2060dd4f5/README.md)
標示權重授權為 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)。
請保留來源、修改說明與授權連結，並遵守其非商業使用條款。

Release 中的 `dataset_A_finetuned.pt` 與 `dataset_B_finetuned.pt` 是修改後的
MERT 衍生權重：各自更新第 8 至 11 個 Transformer 區塊（從 0 起算），
並加入年代或市場分類頭。它們不是上游原始模型，也不表示上游團隊認可本專案。
凍結的其餘參數會在推論時從固定上游版本重建；本 repo 不重新散布完整預訓練快取。

凍結特徵方法的 `.joblib` 檔包含本次訓練的前處理與分類器。
音訊語言模型比較使用 Qwen2.5-Omni-3B、ggml-org 的 GGUF 轉換與 llama.cpp；
這些外部模型和程式保留各自授權，本 repo 僅提供評估程式、設定和結果。
完整來源、版本與引用列於中文報告及 README。

課程釋出的音訊、預訓練模型快取、帳號憑證及訓練續跑用的 optimizer 狀態均未發布。
