# LLM-automation

將 Youtube 上的影片下載，並自動化產生資料集到訓練專屬模型的的自動化流程

## 如何開始？

### 下載這個 repo 並更新 submodule

```bash
git clone https://github.com/brick2023/LLM-automation.git
cd LLM-automation
git submodule update --init --recursive
```

### 開始使用
```bash
python main.py
```

你可以根據需求修改 `main.py` 中的參數 (Youtube 播放清單連結、主題名稱)，以達到你的需求，整個流程有六個步驟

1. 下載影片
2. 轉換影片成純文字檔案以及字幕檔案
3. 生成摘要檔案 (使用長文本摘要演算法)
4. 生成資料集
5. 訓練模型
6. 加入資料庫 (optional，請根據你的需求來加入資料庫，不需要請註解掉)

## 環境需求

請參考 submodule 中的 `requirements.txt`，根據你的需求來下載套件

子模組的功能如下：

- `modelTool`：提供了一個簡單的介面，讓你可以快速的下載影片、影片轉文字、內容摘要
- `generate_QA`：利用 OpenAI API 來產生資料集
- `alpaca-lora`：使用 lora 方法來訓練專屬模型

## 流程圖

長文本摘要可以參考 modelTool 中的說明

![Imgur](https://i.imgur.com/w247Cxb.png)

整個過程只有 GPT 搜集資料集會需要花錢，其他都是免費的。因為我們希望看有優質的資料集，所以我們選擇了 OpenAI GPT 來產生資料，如果你想要用本地的模型 (像是 Vicuna)，可以使用 `modelTool/summarize` 中的 `introduction` 函式，並參照 `generate_QA` 來修改對應的 prompt

## TODO

- [ ] 模型效能評估以及調整
- [ ] 資料集產生方式改進 (根據影片長度改進，或者用統計方法去除離群值)