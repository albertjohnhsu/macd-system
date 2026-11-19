# 台股趨勢追蹤系統（twtrend）

依《台股趨勢追蹤系統設計書》實作。目標：吃到多頭的大部分漲幅、在空頭「確立初期」降低持股、排除地雷股。
本系統**不**預測高點，也不保證報酬；它用事前寫死的規則換取較小的回撤。

```
資料 ──► L1 大盤狀態機 ──► 持股上限（總開關）
     ├─► L3 安全篩選 ───► 可交易股票池（硬性排除 + 安全分）
     └─► L2 趨勢評分 ───► 在股票池內排名
                    ▼
               L4 進出場與部位 ──► 賣出清單（每日）/ 買入清單（每週）──► 交易日誌 ──► 監控紅黃燈
```

## 模組

| 模組 | 檔案 | 內容 |
|---|---|---|
| L1 大盤 | `twtrend/l1_regime.py` | 9 項指標加權分數；四狀態（多頭 / 轉弱警戒 / 空頭 / 打底轉強）；遲滯、確認期、分批調整；可選等權重或扣除台積電的判斷指數、多週期平均 |
| L2 趨勢 | `twtrend/l2_trend.py` | 依「策略檢視」修正版：gate = 收盤 > MA100、12 個月報酬 > 0、排除極端 3%、52 週高 ≥ 85%；評分 = 52 週高、營收 YoY、相對強度、12 個月報酬的全市場百分位 |
| L3 安全 | `twtrend/l3_safety.py` | 硬性排除（全額交割、處置、注意股、會計師意見、流動性、市值、新上市、虧損）；風險分（F-Score、Z-Score、償債、應計、業外、質押、董監轉讓、更換會計師、股本稀釋、特異波動、樂透特性、融資使用率）；金融 / 營建業改用產業內百分位 |
| L4 部位 | `twtrend/engine.py`、`portfolio.py` | 風險平價部位（每筆 0.5%）、單筆 30 萬、產業 30%、相關分群 1.5R、日均量 5%、組合波動上限；出場精簡為 4 層 + 安全分待出場 |
| 持股健檢 | `twtrend/health.py` | 實盤手冊第 6 點：每檔持股判定 健康 / 減弱 / 結束（8 個檢查項目）；只有「結束」才出場，「減弱」減碼時優先賣；每週結果寫入日誌 |
| 回測 | `twtrend/backtest.py` | 隔日開盤成交、漲跌停鎖死、盤中極端停損、限價不追價、成本與稅、2015/6 前後 7%/10% 制度；三基準比較、歷次空頭期、制度前後分段 |
| 統計 | `twtrend/stats.py` | 損益兩平勝率、R 統計、獲利集中度、區塊 bootstrap CI、順序重排蒙地卡羅、Kelly、CUSUM、Deflated Sharpe、PBO |
| 監控 | `twtrend/monitor.py` | 黃燈 / 紅燈判定與資金階段升降級 |
| 實盤 | `twtrend/live.py` | 每日 / 每週作業、委託、成交回報、交易日誌、決策日誌、提醒 |
| 推播 | `twtrend/notify.py` | Telegram、LINE Messaging API（LINE Notify 已停止服務） |
| 模擬盤 | `twtrend/paper.py`、`broker.py` | 類似股市大富翁的虛擬帳戶：依訊號自動下單、真實行情撮合（與回測同一套撮合規則）、可手動下單、逐日績效、排行榜 |
| 規則檢討 | `twtrend/review.py` | 分組績效（出場原因、安全分、趨勢分、L1 狀態、手動與否）、出場後走勢、候選股機會成本、A/B 規則比較、修正建議 |
| 儀表板 | `twtrend/dashboard.py` | 單一 HTML：大盤燈號、候選清單、持股安全分與停損距離、統計、R 分布、CUSUM、回撤 vs MC 百分位、紅黃燈 |
| 資料 | `twtrend/data/` | FinMind、證交所 / 櫃買 OpenAPI、SQLite、合成資料 |

## 安裝

```bash
pip install -e ".[dev]"
python -m twtrend config > config/default.yaml   # 匯出預設參數後自行修改
```

## 快速體驗（合成資料，不需網路）

```bash
python -m twtrend demo                 # 產生 60 檔、2014–2024 的合成市場
python -m twtrend backtest             # 回測 + 三基準 + output/backtest/report.html
python -m twtrend simulate             # 重現設計書「統計檢視」的模擬
```

合成資料只用來驗證程式邏輯，**回測數字不代表任何真實績效**。

## 接上真實資料

```bash
export FINMIND_TOKEN=...                                 # FinMind 帳號的 API token
python -m twtrend update --start 2005-01-01              # 全市場（含下市股）；第一次很久
python -m twtrend flags                                  # 證交所 / 櫃買：處置、注意、全額交割（只有當日，需每日累積）
```

標「待確認」的欄位與端點（`data/finmind.py` 的 `FIN_MAP`、`data/twse.py` 的 `ENDPOINTS`）上線前必須對照官方文件驗證。
董監質押、會計師意見、更換會計師、增資等（`governance_monthly`、`audit_opinion`、`corporate_events`）需另外從公開資訊觀測站整理後寫入；
沒有這些資料時，相關扣分項自動不計。財報以法定期限推估公告日，避免未來函數。

## 實盤例行作業（對應操作手冊）

| 頻率 | 時間 | 指令 |
|---|---|---|
| 每日 | 收盤後資料更新完成 | `twtrend update --start <近幾日>`、`twtrend flags`、`twtrend daily` |
| 每日 | 開盤 | 依 `twtrend orders` 執行賣出；成交後 `twtrend fill <id> <價格>` |
| 每週 | 週五收盤後 | `twtrend daily`（含持股健檢「結束」的賣出）、`twtrend weekly`（排名、買入清單、健檢報告） |
| 每週 | 週一開盤 | 依清單限價買入、設好極端保護停損條件單；未成交 `twtrend cancel <id>` |
| 每月 | 月營收公告後 | `twtrend monitor`、`twtrend dashboard` |
| 每季 | 財報公告後 | `twtrend update`（財報）後重跑 `twtrend backtest` |

偏離訊號時用 `twtrend fill <id> <價格> --deviation "原因"` 記錄。
推播：設定 `TWTREND_TELEGRAM_TOKEN` / `TWTREND_TELEGRAM_CHAT_ID` 或 `TWTREND_LINE_CHANNEL_TOKEN` / `TWTREND_LINE_USER_ID`；
未設定時只印在終端機（`--dry-run` 同效）。

cron 範例（台北時間）：

```cron
30 17 * * 1-5  cd /path/to/repo && twtrend update --start $(date -d '-7 day' +\%F) --prices-only && twtrend flags && twtrend daily
0  19 * * 5    cd /path/to/repo && twtrend weekly
45 17 * * 1-5  cd /path/to/repo && twtrend paper run && twtrend paper report
```

## 持股趨勢健檢（實盤手冊第 6 點）

```bash
twtrend health            # 目前持股的 健康 / 減弱 / 結束 與觸發項目（也會在 weekly、dashboard 顯示）
```

| 檢查項目 | 頻率 | 健康 | 減弱 | 結束（出場） | 關閉參數 |
|---|---|---|---|---|---|
| 追蹤停利線（3×ATR） | 每日 | 收盤在線上 | — | 收盤跌破，隔日開盤賣出 | — |
| 長期趨勢：收盤 vs MA100、12M 報酬 | 每週 | 兩者皆正向 | 一項轉負 | 兩項皆轉負 | `health.long_trend_exit` |
| 相對強度：3M 報酬 vs 等權重指數 | 每週 | 贏過指數 | 輸給指數 | 趨勢分跌出前 50% | `health.rank_exit` |
| 52 週高點接近度 | 每週 | ≥ 85% | < 85% | 不單獨出場 | — |
| 營收動能：月營收 YoY | 每月 | 正成長 | 連 2 月轉弱 | 連 3 月負成長且長期趨勢減弱 | `health.revenue_exit` |
| 停滯檢查：是否達 +1R | 每週 | 8 週內達到 | 8 週未達 | 12 週未達（待驗證） | `health.time_stop` |
| 安全性 | 每日 / 每季 | 無旗標、≥ 70 | 50–69 | 硬性排除，或 < 50 後 20 日內 | — |
| 大盤 L1 | 每日 | 多頭 | 轉弱警戒 | 依上限減碼，先賣「減弱」 | — |

- 每週項目的「結束」只在每週健檢日（週五收盤後）觸發，非健檢日沿用上次結果但不出場。
- 每週健檢結果寫入 `health_checks`（實盤）或 `paper_health`（模擬盤），每筆交易記錄持有期間曾出現的「減弱」項目。
- `twtrend paper review` 會統計「減弱」後 20 / 60 日的實際走勢，對照同期「健康」持股，並附信賴區間；
  只有顯著較差且樣本足夠的項目，才會被列為「可考慮升級為出場條件」。各「結束」項目的出場後走勢也會列出，太早出場的項目會建議以 A/B 帳戶關閉驗證。
- 偏離健檢規則時用 `twtrend fill ... --deviation "理由"` 記錄，`twtrend monitor` 會按月列出偏離的結果。

## 模擬盤（先驗證、再上實盤）

類似 CMoney 股市大富翁：每個模擬帳戶有虛擬資金，依系統訊號自動下單，以隔日開盤價撮合，
扣手續費、證交稅與滑價，逐日記錄權益、持股、每一筆交易與每週候選清單（買 / 不買及原因）。

```bash
twtrend paper create baseline                                  # 用目前規則開一個帳戶（從今天開始前進測試）
twtrend paper create wide_trail --set l4.trailing_atr_mult=3.5 # 同時開一個「修改版規則」帳戶做 A/B
twtrend paper run                                              # 每日資料更新後執行：撮合 + 產生訊號
twtrend paper status baseline                                  # 持股、待撮合委託、權益
twtrend paper order baseline buy 2330 100 --note "看好"         # 手動下單（標記為人為干預，績效分開統計）
twtrend paper board                                            # 排行榜（Calmar、MDD、勝率、期望值、相對 0050）
twtrend paper review baseline --phase forward                  # 規則檢討
twtrend paper compare baseline wide_trail                      # A/B：期望值差的 95% 信賴區間
twtrend paper report                                           # output/paper.html
```

- **forward vs replay**：帳戶建立日（今天）之後的交易日是 forward，才是真正的樣本外證據；
  `--start` 設在過去的部分是 replay（歷史重播），只能用來熟悉流程，不能當成驗證。
- **規則版本**：每個帳戶保存完整參數快照與雜湊，事後可追溯是哪一版規則產生的績效；`paper compare` 會列出兩帳戶的規則差異。
- **與實盤的關係**：模擬盤就是操作手冊的「階段 0 流程試跑」。紅黃燈與 `twtrend monitor` 的標準同樣適用。

### 用模擬盤修正規則的流程

1. `paper review` 列出「候選修正 / 觀察 / 正常」。只有樣本數足夠、且分組差異的信賴區間不重疊才會列為候選修正
2. 對每個候選修正開一個 A/B 帳戶（`--set 參數=值`），與基準帳戶並行前進測試
3. 兩邊各至少 30 筆以上，`paper compare` 顯示「B 顯著較好」，且不是靠回撤變大換來的
4. 用 `twtrend backtest` 檢查 2015 前後兩段都穩定、期望值打對折後仍為正
5. 才把新參數寫進 `config/default.yaml`，並保留舊版帳戶繼續跑一段時間作對照

不要因為連虧或一次股災就改規則（設計書「不要做的優化」）；黃燈只檢討、不改規則。

## 資金階段

| 階段 | 條件 | 資金 | 每筆風險 |
|---|---|---|---|
| 0 流程試跑 | 可產生訊號 | 紙上交易 2–4 週 | — |
| 1 小額實盤 | 階段 0 無流程錯誤 | NT$100 萬 | 0.25% |
| 2 全額實盤 | 滑價與 1R 執行在黃燈內、遵守率 ≥ 95% | NT$300 萬 | 0.5% |

`twtrend monitor --stage 1` 會依紅黃燈給升降級建議。前 30 筆只檢查執行品質，不判斷策略好壞。

## 與設計書的對應與取捨

- L2 採「策略檢視」的修正版，不使用橫斷面 6 個月動能前 30%；ADX、量能、法人買超「待驗證」，預設不納入。
- 出場：L3 規則性排除、極端保護停損、L1 減碼、一個追蹤停利（Chandelier 3×ATR22）、安全分 < 50 的 20 日待出場，
  加上持股健檢的「結束」條件（長期趨勢、趨勢分跌出前 50%、營收、時間止損，各自可關閉）。
  注意：健檢的「趨勢分跌出前 50%」等同「策略檢視」建議刪除的排名輪動出場；預設依新版手冊啟用，建議用
  `twtrend paper create no_rank --set health.rank_exit=false` 並行驗證。週線 MA60 破線（`l4.use_ma60_exit`）預設關閉。
- 空頭持股上限預設 0%、打底轉強 40%，由回測決定。
- L1 判斷指數預設加權指數；`l1.index_source: equal_weight` 或 `ex_tsmc` 對應「失效情境」優化 #1。
- 快速熔斷、V 型反轉的寬度急升回補已以「寬度推力」進入打底轉強實作；其餘「待驗證」項目未加入，以免參數過多。
- 所有數字都是起始值，正式上線前請用含下市股的真實資料做 walk-forward 驗證，並把期望值打對折後仍為正才上線。

## 測試

```bash
pytest -q
```
