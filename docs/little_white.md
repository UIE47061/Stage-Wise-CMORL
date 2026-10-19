# 小白 V3 操作說明

先依[主 README](../README.md)準備環境，所有指令在專案根目錄執行：

```bash
source scripts/activate_env.sh
```

## 教師

從頭訓練使用主 README 的自由 RL 基準指令。落地與等待／蹲下微調，分別使用 `landing`、`preparation` 同名 task／comoppo algo。微調來源與參數可用 `python main_teacher.py --help` 查看；歷史 checkpoint 應搭配該次 `backup/` 內的設定。

五階段為 stand、sit、jump、air、land。基準控制採 `default_joint_positions + 0.25 × action`，再以 0.5 平滑；各階段關節動作由 policy 學習。進入 land 與完整空中翻轉、穩定落地的嚴格成功判定分開計算。

## 學生

先備妥教師模型及 normalization。DAgger／隨機化 algo 的 `teacher` 預設為 comoppo seed 24／100000000；使用自訓教師時，修改 `teacher.seed`、`teacher.model_num`，並確認 task 相容。

從頭蒸餾範例；請選擇未使用的 student seed：

```bash
python main_student.py \
  --task_cfg_path tasks/little_white_v3_backflip_preparation.yaml \
  --algo_cfg_path algos/student/little_white_v3_backflip_dagger.yaml \
  --seed 1 --total_steps 1000000 --save_freq 100000
```

結果位於 `results/littlewhitev3backflip_student/seed_1/`。DAgger 設定在前 10 萬步由教師控制；10–50 萬步逐回合增加學生控制機率，之後由學生控制，教師持續提供動作標籤。

接續同一分支的完整 checkpoint，使用原 task／algo／seed，加上 `--model_num`，例如：

```bash
python main_student.py \
  --task_cfg_path tasks/little_white_v3_backflip_preparation.yaml \
  --algo_cfg_path algos/student/little_white_v3_backflip_dagger.yaml \
  --seed 1 --model_num 1000000 \
  --total_steps 1500000 --save_freq 100000
```

`total_steps` 是累計目標；replay buffer 會重新收集。請勿對同一 seed 同時啟動多個訓練。

## 隨機化與播放

V1 使用 `tasks/little_white_v3_backflip_randomized.yaml` 與 `algos/student/little_white_v3_backflip_randomized.yaml`；V2 使用名稱帶 `_randomized_v2` 的對應設定。實際擾動範圍以 YAML 為準。這兩組 algo 的學生控制排程為 100 萬步暖機、接著 100 萬步增加機率，與 DAgger 基準不同。

播放已完成的模型：

```bash
python main_student.py --test --render \
  --task_cfg_path tasks/little_white_v3_backflip_preparation.yaml \
  --algo_cfg_path algos/student/little_white_v3_backflip_dagger.yaml \
  --seed 1 --model_num 1000000
```

隨機化播放需換成對應 randomized task／algo，再加 `--randomized_test`。省略此旗標為固定條件測試。模型存在與訓練結束都不代表通過後空翻驗收，需獨立評估。

## 跨電腦同步

透過 Git 同步程式、YAML、資產與本文件。`results/`、`runs/` 另行搬移；接續訓練或播放需保留相同步數的 checkpoint、normalization 與 `backup/`，並備妥學生 algo 指定的教師結果。新電腦設定自己的 `ISAACGYM_VENV`。

舊實驗配置、六份詳細歷史文件及固定 seed 的續訓腳本已移至原整理電腦的 repository 外備份目錄 `../Stage-Wise-CMORL_local_history/`，不屬於目前同步內容。
