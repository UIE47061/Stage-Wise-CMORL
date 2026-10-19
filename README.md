# Stage-Wise CMORL — Little White V3

本專案以 [rllab-snu/Stage-Wise-CMORL](https://github.com/rllab-snu/Stage-Wise-CMORL) 為基礎，加入小白 V3 四足機器人的後空翻任務、教師微調、學生蒸餾與隨機化實驗。原始研究為 [Stage-Wise Reward Shaping for Acrobatic Robots: A Constrained Multi-Objective Reinforcement Learning Approach](https://arxiv.org/abs/2409.15755)。

保留原版 Go1（後空翻、側空翻、側滾、雙前腳行走）與 H1（後空翻、雙手行走）任務。小白實驗的 stage 切換與嚴格成功評估分開計算，進入 `land` 不等於成功完成後空翻；各版本的量測與限制見下方文件。

## 專案結構

```text
Stage-Wise-CMORL/
├── algos/                  # CMOPPO 教師、student 與設定
├── assets/                 # Go1、H1、小白 V3 的 URDF／meshes
├── tasks/                  # 任務環境、獎勵與設定
├── utils/                  # normalization、紀錄及 rollout 工具
├── scripts/                # 環境啟用、診斷與測試工具
├── docs/little_white.md    # 小白教師／學生操作說明
├── main_teacher.py
├── main_student.py
└── requirements.txt
```

`results/`（模型、normalization、設定備份）與 `runs/`（log、評估 JSON、lock）是本機產物，不納入 Git。此 repository 不包含預訓練權重；文件中的歷史 seed 與 checkpoint 必須另外取得，或自行訓練。

## 環境設定

本機使用 Python **3.8.20**、PyTorch **1.12.1+cu113**、NumPy **1.21.5**、Isaac Gym **Preview 4** 與 IsaacGymEnvs **1.5.1**。`requirements.txt` 記錄本機直接依賴版本，不是完整環境鎖定檔。原專案的 Python 3.7 需求與此版本的本機環境不同。

準備 Linux、NVIDIA GPU／驅動與建置工具，將 Isaac Gym Preview 4 SDK、IsaacGymEnvs 放在 repository 外。於已準備好的 Python 3.8 環境安裝：

```bash
# 路徑請改為自己的 SDK 與 IsaacGymEnvs 位置
export ISAACGYM_ROOT=/path/to/isaacgym-preview4/isaacgym
export ISAACGYM_ENVS_ROOT=/path/to/IsaacGymEnvs

python -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install -e "$ISAACGYM_ROOT/python"
python -m pip install -e "$ISAACGYM_ENVS_ROOT"
python -m pip install -r requirements.txt
```

每次開啟 Bash 終端機後，進入專案根目錄並執行：

```bash
# 自訂安裝位置時設定此變數；指向 Python 環境，並非 SDK 原始碼
export ISAACGYM_VENV=/path/to/isaacgym-python-env
source scripts/activate_env.sh
```

啟用腳本依序使用 `ISAACGYM_VENV`、已啟用的 virtualenv／Conda，或相對於專案的 `../deps/isaacgym-preview4/.venv`，並加入該環境的共享函式庫路徑。使用既有相鄰目錄配置時可省略 `export`。以下命令均從專案根目錄執行；有視窗播放另需可用的圖形桌面。

## 教師訓練與播放

從頭訓練小白自由 RL 基準（請選擇尚未使用的 seed）：

```bash
python main_teacher.py \
  --task_cfg_path tasks/little_white_v3_backflip.yaml \
  --algo_cfg_path algos/comoppo/little_white_v3_backflip.yaml \
  --seed 1 --total_steps 1000000 --save_freq 100000
```

結果位於 `results/littlewhitev3backflip_comoppo/seed_1/`。完成對應步數後播放：

```bash
python main_teacher.py --test --render \
  --task_cfg_path tasks/little_white_v3_backflip.yaml \
  --algo_cfg_path algos/comoppo/little_white_v3_backflip.yaml \
  --seed 1 --model_num 1000000
```

續訓時使用同一組 task／algo／seed，加上已存在的 `--model_num`，並把 `--total_steps` 設為新的累計目標。原版 Go1／H1 可替換為對應 task 與 algo，例如 `go1_backflip.yaml`；需要 Weights & Biases 時加 `--wandb`。

## 小白版本與學生訓練

| 用途 | Task 設定（`tasks/` 下） | Algo 設定（`algos/` 下） |
|---|---|---|
| 自由 RL 教師基準 | `little_white_v3_backflip.yaml` | `comoppo/little_white_v3_backflip.yaml` |
| 落地微調 | `little_white_v3_backflip_landing.yaml` | `comoppo/little_white_v3_backflip_landing.yaml` |
| 等待／蹲下微調 | `little_white_v3_backflip_preparation.yaml` | `comoppo/little_white_v3_backflip_preparation.yaml` |
| 漸進式學生蒸餾 | `little_white_v3_backflip_preparation.yaml` | `student/little_white_v3_backflip_dagger.yaml` |
| 學生隨機化 V1 | `little_white_v3_backflip_randomized.yaml` | `student/little_white_v3_backflip_randomized.yaml` |
| 學生隨機化 V2 | `little_white_v3_backflip_randomized_v2.yaml` | `student/little_white_v3_backflip_randomized_v2.yaml` |

訓練、續訓、播放及跨電腦搬移方式統一見 [小白操作說明](docs/little_white.md)。

學生使用 `main_student.py`，必須先有與 task 相容的教師 checkpoint 與 normalization。小白 DAgger／隨機化 algo 預設引用教師 **seed 24／100000000**；這些檔案不隨 repository 提供。使用自己的教師時，請修改 algo 的 `teacher.seed`／`teacher.model_num`，並核對來源設定。學生播放加 `--test --render`；隨機化測試另加 `--randomized_test` 並搭配 randomized task。

## 驗證與診斷

在 Isaac Gym 環境執行現有回歸測試（不啟動長時間訓練）：

```bash
source scripts/activate_env.sh
python scripts/test_little_white_backflip.py
python scripts/test_student_rollout.py
```

教師無視窗評估使用 `scripts/diagnose_little_white.py`，參數與輸出選項可用 `--help` 查看；動作軌跡摘要使用 `scripts/summarize_little_white_motion.py TRACE.json`。診斷需提供對應 checkpoint 與 normalization。

## Git 與模型保存

提交原始碼、YAML 設定、機器人資產、腳本與文件。`.gitignore` 排除訓練結果、log、checkpoint、normalization、Python 環境及本機憑證；shell 腳本與模型貼圖可正常追蹤。

```bash
git status --short
git diff --check
git add .
git diff --cached --stat
git commit -m "Add Little White V3 training workflows and documentation"
git remote -v
```

推送前確認遠端是自己的 repository。本工作副本原有 `origin` 指向原作者 `rllab-snu/Stage-Wise-CMORL`；若要推到自己的 GitHub，先設定自己的 remote，再執行 push。

checkpoint 應另外保存／分享，並一起保留相同步數的 normalization 與 `backup/` 設定。只有 `.pt` 通常不足以正確重現；舊 checkpoint 應使用其當次備份設定。

## 授權

保留原專案 [MIT License](LICENSE) 與資產附帶的 [Unitree BSD 3-Clause License](assets/LICENSE)。
