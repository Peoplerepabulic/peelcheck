# PeelCheck — 无实物香蕉数据方案（已核查）

结论：没有香蕉不阻塞。训练靠公开数据集，实拍只是 demo 加分项。

## 训练数据（主力，不需要你买香蕉）
- Kaggle `Banana Ripeness Classification Dataset`（13,478 张，多熟度分级）
  - 来源页: https://www.kaggle.com/datasets/shahriar26s/banana-ripeness-classification-dataset
  - 用法：比赛期间下载、清洗、重划分 train/val/test，当场训练并记录曲线/混淆矩阵。
  - 合规：数据集是公开 resource，在 README + Devpost 的 Resources 里引用来源即可；
    被评的是你搭的训练管线和 App，不是让你从零拍 1 万张照片。
- Kaggle `Banana Ripeness Image Dataset`（同一批香蕉连续 10 天的时序图）
  - 来源页: https://www.kaggle.com/datasets/wiratrnn/banana-ripeness-image-dataset/data
  - 用法：作增补集 / 做 “Day N → 熟度” 的误差分析图，视频里很好讲。

## 类别映射（先定死，训练时不再改）
- 原始多级 → PeelCheck 3 类：
  - Underripe：偏绿、绿大于黄
  - Ripe：全黄、无/极少褐斑
  - Overripe：黄底褐斑多、发黑
- 映射表写进 `train/label_map.md`，评估时按此口径报 per-class precision/recall。

## 增广（补“没香蕉”的短板，还能当技术亮点讲）
- 几何：rotation / flip / random crop
- 颜色：brightness/contrast/hue jitter —— 模拟不同光照
- 合成褐斑：在 Ripe 图上程序化叠加褐色斑点 → 扩增 Overripe 类，缓解类别不平衡
  （这一段在视频里可以讲成 “synthetic generation for the minority class”，贴 Image AI 主题）

## Demo 输入（三选一，按成本从低到高）
1. 零出门：直接用 test 集留出图上传 live demo（可复现、必成功，适合评委点验）。
2. 低成本加分（推荐）：去最近超市买 3 根不同熟度的香蕉回来拍真图上传，
   视频里 “真香蕉 → 真预测” 比纯数据集图更有说服力，Design/Demo 分更高。
   - 最近：Schnucks Richmond Center, 6600 Clayton Rd（距 2020 McCausland Ave 约 0.9 英里），
     周六 6:00 AM–10:00 PM 营业；买青皮、黄皮、带褐斑各 1 根即可，总价约 $2–3。
3. 纯合成演示：对同一张香蕉图做色相推移，展示模型随 “变黄→长斑” 的预测变化曲线
   （适合放在视频 1:00 处的管线讲解，不替代真实 demo）。

## 时间（solo 可执行）
- 0.5h 下载+清洗+映射 → 1h Colab 训练 → 0.5h 评估导出 → 1.5h Streamlit+部署 → 1h 视频+Devpost
- 合计约 4.5h 出可交版；买香蕉往返约 20 分钟，不占关键路径。
