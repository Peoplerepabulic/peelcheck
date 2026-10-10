# Banana Hacks 2026 — 选题备忘（待用户确认）

> 比赛：Banana Hacks 2026（Image AI 主题，线上）
> 官网：https://www.bananahacks.tech
> Devpost：https://banana-hacks-1111.devpost.com/
> GitHub 账号：Peoplerepabulic

## 已核实的时间线（以 Devpost + Schedule 页为准）
- Hacking 已开始：Fri Oct 9, 2026 8:45 PM EDT（= 7:45 PM CDT，中部时间）
- Submissions Close：Sun Oct 11, 2026 11:59 PM EDT（= Sun Oct 11 10:59 PM CDT）
  - Devpost 列表显示为 Oct 12, 2026 12:00 AM EDT to submit，与上行一致（午夜）。
  - ⚠️ 官网 Register / About / Prizes 页写的是 “Oct 11, 2026 at 11:59 PM AoE”。
    AoE = UTC-12，换算后 = Mon Oct 12, 2026 06:59 AM CDT，比 EDT 截止晚约 8 小时。
  - **在用户确认前不要依赖 AoE 口径。默认按更早的 EDT 截止倒排：Sun Oct 11 10:59 PM CDT。**
- Closing Ceremony：Mon Oct 12, 2026 6:00–7:00 PM EDT

## 资格
- 学生，14–24 岁，在读（高中/大学均可），全球可参加
- 队伍 1–4 人；单人队与满员队同池评审
- 免费；Discord 为社区中心，Zoom 办工作坊/仪式（均录播）

## 提交要求（缺一不可）
1. Project Name
2. Team members listed（Devpost 上列全；若组队，每人都要有实际贡献）
3. ≤3 分钟 YouTube 公开视频——时长严格卡 3 分钟；视频须覆盖：brainstorming、inspiration、design process、resources、final product
   - Rules 页写 “2–3 minute demo video to be judged”，与 ≤3 分钟一致
4. Devpost 项目描述——与视频同样覆盖上述五点
5. GitHub 源码链接（必填）
6. 所有代码必须在比赛期间内创作；可复用“想法”，不可复用旧代码/素材，不可提交旧项目
7. 禁止跨黑客松交叉提交（cross-submission is not allowed），违规取消资格
8. Prizes 页补充：working demo（live URL 或视频）+ GitHub repo + ≤500 words 的 what/why 说明
   - “A working link is better than a polished video. Judges will poke at live demos where they can.”

## 评审（五项等权重，各 1–5 分）
1. Technical Execution and Difficulty — 是否有真实的 Image AI / CV / 生成模型工程挑战、模型选型与非平凡集成；**“Thin API wrappers should score lower.”**
2. Innovation and Creativity — 是否有新角度/出人意料的组合；generic clones 扣分
3. Use of the Image AI Theme — Image AI 必须是核心，抽掉就垮才算达标
4. Impact and Usefulness — 明确用户、明确收益、有赛后继续被用下去的路径
5. Design, Presentation, and Demo — 界面与视觉输出干净；3 分钟视频 + Devpost 描述讲清楚“是什么、解决什么、怎么工作”

## 奖项
- $500 cash（top project，1 名）
- CodeCrafters VIP：1st 2 年 / 2nd 1 年 / 3rd 6 个月（获奖队每人一份）
- Interview Cake licenses ×50
- .xyz 域名（首年免费）
- 类别奖（自动参评，无需另行申请）：Best Creative Tool / Best Fine-tune / Most Surprising Output / People’s Choice（demo day 投票）

## 与 HomeKeeper 的关键区别
- HomeKeeper（Amazon Build, Ship, Shape）：AWS + MCP + Lambda/DynamoDB + 模拟器，周期 10/4–10/22，工程量大。
- Banana Hacks：周末赛、Image AI 单主题、评委明确惩罚“薄 API 套壳”，奖励“把生成式 AI 放到创作者手里 / 真正的微调管线 / 出人意料的输出”。
- 本项目策略：**单页 Web Demo + 一条真实的 Image AI 管线（检测/分类/分割/生成至少其一为自研或微调），2 天内可上线，视频 3 分钟讲清五点。**

---

## 候选选题（均按“比 HomeKeeper 简单且短”设计）

### A. PeelCheck — 香蕉熟度分类器（主推：最稳、最贴题）
- 一句话：拍一张香蕉照片 → 输出熟度阶段（未熟/刚好/过熟）+ 对应用法建议（直接吃/做香蕉面包/冷冻打冰沙）。
- Image AI 核心：**自训练/微调的图像分类模型**（MobileNetV3 / EfficientNet-Lite 迁移学习），含数据收集、增广、训练曲线、混淆矩阵——正好命中 **Best Fine-tune** 且避开“套壳”扣分。
- Demo 形态：Streamlit / Gradio 单页（上传 → 预测 + 置信度条 + 建议卡片）；可部署到 Hugging Face Spaces，给评委一个可点的 live URL。
- 数据：公开香蕉熟度图片集 + 手机自拍补拍（比赛期间当场拍，符合“期间创作”要求）；规模小，Colab 免费 GPU 1 小时内可训完。
- 预估工作量：约 4–6 小时出可提交版本（数据 1h + 训练 1h + 前端 1.5h + 部署/录屏 1h）。
- 风险：题材偏“玩具”，Impact 分需靠文案拉（减少食物浪费、帮超市/食堂分拣）；技术分靠训练管线与误差分析补。

### B. FridgeLens — 冰箱照片 → 菜谱 + 成品图（冲击力强，稍复杂）
- 一句话：拍一张冰箱/食材照片 → 目标检测列出食材 → LLM 给 3 道菜谱 → 文生图生成成品摆盘图。
- Image AI 核心：零样本检测（OWL-ViT / YOLO-World）+ 文生图，双管线串联，Image AI 抽掉即垮。
- Demo 形态：Streamlit 单页（照片墙 → 食材标签 → 菜谱卡 → 生成图）；部署 HF Spaces。
- 预估工作量：约 6–8 小时；主要风险是检测在杂乱冰箱照片上不稳、文生图延迟/额度。
- 适合冲：Innovation + Impact（明确用户：独居学生/新手做饭）；类别可对 **Best Creative Tool / Most Surprising Output**。

### C. StudySnap — 白板/课本照片 → 干净讲义 + 闪卡（对你背景最友好）
- 一句话：上传课堂白板或课本照片 → 透视矫正 + 去阴影 → OCR/版面提取 → 自动生成 Anki 式闪卡（正面问题、背面答案，附原图裁剪）。
- Image AI 核心：文档矫正（OpenCV）+ 图像裁剪/分类（按信息量排序裁图）+ 版面理解；CV 成分扎实，非套壳。
- Demo 形态：单页 Web（原图 → 矫正图 → 闪卡翻卡交互），可导出 CSV 给 Anki。
- 预估工作量：约 5–7 小时；风险是 OCR 中文/手写公式不稳，demo 需选清晰样本。
- 适合冲：Impact（学生党刚需，与你的 WashU 身份天然契合）。

## 推荐
- 若只求“稳交 + 不熬夜”：选 **A PeelCheck**。
- 若想冲名次且愿多花 2 小时：选 **B FridgeLens**。
- 若想和你的学生身份/PhD 申请故事线呼应：选 **C StudySnap**。

## 下一步（选定后我直接执行）
1. 在 GitHub 新建 public repo（Peoplerepabulic 下，名称待定），所有 commit 必须在比赛期间内（已在进行中，注意起始时间戳）。
2. 搭最小可运行管线 → 部署 live demo（HF Spaces 优先）。
3. 写 Devpost 描述草稿（≤500 words，覆盖 brainstorming/inspiration/design/resources/final product 五点）。
4. 列 3 分钟视频分镜（0:00 问题 → 0:30 灵感与设计 → 1:00 怎么做（模型/管线）→ 1:45 现场 demo → 2:45 总结与展望），严格 <3:00。
5. 提交前双人核对清单（视频公开？仓库公开？描述五点齐？无交叉提交？）。
