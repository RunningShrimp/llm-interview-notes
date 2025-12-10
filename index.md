# AI大模型应用开发岗位面试知识点清单（含详实示例回答与代码示例，完整版）

> 📅 更新时间：2024年12月（含补充：MCP / A2A / 视频→文本多模态 / Function Calling 进阶 / RAG 处理图片与表格 / 代码示例 / 后端转岗指引）  
> 🎯 适用范围：AI大模型应用开发工程师、LLM应用架构师、AI应用开发工程师（特别适合 3–5 年后端经验转岗同学）  
> 📊 覆盖度：约 80% 的面试高频问题  
> 🔖 深度标记：了解概念 / 理解原理 / 能动手实现 / 能优化设计

---

## 📋 使用说明

- 先扫目录定位薄弱点，再结合项目经验准备「可讲的故事」与量化指标。
- 回答建议采用「总—分—总」结构，多用自己项目里的数据、对比和取舍理由。
- 面试中尽量结合：
  - 目标：想解决什么问题？业务/技术痛点是什么？
  - 设计：整体架构和方案取舍。
  - 实现：关键技术细节、工程挑战、性能/成本优化。
  - 指标：效果（准确率/解决率）、效率（延迟/QPS）、成本（算力/人力）、稳定性。
  - 迭代：如何基于日志和反馈持续优化（数据闭环）。

---

## 一、大模型核心概念

### 1.1 大模型基础理论（理解原理）

**1.1.1 大模型训练流程**

整体流程：**预训练 → 指令微调（SFT）→ 对齐（对齐/偏好优化）**

- 预训练（Pretrain）  
  - 目标：学习通用语言/知识能力。  
  - 数据：大规模通用语料（网页、书籍、代码等），自监督目标（Causal LM / MLM）。  
  - 要点：
    - 数据清洗去重：去掉垃圾文本、毒性内容、泄露信息，降低重复度。  
    - 验证集：分布与训练集相近，避免泄露测试集。  
    - tokenizer 选型：BPE / SentencePiece 等，对中文/代码友好。  
    - 训练监控：loss 曲线、梯度范数、learning rate schedule、early stop。  
    - 工程：混合精度（bf16/fp16）、梯度累积、分布式训练、checkpoint 容灾。

- 指令微调（SFT, Supervised Fine-tuning）  
  - 目标：让模型「听懂人话」并按指令输出。  
  - 数据：高质量「指令-回答」对（通用+领域），覆盖问答、推理、代码、工具调用等。  
  - 要点：
    - 格式统一（包括 role、指令格式、输出风格）。  
    - 覆盖主要任务模式（对话、摘要、翻译、问答、结构化输出等）。  
    - 避免模板化和评测集泄露。

- 对齐（RLHF / RLAIF / DPO / KTO 等）  
  - 思路：通过人类偏好数据或自动偏好信号，让模型倾向于输出「更安全、更有用」的回答。  
  - RLHF：偏好数据 → 奖励模型 → 强化学习（如 PPO）。  
  - RLAIF：用「评估模型」自动生成偏好标签。  
  - DPO/KTO：直接在偏好数据上优化策略，无需显式奖励模型。  
  - 要点：偏好数据质量；安全规则和奖励设计；对齐数据/奖励模型版本可溯源。

---

**1.1.2 大模型核心能力**

- 涌现能力：参数/数据/算力到一定规模后，模型在推理、多步指令、多语言上能力突增。  
- 上下文理解与指令跟随：对系统指令、角色、示例非常敏感。  
- 思维链推理（CoT）：通过「先思考再作答」的方式提升复杂推理效果。  
- 零样本/少样本学习：在几乎没有标注的情况下解决新任务。  
- 多语言/多模态：支持多语种、图文多模态甚至视频/音频。

**评估维度**

- 通用能力：MMLU / BBH / GSM8K / HellaSwag 等。  
- 安全/真实性：TruthfulQA / AdvBench / 红队测试。  
- 指令跟随：AlpacaEval / IF eval。  
- 业务效果：自建 golden set（真实问题 + 理想答案），持续迭代。

---

**1.1.3 参数与结构**

- 参数：模型中的权重/偏置，规模越大，潜在能力越强，但推理/存储/带宽成本上升。  
- 指令微调：在基座模型上通过 SFT/对齐，使模型更符合人类期望。  
- PEFT（参数高效微调）：
  - LoRA / QLoRA / Prefix Tuning / AdaLoRA 等，在基座不变前提下，用少量参数适配新任务。  
  - 优点：显存友好、易部署、便于多任务多版本管理。

---

**1.1.4 幻觉问题（能动手实现相关缓解）**

- 成因：
  - 训练数据缺失/噪声，模型「硬编」填补空白。  
  - 语言模型本质是「下一个 token 预测」，倾向于输出看起来合理的内容。  
  - 缺乏检索或检索质量差，模型只能依赖参数内记忆。  
  - 温度/采样策略过于激进。

- 缓解策略：
  - RAG：引入外部知识库，要求回答必须基于检索内容，并引用来源。  
  - 提示设计：显式要求「没有依据就说不知道」，限制编造细节。  
  - 解码策略：温度降低、限制生成长度、避免极端采样参数。  
  - 结构化输出：用 JSON/Schema/Function Calling + 规则/正则/Schema 校验。  
  - 事实校验：二次调用 LLM 做 fact-check，或用规则/知识库/外部服务校验。  
  - 高风险场景：设定人工审核、白名单/黑名单规则和兜底流程。

---

### 1.2 Transformer 架构（理解原理）

**1.2.1 基本结构**

- Encoder-Decoder：经典 seq2seq 结构（翻译等）。  
- Decoder-only（GPT 系）：只用 Decoder，适合生成任务、推理效率更高。  

**关键组件**

- 多头自注意力（Multi-Head Self-Attention）  
- 前馈网络（FFN）  
- 残差连接 + LayerNorm  
- 位置编码 / 旋转位置编码（RoPE）  

**长上下文机制**

- RoPE / ALiBi / 位置插值（Position Interpolation）  
- 分块/滑窗/分层摘要  
- 专用长上下文架构（如一些新型序列模型：Mamba 等，了解即可）

---

**1.2.2 注意力缩放与多头**

- Self-Attention 中除以 $\sqrt{d_k}$：
  - 避免随着维度增大，点积值变大导致 softmax 过陡、梯度不稳定。  
  - 缩放后使方差稳定，便于训练收敛和数值稳定。

- 多头注意力：
  - 将注意力分成多个子空间（多个「关注视角」），并行建模不同关系。  

- FlashAttention / v2：
  - 通过块化计算、减少显存读写，提高长序列训练/推理效率。

---

**1.2.3 常见变体与代表模型**

- 模型类型：Prefix LM（Encoder-Decoder）、Causal LM（Decoder-only）。  
- 代表开源模型：LLaMA 3、Mistral/Mixtral、Qwen/Qwen2 系列、GLM 等。  
- 特点：
  - 优点：长依赖建模能力强、GPU 并行友好。  
  - 缺点：自注意力复杂度 $O(n^2)$，长上下文成本高，需要工程优化。

---

### 1.3 Token 与 Embedding

- Token：
  - 常见 tokenizer：BPE / WordPiece / SentencePiece。  
  - token 数直接影响：成本（按 token 计费）与上下文长度占用。  
  - 中文常被切分得更碎，需要注意切分策略和 chunk 大小选取。

- Embedding：
  - 评价维度：语言覆盖、向量维度、延迟、检索精度。  
  - 常见文本 embedding：
    - `text-embedding-3-large` / `text-embedding-3-small`  
    - bge-m3、bge-large-zh、m3e、sentence-BERT 等。  
  - 多模态 embedding：CLIP / BLIP / LLaVA / Qwen-VL 等。  
  - RAG 中需保证：
    - query 与 document 使用相同或兼容的 embedding 模型。  
    - 领域数据可考虑做 embedding 微调。

---

## 二、技术栈与工具

### 2.1 主流大模型（闭源 + 开源）

**商业 API 模型**

- GPT-4/4o/4o-mini、Claude、Gemini 等。  
- 关注：
  - 性能：推理/推断能力、长上下文支持、工具调用能力。  
  - 成本：单 token 价格、上下文长度对总成本影响。  
  - 功能：Function Calling/JSON 模式、多模态（图像/音频/视频）。  
  - 安全与合规：数据留存策略、企业版隐私保障。

**开源模型**

- 代表：
  - LLaMA 3 系列  
  - Qwen / Qwen2 / Qwen2.5  
  - Mistral / Mixtral  
  - GLM / DeepSeek 等  
- 考量：
  - 许可证与商业可用性。  
  - 中文/多语支持与代码能力。  
  - 推理速度、可量化性、社区生态。  
- 部署：
  - vLLM / TensorRT-LLM / TGI 等推理引擎。  
  - 量化方案：AWQ / GPTQ / 8bit/4bit / FP8 等。  
  - 私有化部署适合数据不出域和成本敏感场景。

---

### 2.2 开发框架与编排

**LangChain**

- 能力：链式调用（Chain）、Agent 编排（Agent+Tool）、LCEL、TextSplitter、Output Parser、Memory。  
- 优点：组件丰富，原型搭建快。  
- 注意：
  - 抽象层过多时，需谨慎控制逻辑放置位置。
  - 生产中要做好日志/Tracing，避免「黑盒逻辑」。

**LlamaIndex**

- 特点：更强调数据与索引侧抽象。  
- 核心概念：Index / QueryEngine / Retriever 等。  
- 擅长：复杂数据摄取管线、不同索引组合、多源数据查询。  
- 可与 LangChain 搭配使用：LlamaIndex 处理数据/索引，LangChain 负责上层流程。

**Agent 框架**

- AutoGen：多 Agent 协作对话。  
- CrewAI：按角色分工执行任务。  
- 基于 MCP / 自研框架的 Agent：统一工具/资源规范，更易审计和路由。

---

### 2.3 向量数据库

- 相似度：Cosine / Inner Product (IP) / L2。  
- 索引结构：IVF / HNSW / PQ / IVF_PQ 等。  
- 关键特性：元数据过滤、混合检索（向量 + 关键词）、分片与副本、多租户、安全控制。

**选型建议**

- Milvus / Zilliz：分布式、云原生，适合大规模生产环境。  
- Qdrant：Rust 实现，高性能，支持丰富过滤，Serverless / 托管服务发展迅速。  
- Weaviate：GraphQL + 向量检索，复杂查询友好。  
- pgvector：PostgreSQL 插件，将向量功能融入传统数据库，非常适合已有 PG 体系。  
- FAISS：本地库，嵌入式、实验与 POC 常用。  
- Chroma：轻量级，本地/小规模项目。

**工程优化**

- 高 QPS：HNSW 或适合内存/SSD 的 ANN 索引。  
- 存储成本：IVF_SQ8 / PQ 等压缩索引。  
- 集群：合理分片（按语种/业务线）、副本数、冷热数据分层。  
- 写入：WAL、compaction 策略；并发写入与在线查询的平衡。  
- 压测：关注延迟 P95/P99、召回率与吞吐，并考虑实际业务 query 分布。

---

### 2.4 其他关键工具

**推理优化**

- vLLM：
  - PagedAttention + Continuous Batching，提高吞吐和 GPU 利用率。  
  - 多租户场景（同时服务大量会话）友好。  
- TensorRT-LLM：
  - 靠 kernel fusion、量化（INT8/FP8）等加速，适合延迟敏感场景。  
- TGI、Triton、TGI-like：
  - 云厂商或开源提供的推理服务框架，各有生态和特性。  

**微调工具链**

- Hugging Face Transformers + PEFT：  
  - LoRA / QLoRA / Prefix Tuning / AdaLoRA 等参数高效微调方案。  
- 关键配置：
  - rank、α、dropout、学习率、epoch、梯度累积。  
- 训练技巧：
  - bf16 / fp16 混合精度、FlashAttention、梯度裁剪、权重衰减、早停。

---

### 2.5 MCP 与 A2A（多助手协同）

**MCP（Model Context Protocol）**

- 作用：统一模型与工具/资源交互的协议和规范。  
- 特点：
  - 显式声明工具/资源、输入/输出 Schema、权限要求。  
  - 工具可以跨模型/框架复用，类似「为 LLM 定义一套 OpenAPI」。  
  - 易于审计和路由，调用链可记录在 tracing 中。

**设计要点**

- 工具 Schema：用 JSON Schema 明确字段类型、必填项、取值范围。  
- 权限：区分只读/写入/高危操作，设置细粒度访问控制。  
- 错误处理：统一 err_code/err_msg 格式，便于 LLM 做错误恢复。  
- 幂等与配额：对于有副作用的工具，设计幂等键和配额控制。

---

**A2A（Assistant-to-Assistant，多助手协同）**

- 核心思想：
  - 角色分工：Planner / Executor / Judge / Router 等。  
  - 并行与冗余：多个执行者互检、主从双轨。  
  - 目标：提升复杂任务的鲁棒性与可解释性。

**工程实践**

- 消息通道：  
  - 使用 MQ / 事件流（Kafka/RabbitMQ/Redis Streams 等）传递任务消息。  
- 上下文管理：  
  - 每个 Agent 只共享必要上下文，避免全局混乱。  
- 健壮性：  
  - 超时、重试、幂等与熔断与传统微服务类似。  
- 冲突解决：  
  - 投票、优先级、规则系统，由业务逻辑最终裁决。  
- 成本/时延限制：  
  - 设置上限，超限时走降级方案（如简化 Agent 流程或用静态规则兜底）。

---

## 三、应用开发关键技术

### 3.1 Prompt 工程

**基础结构**

- 角色（system）+ 指令（instructions）+ 上下文（context）+ 用户输入（user）+ 输出格式（format）。  
- 零样本 vs 少样本：  
  - 零样本适合通用任务；  
  - 少样本可补充任务示例、风格、边界。

**高级技巧**

- CoT（Chain-of-Thought）：要求模型按步骤推理。  
- Self-Consistency：多次采样 CoT，选最一致/最合理路径。  
- ToT（Tree-of-Thought）：拓展多种思路路径并裁剪。  
- ReAct：交替「思考（Thought）」和「行动（Action）」，适用于工具调用。  
- Reflexion：自我审查、反思与重写回答。

**结构化输出**

- 使用 JSON 模式/Schema/函数调用（Function Calling）或语法约束解码。  
- 给出清晰字段定义和示例，便于下游解析。

**Prompt 优化流程**

- A/B 测试不同 prompt 模板。  
- 建立 prompt 版本管理体系（加版本号和简要变更说明）。  
- 安全：
  - 防提示注入：隔离 system 指令和用户输入。  
  - 对输入进行净化和限制，避免用户覆盖内部策略。

---

### 3.2 RAG（检索增强生成）

**基础流程**

1. 数据摄取：从文档/网页/数据库/知识库抽取数据。  
2. 清洗与切分：使用 RecursiveCharacterTextSplitter / 语义切分等。  
3. 生成 embedding：对每个 chunk 生成向量。  
4. 入库：写入向量数据库 + 元数据（来源、时间、权限等）。  
5. 查询时：query → embedding → 检索 → 重排 → 组装 Prompt → LLM 生成。

**图片与表格处理（RAG 中多模态）**

- 图片：
  - OCR + 版面分析提取文本。  
  - 使用多模态模型（CLIP/LLaVA/Qwen-VL）生成视觉 embedding 与 caption。  
  - 双通道入库：文本向量 + 视觉向量。  
  - 检索时文本/视觉并行召回，结果融合再重排。  
  - 生成时必须引用图片来源/文件名/位置，避免虚构图中内容。

- 表格：
  - 表格结构恢复：转为 Markdown / CSV / JSON，保留表头/单位/时间。  
  - 行级/单元格级 embedding，便于精确检索具体数值。  
  - 检索对表头/行标签匹配，重排偏向高匹配度记录。  
  - 生成时可要求输出 Markdown 表，引用原字段名和单位。

- 版面/布局：
  - 使用 Layout 分析（段落/图/表分块），对每个区块 embedding。  
  - 检索时多路召回（段落/表/图）并加权融合。  
  - Prompt 中加入区块描述与引用信息。

**整体优化**

- 检索：
  - 查询改写/扩展，意图分类（FAQ/知识问答/数据查询等）。  
  - Hybrid 检索：稠密向量 + BM25/关键词。  
  - ColBERT、E5-mistral 等新一代检索模型。  
  - 多路召回（不同 chunk 策略/索引）并行。

- 重排：
  - 使用 cross-encoder 或 rerank 模型对 top-k 精排。  

- 多轮对话：
  - 对话摘要 + 关键信息提取入库。  
  - 使用会话 ID 检索相关历史对话片段。

- 幻觉控制：
  - 强制要求引用来源（引用编号、文档名、页码等）。  
  - 降低温度，结构化输出后再校验。  

**评估**

- 检索指标：P@k / Recall@k / MRR / NDCG。  
- 生成指标：事实性、相关性、流畅度，结合 LLM-as-judge 和人工标注。  
- 端到端：A/B 测试，监控覆盖率/幻觉率/延迟/成本。

---

### 3.3 微调（Fine-tuning）

**场景选择**

- 适合微调：
  - 领域知识强且相对稳定。  
  - 输出风格/格式高度统一（如合同、病历、报告）。  
  - 低延迟需求（用小模型离线微调替代大模型在线推理）。  

- 更适合 RAG：
  - 知识变动频繁（政策、产品、FAQ）。  
  - 需要可追溯性和解释性（引用文档来源）。

**PEFT：LoRA vs QLoRA**

- LoRA：
  - 在全精度基座上增加低秩矩阵，训练少量参数。  
  - 显存占用中等，实现简单。  
- QLoRA：
  - 先 4bit 量化基座，再加 LoRA。  
  - 显存最省，适合单机/单卡，流程略复杂。  
- 选择建议：
  - 显存紧张 → QLoRA；显存充足且想简化流程 → LoRA。

**微调实践**

- 数据：
  - 清洗去重，指令-回答格式统一。  
  - 多样化任务/风格，防止过度模板化。  
- 配置：
  - rank、α、dropout、学习率、batch size、梯度累积。  
- 防灾难遗忘：
  - 混入部分通用数据或使用正则化。  
- 评估：
  - 专门领域测试集 + 人工评估。  
  - 监控 train/valid gap 与行为漂移。

---

### 3.4 Agent 开发

**基础模式**

- ReAct：交替「思考（Thought）」和「行动（Action）」，结合工具调用进行推理。  
- 计划-执行-反思：
  - Planner：拆解任务为若干子任务。  
  - Executor：按计划调用工具。  
  - Reflector/Judge：校验和修正结果。

**多 Agent**

- 角色设计：Planner / Executors（各工种）/ Judge / Router / Critic。  
- 消息路由：使用消息队列或统一总线，将任务派发到对应 Agent。  
- 健壮性：
  - 幂等键、超时、重试、熔断。  
- 冲突解决：
  - 多 Agent 输出冲突时，采用投票/规则/优先级等策略。  
- 降级兜底：
  - 在 Agent 失效或成本过高时，回退到单 Agent 或较简单的流程。

**评估与监控**

- 指标：
  - 成功率、平均调用次数、平均/尾部时延、错误恢复能力、成本。  
- 常见坑：
  - 工具幻觉（调用不存在/不该调用的工具）。  
  - 死循环或无意义反复调用。  
  - 职责边界混乱导致输出不稳定。

---

### 3.5 视频→文本多模态

**典型任务**

- 视频摘要、章节划分、重点片段提取、基于视频的问答。  

**处理流程**

1. 音频 ASR：提取音轨文字（Whisper/Paraformer 等），保留时间戳。  
2. 抽帧/分段：场景切换检测 + 均匀采样。  
3. 视觉编码：使用 CLIP/Video-LLaVA/Qwen-VL-Video 等生成视觉 embedding 与 caption。  
4. 对齐：基于时间戳对齐文本与帧，构建分段级 multimodal 表示。  
5. 检索+重排：对用户问题检索相关分段，使用交叉编码器重排。  
6. 生成：先分段回答/摘要，再层次汇总为整体结果。

**成本优化**

- 先用低分辨率/低帧率粗分析，对重点片段进行细粒度处理。  
- 分层摘要减少长上下文调用。  
- 对 embedding 做聚类/去冗余，减少存储和检索负担。

**评测指标**

- 事实性：与字幕/脚本对比，避免虚构。  
- 覆盖率：是否覆盖关键事件/场景。  
- 延迟与成本：整体 pipeline 性能与费用。  
- 用户满意度：通过实际用户反馈评估效果。

---

## 四、工程实践与架构设计

### 4.1 应用架构

**典型结构**

前端 → API 网关 → 编排/业务服务 → 模型服务（路由/推理/缓存） → 数据层（向量库/关系型 DB/对象存储）。

**RAG 架构**

- 摄取流水线：数据同步→清洗→切分→embedding→入向量库。  
- 查询流水线：query→检索/重排→Prompt 组装→LLM 生成→后处理。  

**可扩展性**

- 无状态服务 + 自动扩缩容（K8s）。  
- 向量库/DB 分片与副本，多区域部署。  
- 使用异步队列处理长任务和批量任务。

**性能优化**

- 多级缓存：Prompt 裁剪缓存、语义缓存、结果缓存。  
- 批处理与连续批处理：提高 GPU 利用率和 QPS。  
- KV Cache：减少重复计算。  
- 模型路由：小模型优先，大模型兜底。

---

### 4.2 部署与运维

**部署方式**

- 直接调用云端 API（OpenAI、Gemini 等）。  
- 自建/托管推理服务（vLLM/TGI/TensorRT-LLM 等）。  
- 容器化（Docker）+ K8s 编排。  
- 蓝绿部署/金丝雀发布。

**成本管理**

- 量化（8/4bit）、批处理、缓存、路由。  
- 用量监控与配额控制，预算告警。  
- 热门问题预计算与缓存。

**监控与可观测性**

- 基础指标：延迟/吞吐/错误率/成本。  
- 资源：GPU/CPU/内存/网络。  
- 依赖：向量库/DB 的 QPS/延迟。  
- Tracing（如 OpenTelemetry）：跟踪一次请求的全链路。  
- SLO：为关键接口设置可用性/延迟目标和相应告警。

**可靠性**

- 断路器模式，避免雪崩。  
- 幂等与重试策略，合理设置超时。  
- 优雅降级：小模型、静态答案、缓存结果、多级兜底。  
- 多活/备份与灾备演练。

---

### 4.3 安全与合规

**数据安全**

- 传输与静态加密（TLS / KMS）。  
- 脱敏处理（PII/敏感字段），遵守最小权限原则。  
- RBAC 与审计日志，记录关键操作。  
- 数据留存与删除策略（保留周期、日志脱敏）。

**模型安全**

- 输入净化（黑白名单、正则）。  
- 防提示注入：区分 system 与 user，避免用户覆盖系统指令。  
- 输出过滤：敏感词、攻击内容、合规风险检测。  
- 越狱与攻击测试：维护越狱测试集，持续迭代。

**合规要求**

- GDPR / CCPA / HIPAA 关注点：  
  - 数据最小化、用途限制。  
  - 用户访问/删除权。  
  - 跨境数据传输合规。  

---

### 4.4 测试与质量

**测试类型**

- 单元测试、集成测试、端到端测试。  
- 非确定性输出测试：规则+统计（正则/Schema 校验/模板相似度）。  
- 性能与压测：并发负载、长尾延迟、退化路径。

**评估与迭代**

- 评估：
  - 自动指标 + LLM-as-judge + 人工评审。  
  - 检索指标与生成指标并重。  
  - 安全/合规性测试。  
- 持续改进：
  - 日志 → 标注 → 增量微调/Prompt 调整。  
  - A/B 测试评估新版本。  
  - MLOps：模型/数据版本化，实验追踪（W&B/MLflow/自建平台）。

---

## 五、场景设计与应用（示例回答框架）

### 5.1 企业知识库 RAG

- 架构：摄取(清洗/切分/embedding/入库向量库+对象存储) + 查询(检索/重排/生成) + 观测(日志/Tracing/指标)。  
- 难点：幻觉、延迟、成本、权限控制。  
- 策略：hybrid 检索、交叉编码重排、引用来源、低温度、缓存和批处理。  
- 监控：覆盖率、幻觉率、延迟、成本；按业务线/用户类型/版本分桶。

### 5.2 复杂工具调用 Agent

- 流程：计划 → 工具调用 → 校验 → 汇总。  
- 设计：
  - 工具 Schema 显式，限制输入/输出格式。  
  - 幂等键、超时策略、失败重试与熔断。  
  - 并行工具调用后统一汇总。  
- 审计：记录每次工具调用的参数与结果，便于排查和安全审计。

### 5.3 代码助手

- 使用代码专用模型（CodeLlama/StarCoder 等）或具备代码能力的通用 LLM。  
- 支持多文件上下文和项目级分析。  
- 结构化输出：JSON 或函数调用方式返回修改建议。  
- 安全：过滤高危操作（rm -rf、删除数据库等），避免泄露敏感代码。  
- 功能：代码生成、重构建议、单元测试生成、错误解释等。

### 5.4 视频/多模态问答

- 流程：ASR + 抽帧分段 → caption/embedding → 检索 + 重排 → 层次摘要。  
- 工程要点：时间戳对齐、多模态索引、成本分层（粗+细）、缓存策略。  

### 5.5 垂直合规（医疗/金融/法律）

- 数据脱敏与权限控制。  
- 专业知识库与基准评测。  
- 高风险输出必须人工审核或走审计流程。  
- 对输出做合规过滤与解释。

---

## 六、面试高频问题 FAQ（20 问，详实示例）

> 建议在准备面试时选取其中 5–8 题，结合自己项目讲「故事 + 指标 + 取舍」。

**Q1: 如何设计一个企业级 RAG 系统？**  
**Q2: 如何评估 RAG 的效果？**  
**Q3: 如何进行大模型选型与多模型路由设计？**  
**Q4: 如何从工程角度优化大模型调用成本？**  
**Q5: 如何降低大模型幻觉？**  
**Q6: 微调 vs RAG，如何选择？**  
**Q7: 如何设计一个可扩展的多 Agent 系统？**  
**Q8: 数十亿文档检索系统如何设计？**  
**Q9: 大模型服务突然变慢时如何排查？**  
**Q10: 如何提升 RAG 检索相关性？**  
**Q11: LoRA 和 QLoRA 的区别与应用场景？**  
**Q12: Self-Attention 为什么需要除以 $\sqrt{d_k}$？**  
**Q13: 如何选择合适的向量索引结构？**  
**Q14: 多轮对话的上下文如何管理？**  
**Q15: Function Calling 的原理与工程要点？**  
**Q16: Function Calling 进阶设计（错误处理 / 安全 / 并行）？**  
**Q17: 如何设计视频→文本的问答/摘要系统？**  
**Q18: MCP 是什么，有何价值？**  
**Q19: A2A 多助手协同如何设计？**  
**Q20: RAG 中如何处理图片/表格等非纯文本数据？**

> 详细答案可以保留你原始版本中的长回答，这里不全部展开，面试时按场景剪裁。

---

## 七、面试准备建议

- 技术：  
  - Python / 常用深度学习框架。  
  - 能从零实现一个最小 RAG Demo（切分→embedding→检索→重排→生成）。  
  - 能做一次 LoRA/QLoRA 微调实验。  
  - 理解 vLLM/TensorRT-LLM 部署与压测要点。  

- 项目讲述：  
  - 准备 2–3 个案例，每个从目标、数据、架构、难点、指标、成本/安全/合规、迭代来讲。  

- 算法与代码：  
  - 适度刷 LeetCode 中等题，巩固数据结构/算法思维。  

- 实战演练：  
  - Prompt 改写与 A/B 测试案例。  
  - 「日志→标注→迭代」的闭环优化故事。  

---

## 八、学习资源

- 文档：Hugging Face、LangChain、LlamaIndex、OpenAI、Pinecone、Milvus/Qdrant 官方文档。  
- 论文：
  - Transformer（Attention Is All You Need）  
  - GPT-3 / InstructGPT  
  - LoRA（Low-Rank Adaptation）  
  - RAG（Retrieval-Augmented Generation）等。  
- 开源项目：
  - LangChain、LlamaIndex、AutoGen、CrewAI  
  - vLLM、TGI、TensorRT-LLM  
  - Milvus/Qdrant/Weaviate/pgvector 等向量库

---

## 九、快速复盘清单

- 能否画出一个支持图片/表格的 RAG 全链路，并指出性能与幻觉优化点？  
- 能否清楚讲明 LoRA/QLoRA 的差异与资源收益？  
- 能否设计完整的监控 + 告警 + 降级策略？  
- 能否用实例说明如何降成本（路由/量化/缓存/批处理）及其权衡？  
- 能否在 5 分钟内给出最小 RAG / Agent / Function Calling 实现思路？  
- 能否说明图片/表格/视频在 RAG 系统中的处理与提示设计要点？  
- 是否准备好了可运行的最小代码片段，便于现场演示或白板讲解？

---

## 十、5 年后端工程师转大模型工程应用开发：重点补充指南

这一节专门面向「有 3–5 年后端开发经验，正在转向 LLM 应用工程」的同学，强调 **从已有后端技能到 LLM 工程栈的映射与升级路径**，以及 2024 年以后的工业界实践。

### 10.1 技能映射：从传统后端到 LLM 工程

| 传统后端能力                    | LLM 工程中的映射                                         |
| ------------------------------- | -------------------------------------------------------- |
| HTTP API / REST / gRPC          | 模型服务 API、RAG 查询接口、Function Calling 网关        |
| SQL/NoSQL + 缓存                | 向量数据库（Milvus/Qdrant/pgvector）、文档存储、特征缓存 |
| MQ / 任务队列（Kafka/RabbitMQ） | 多 Agent 消息通道、异步管线（批量 embedding、索引构建）  |
| 服务治理（熔断/限流/降级）      | LLM 服务路由、API 限流、多模型降级与兜底                 |
| 监控与日志（Prom/Grafana）      | LLM Observability：请求日志、Tracing、评估指标           |
| CI/CD & DevOps                  | Prompt/模型/索引版本管理，灰度发布，回滚与实验对照       |

关键是：**先把「大模型」当成一种新型依赖服务，然后再逐步深入模型/检索/评估层**。

---

### 10.2 2024+ LLM 工程技术栈更新

**开源模型路线**

- 新一代主流开源模型：
  - LLaMA 3 系列  
  - Qwen2 / Qwen2.5  
  - Mistral / Mixtral  
  - DeepSeek 等  
- 关注：
  - 长上下文、Function Calling、多模态支持情况。  
  - 社区 Ecosystem：有无现成 LoRA、推理引擎适配、评估脚本。

**推理引擎与性能**

- vLLM：
  - PagedAttention，极大提升多会话场景吞吐。  
  - 适合后端多租户服务，支持 HTTP/gRPC。  
- TensorRT-LLM：
  - 利用 TensorRT 做 kernel fusion 和量化（INT8/FP8），适合低延迟、高并发。  
- TGI / Triton / 各云厂商引擎：
  - 理解调用模型、调参和限流方式即可。

---

### 10.3 RAG 与向量数据库的新趋势（后端视角）

**工程挑战**

- 数据管道：多源数据同步、清洗、切分、embedding、入库。  
- 索引维护：增量更新、在线重建对服务的影响。  
- 性能 & 成本评估：QPS、延迟 P95/P99、存储与检索成本。

**新趋势**

- Hybrid & Learned Sparse：
  - 稠密向量 + 稀疏向量（BM25 / learned sparse）组合。  
  - 使用如 ColBERT/E5-mistral 等模型提升检索质量。

- 向量数据库：
  - pgvector：在 PostgreSQL 中内嵌向量支持，适合已有 PG 架构。  
  - Milvus / Zilliz：云原生分布式方案。  
  - Qdrant：托管服务+原生 Rust 实现，性能好。  

- 数据治理：
  - embedding 覆盖率与质量监控。  
  - 语料版本化 + 回溯分析。  
  - RAG pipeline 的端到端效果评估。

---

### 10.4 多模态：图像 / 视频 / 结构化数据的工程模式

**图像/文档（含表格）**

- OCR + 布局分析 → 文本/表格结构。  
- 多模态模型（Qwen-VL/GPT-4o/LLaVA）做高层语义理解。  
- 典型工程模式：  
  - 离线索引：文档 → OCR/解析 → 分块 → embedding → 入库。  
  - 在线：上传即解析+多模态理解 → 返回结构化结果或进入 RAG 流。

**视频**

- 音频抽取 + ASR → 文本索引。  
- 抽帧 → 关键帧/片段 embedding → 视频检索。  
- 结合时间戳对齐，提供「按时间片段」的问答或摘要。

**结构化数据**

- NL → SQL / DSL，由 LLM 生成查询语句，后端执行。  
- 强校验：字段白名单、limit 限制、SQL 注入防护。  
- 执行结果再用 LLM 做自然语言解释。

---

### 10.5 MCP / 多 Agent / A2A 的后端视角

**工具/资源层**

- 把 DB 查询、HTTP 调用、内部 RPC、计算任务封装成 MCP 工具。  
- 规范输入/输出 Schema，类似于 gRPC proto 或 OpenAPI。  

**Agent 层**

- 每个 Agent 就是一个「工作流节点」，可以挂在不同服务上。  
- 用 MQ（Kafka/RabbitMQ/Redis Streams）连接各 Agent，形成事件驱动系统。  

**鲁棒性与治理**

- 对 Agent 调用同样使用：幂等键、超时、重试、熔断。  
- 冲突解决以业务规则为主，LLM 做辅助建议。  
- 成本与时延控制，必要时降级为单 Agent 或静态策略。

---

### 10.6 LLM Observability：从监控到评估

**请求级 Observability**

- 记录：Prompt / Model / Hyperparams / Latency / Token 用量 / 成本。  
- 用 Trace ID 将 LLM 请求与上游 HTTP / RPC 请求关联。  
- 工具：Langfuse / Phoenix / TruLens / DeepEval / 自建平台。

**质量评估（Evaluation）**

- 评估模型（Judge LLM）自动打标签（正确性/相关性/安全性）。  
- 回归集：针对关键任务维护固定问题集，版本升级前回归测试。  
- 用户反馈闭环：将点赞/差评/修正纳入数据，驱动下一轮迭代。

**安全与合规监控**

- 敏感内容与攻击行为检测。  
- 对话日志脱敏，访问控制和审计。  
- 异常行为（频繁攻击、异常调用）触发告警与限流。

---

### 10.7 典型转型案例：从后端项目到 LLM 项目

**案例 1：企业知识库 + 智能客服重构**

- 旧系统：FAQ + 关键词检索 + 规则引擎。  
- 新系统：RAG + LLM + 多轮对话：
  - 架构：API 网关 → 编排服务 → RAG 服务 → 模型服务 → 向量库/文档存储。  
  - 指标：解决率、转人工率、响应时间、成本。  
  - 亮点：新增对话记忆、多模态支持（票据/截图）、更好的召回率和满意度。

**案例 2：内部 BI 报表 → 自然语言问数**

- 旧：固定报表 + 手写 SQL。  
- 新：NL → SQL / DSL：
  - LLM 将问题翻译为 SQL（或某种 DSL），后端执行。  
  - 对 SQL 做严格安全校验。  
  - 执行结果再由 LLM 解释为自然语言或图表说明。  
- 指标：业务同学自助分析比例、响应时间、报表开发人力节省。

---

### 10.8 建议学习路径（针对 3–5 年后端）

1. **第 1 阶段（1–2 个月）：把 LLM 当外部 API 集成**
   - 学会用熟悉的后端框架调用 OpenAI / Qwen / Claude 等。  
   - 实现 Chat API + Function Calling。  
   - 实现一个最小 RAG：向量库可用本地 FAISS / Chroma 或托管服务。

2. **第 2 阶段（2–3 个月）：私有化部署 + RAG 生产级**
   - 用 vLLM 或 TGI 部署一个开源模型（Qwen2 / LLaMA3 等）。  
   - 搭建向量库（Qdrant/Milvus/pgvector），做完整流水线：摄取→检索→生成。  
   - 引入 Observability 工具，对请求日志、评估指标做可视化。

3. **第 3 阶段（3–6 个月）：多模态 / 多 Agent / 微调**
   - 接入图像或文档多模态场景。  
   - 实现一个 Planner + Executor 样式多 Agent 流程，使用 MQ 做编排。  
   - 用 LoRA/QLoRA 做一次小规模微调，并将新模型接入灰度测试流量。

做到这一步，你不仅能写「大模型应用」，还具备**从架构、工程、评估到优化的全链路能力**，与多数 AI 应用开发岗位要求基本对齐。

---

## 十一、代码示例（示意）

> 注意：以下代码偏「骨架示例」，实际工程中需增加日志、监控、重试、鉴权等。

### 示例 1：最小 RAG（文本）

```python
# pip install langchain-openai langchain-community chromadb tiktoken
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate

docs = [Document(page_content=open("policy.txt").read())]
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(docs)

emb = OpenAIEmbeddings(model="text-embedding-3-large")
vectordb = Chroma.from_documents(chunks, emb)
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

template = """你是企业知识库助手，请基于检索到的内容回答。
问题：{question}
检索内容：
{context}
要求：引用来源，若无依据请回答“未找到依据”。
"""
prompt = PromptTemplate.from_template(template)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def answer(q):
    docs = retriever.get_relevant_documents(q)
    context = "\n\n".join([f"[{i}] {d.page_content[:400]}" for i, d in enumerate(docs)])
    return llm.invoke(prompt.format(question=q, context=context)).content

print(answer("我们的差旅报销标准是什么？"))
```

---

### 示例 2：Function Calling（工具调用 + 参数校验）

```python
# pip install openai
from openai import OpenAI
client = OpenAI()

tools = [{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "按城市查询天气",
    "parameters": {
      "type": "object",
      "properties": {"city": {"type": "string"}},
      "required": ["city"]
    },
  },
}]

def get_weather(city):
    # 实际工程中这里会调用你的天气 API，而不是写死数据
    return {"city": city, "temp_c": 26, "condition": "sunny"}

messages = [{"role":"user","content":"北京天气怎样？"}]
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)
msg = resp.choices[0].message
if msg.tool_calls:
    for tc in msg.tool_calls:
        if tc.function.name == "get_weather":
            args = eval(tc.function.arguments)  # 注意：生产中不要直接 eval，需用 json.loads
            result = get_weather(**args)
            messages.append(msg)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})
final = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
print(final.choices[0].message.content)
```

---

### 示例 3：RAG 处理图片（OCR + 视觉双通道，伪代码）

```python
# 需替换为实际 OCR、CLIP、向量库实现
from some_ocr import ocr_extract_text
from some_clip import clip_embed
from some_vectordb import VectorDB
from langchain_openai import OpenAIEmbeddings

db = VectorDB()
emb = OpenAIEmbeddings(model="text-embedding-3-large")

img_path = "invoice.jpg"
text = ocr_extract_text(img_path)
text_vec = emb.embed_query(text)
vis_vec = clip_embed(img_path)

db.upsert([
  {"id": "img1-text", "vec": text_vec,
   "meta": {"source": img_path, "ocr": text, "mod": "text"}},
  {"id": "img1-vis", "vec": vis_vec,
   "meta": {"source": img_path, "mod": "vision"}}
])

# 查询：文本 query → 文本向量检索 text 通道 + CLIP 编码后检索 vision 通道 → 融合重排 → 生成时要求引用来源
```

---

### 示例 4：RAG 处理表格（行级 chunk）

```python
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

df = pd.read_csv("finance.csv")
rows = []
metas = []

for idx, row in df.iterrows():
    md = "| " + " | ".join([f"{col}: {row[col]}" for col in df.columns]) + " |"
    rows.append(md)
    metas.append({"type": "table_row", "row_index": int(idx)})

emb = OpenAIEmbeddings(model="text-embedding-3-large")
vectordb = Chroma.from_texts(rows, emb, metadatas=metas)

# 检索后提示模型：引用表头/数值/单位/时间，不编造；必要时输出 Markdown 表
```

---

### 示例 5：视频→文本（ASR + 抽帧 + 检索，伪代码）

```python
from some_asr import transcribe_audio         # 返回 [(text, start, end)]
from some_vcap import extract_keyframes       # 返回 [(frame_img, timestamp)]
from some_embed import embed_text, embed_image
from some_vectordb import VectorDB

audio_segments = transcribe_audio("talk.mp4")
frames = extract_keyframes("talk.mp4", fps=0.5)

db = VectorDB()
for i, (text, ts0, ts1) in enumerate(audio_segments):
    db.upsert([{
        "id": f"asr-{i}",
        "vec": embed_text(text),
        "meta": {"t0": ts0, "t1": ts1, "text": text, "mod": "asr"}
    }])

for j, (img, ts) in enumerate(frames):
    db.upsert([{
        "id": f"frm-{j}",
        "vec": embed_image(img),
        "meta": {"t": ts, "mod": "frame"}
    }])

# 查询：文本 query → 检索 ASR 文本段 + 相关图像帧 → 融合重排 → 生成时间戳对齐的回答/摘要
```

---

### 示例 6：多 Agent / A2A 协同（简化）

```python
from openai import OpenAI
client = OpenAI()

def call_llm(msgs, model="gpt-4o-mini", **kw):
    return client.chat.completions.create(
        model=model, messages=msgs, **kw
    ).choices[0].message

user_query = "请给我北京三日游行程，含美食与博物馆。"

# Planner 负责拆解任务
planner = call_llm([
    {"role": "system", "content": "你是规划者，负责分解用户需求成若干步骤。"},
    {"role": "user", "content": user_query}
]).content

# Executor 负责生成草案
draft = call_llm([
    {"role": "system", "content": "你是执行者，按规划给出详细行程草案。"},
    {"role": "user", "content": planner}
]).content

# Judge 负责审查与优化
final = call_llm([
    {"role": "system", "content": "你是裁决者，检查完整性、可行性与预算合理性，并改进草案。"},
    {"role": "user", "content": draft}
]).content

print(final)
```

---

### 示例 7：MCP 工具声明（概念示例）

```yaml
tools:
  - name: search_docs
    description: "在企业文档中搜索段落"
    input_schema:
      type: object
      properties:
        query: { type: string }
        top_k: { type: integer, default: 5 }
      required: [query]
    auth: bearer

  - name: get_kb_entry
    description: "按 ID 获取知识库条目"
    input_schema:
      type: object
      properties:
        id: { type: string }
      required: [id]
    auth: bearer
```

---

> 使用建议：  
> - 面试前可以先通读一遍文档，划出自己不熟的部分重点补课。  
> - 准备 2–3 个项目故事，对照本清单补充各个维度（目标/方案/实现/指标/迭代）。  
> - 挑选几段代码示例熟悉，必要时可现场在白板/IDE 中快速写出一个简化版本 Demo。
