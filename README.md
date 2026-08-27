# 后端 → AI 应用 / Agent 工程师 · 转型与面试手册

> 把后端积累的工程能力，迁移到 AI 应用与 Agent 开发。9 层知识地图 + 100 道算法题 + 13 道系统设计，按面试考察频率组织——免费、无需登录，打开就能读。

## 写给谁

**适合你，如果——**

- 干了 2-5 年后端（Java / Go / Python 都行），正在考虑或刚开始转 AI 应用工程师、Agent 工程师、AI 平台岗
- 面试前 1-3 个月，想系统梳理一遍工程视角的 AI 知识，而不是背八股
- LLM 的概念都听过，但评测、可观测、成本、安全这段生产化的空缺一直没补上

**可能不适合——**

- 纯算法研究员，或已在 AI Infra 一线多年（这些内容对你偏浅）
- 想找"七天速成"的人——手册是拿来读的，不是拿来收藏的

## 这份手册里有什么

- **9 层知识地图（L1-L9）**：每章结构固定——先看考察频率和面试官想听什么，再拆知识点（是什么 → 为什么 → 怎么做），最后是「面试这样答」口述版与章末速查表
- **算法热题 100**：17 组 100 题，每题带可编译的 Go 参考实现和口述模板
- **系统设计 13 题**：后端经典 5 道 + AI 工程化 8 道，按白板面试的讲法组织
- 全站支持中文全文搜索（左上角搜索框），侧栏可直达任意章节

## 知识地图

<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px;margin:20px 0 8px;"><div style="border:1px solid #e4e7ec;border-left:3px solid #2c5fb2;border-radius:8px;padding:12px 14px;background:#f9fafc;"><a href="#/1-基础盘/01_L1_后端基础" style="display:inline-block;border-bottom:none;font-family:var(--mono),monospace;font-size:11px;font-weight:700;letter-spacing:.1em;color:#2c5fb2;margin:0 0 6px;">L1 · 后端基础</a><div style="font-size:13px;line-height:1.65;color:#3b4450;">网络 / DB / Redis / MQ 速复习，并映射到 AI 场景</div></div><div style="border:1px solid #e4e7ec;border-left:3px solid #2c5fb2;border-radius:8px;padding:12px 14px;background:#f9fafc;"><a href="#/2-LLM应用核心/02_L2_LLM应用基础" style="display:inline-block;border-bottom:none;font-family:var(--mono),monospace;font-size:11px;font-weight:700;letter-spacing:.1em;color:#2c5fb2;margin:0 0 6px;">L2 · LLM 应用基础</a><div style="font-size:13px;line-height:1.65;color:#3b4450;">把模型当软件组件：Token、上下文窗口、API 工程化</div></div><div style="border:1px solid #e4e7ec;border-left:3px solid #2c5fb2;border-radius:8px;padding:12px 14px;background:#f9fafc;"><a href="#/2-LLM应用核心/03_L3_Prompt与Context工程" style="display:inline-block;border-bottom:none;font-family:var(--mono),monospace;font-size:11px;font-weight:700;letter-spacing:.1em;color:#2c5fb2;margin:0 0 6px;">L3 · Prompt 与 Context 工程</a><div style="font-size:13px;line-height:1.65;color:#3b4450;">prompt 即程序：结构化输出、上下文工程</div></div><div style="border:1px solid #e4e7ec;border-left:3px solid #2c5fb2;border-radius:8px;padding:12px 14px;background:#f9fafc;"><a href="#/2-LLM应用核心/04_L4_知识库与数据工程" style="display:inline-block;border-bottom:none;font-family:var(--mono),monospace;font-size:11px;font-weight:700;letter-spacing:.1em;color:#2c5fb2;margin:0 0 6px;">L4 · 知识库与数据工程</a><div style="font-size:13px;line-height:1.65;color:#3b4450;">切块、元数据、入库流水线——数据质量决定上限</div></div><div style="border:1px solid #e4e7ec;border-left:3px solid #2c5fb2;border-radius:8px;padding:12px 14px;background:#eef3fa;"><a href="#/2-LLM应用核心/05_L5_RAG检索增强生成" style="display:inline-block;border-bottom:none;font-family:var(--mono),monospace;font-size:11px;font-weight:700;letter-spacing:.1em;color:#1d4486;margin:0 0 6px;">L5 · RAG 检索增强生成</a><div style="font-size:13px;line-height:1.65;color:#3b4450;">面试主战场：混合检索、Rerank、Agentic RAG</div></div><div style="border:1px solid #e4e7ec;border-left:3px solid #2c5fb2;border-radius:8px;padding:12px 14px;background:#f9fafc;"><a href="#/2-LLM应用核心/06_L6_Agent构建与编排" style="display:inline-block;border-bottom:none;font-family:var(--mono),monospace;font-size:11px;font-weight:700;letter-spacing:.1em;color:#2c5fb2;margin:0 0 6px;">L6 · Agent 构建与编排</a><div style="font-size:13px;line-height:1.65;color:#3b4450;">Workflow vs Agent、MCP、多 Agent 协作</div></div><div style="border:1px solid #e4e7ec;border-left:3px solid #2c5fb2;border-radius:8px;padding:12px 14px;background:#f9fafc;"><a href="#/3-生产工程化/07_L7_评测体系" style="display:inline-block;border-bottom:none;font-family:var(--mono),monospace;font-size:11px;font-weight:700;letter-spacing:.1em;color:#2c5fb2;margin:0 0 6px;">L7 · 评测体系</a><div style="font-size:13px;line-height:1.65;color:#3b4450;">Demo 工程师与生产工程师的分水岭</div></div><div style="border:1px solid #e4e7ec;border-left:3px solid #2c5fb2;border-radius:8px;padding:12px 14px;background:#f9fafc;"><a href="#/3-生产工程化/08_L8_生产化_可观测_成本_安全" style="display:inline-block;border-bottom:none;font-family:var(--mono),monospace;font-size:11px;font-weight:700;letter-spacing:.1em;color:#2c5fb2;margin:0 0 6px;">L8 · 生产化</a><div style="font-size:13px;line-height:1.65;color:#3b4450;">上线后的大考：可观测、成本、安全、注入防御</div></div><div style="border:1px solid #e4e7ec;border-left:3px solid #2c5fb2;border-radius:8px;padding:12px 14px;background:#f9fafc;"><a href="#/4-面试冲刺/09_L9_热点与AI系统设计" style="display:inline-block;border-bottom:none;font-family:var(--mono),monospace;font-size:11px;font-weight:700;letter-spacing:.1em;color:#2c5fb2;margin:0 0 6px;">L9 · 热点与 AI 系统设计</a><div style="font-size:13px;line-height:1.65;color:#3b4450;">前沿热点 + 13 道白板压轴 + STAR-L 讲法</div></div></div>

**两条支线，按需取用：**

- 目标公司有算法面 → [算法热题 100 总览](/5-算法热题100/00_热题100总览)（17 组完整目录在侧栏）
- 准备二面三面的白板压轴 → [系统设计 13 题总览](/6-系统设计题/00_设计题总览)

## 怎么读这份手册

1. **先读[总纲](/00_面试知识大纲)，定优先级**——九层不必从头啃到尾。总纲给出了高频考点排序和贯穿全书的主线：后端映射 → 模型即组件 → prompt 即程序 → 数据定上限 → 检索定精度 → 编排定能力 → 评测定可信 → 生产定生死。
2. **主线推 L2 → L3 → L5**：先建立"模型即组件"的视角（[L2](/2-LLM应用核心/02_L2_LLM应用基础)），学会把 prompt 当程序设计（[L3](/2-LLM应用核心/03_L3_Prompt与Context工程)），然后攻坚考察频率最高的 [RAG](/2-LLM应用核心/05_L5_RAG检索增强生成)；[L4 数据工程](/2-LLM应用核心/04_L4_知识库与数据工程)与 [L6 Agent](/2-LLM应用核心/06_L6_Agent构建与编排)顺势串联。
3. **面试前一两周压轴看** [L9](/4-面试冲刺/09_L9_热点与AI系统设计)：热点决定像不像，系统设计决定是不是。再回头把目标岗位相关章节的进阶追问过一遍。

<details>
<summary><strong>算法刷题建议（目标公司有 OJ 轮再看）</strong></summary>
<p>后端转 AI 应用，算法题仍是一二面的硬门槛，但难度通常到 LC 中等偏上即可，不需要啃完《算法导论》。建议按 <a href="#/5-算法热题100/00_热题100总览">热题 100 总览</a> 给出的 4 阶段顺序（基础盘 → 套路升级 → DP 大系 → 查缺补漏），用 4-6 周过一遍 17 组 100 题，每题都带可编译的 Go 参考实现和口述模板。</p>
</details>

## 进阶追问怎么用

每章末尾有一节「进阶篇」，收录高级岗 / 二三面才会被深挖的问题——更长的追问链、更细的 trade-off 对比。第一次读可以跳过，面试前回头再翻。

<details>
<summary><strong>想自己部署一份？本地预览与发布</strong></summary>
<p>本站是纯静态 Docsify 站点，fork 仓库后即可拥有自己的副本：</p>
<p>① 本地预览：仓库根目录执行 <code>python3 -m http.server 3000</code>，浏览器打开 localhost:3000。Docsify 依赖 fetch 加载 markdown，直接双击 index.html 会白屏，必须走本地 HTTP 服务。<br>② 推送到 GitHub（根目录的 .nojekyll 已就位，保证 _sidebar.md 等下划线文件不被忽略）。<br>③ 仓库 Settings → Pages → Source 选分支 + /(root)，保存后一两分钟即上线。</p>
</details>

<details>
<summary><strong>附录：知识库原始参考素材（备查）</strong></summary>
<p><a href="#/sources/1">原始素材 · 阶段叙事版</a> · <a href="#/sources/2">原始素材 · 工程深度版</a> · <a href="#/sources/3">原始素材 · 十层全景版</a></p>
</details>

---

<sub>后端 → AI 应用 / Agent 工程师 · 转型与面试手册 · 2026 版 · 纯静态站点，基于 Docsify</sub>
