# 企业知识库 Advanced RAG 智能问答系统

面向企业内部资料，基于 Vue 3 + FastAPI 的检索增强问答平台：登录后按知识库隔离检索，以 SSE 流式输出带来源的答案，检索、融合、重排、保存、评估由确定性代码编排，LLM 只负责理解与生成。

[![pytest](https://github.com/baiye-banned/advanced-rag-qa/actions/workflows/python-tests.yml/badge.svg?branch=develop)](https://github.com/baiye-banned/advanced-rag-qa/actions/workflows/python-tests.yml)
[![node --test](https://github.com/baiye-banned/advanced-rag-qa/actions/workflows/frontend-tests.yml/badge.svg?branch=develop)](https://github.com/baiye-banned/advanced-rag-qa/actions/workflows/frontend-tests.yml)
[![vite build](https://github.com/baiye-banned/advanced-rag-qa/actions/workflows/build.yml/badge.svg?branch=develop)](https://github.com/baiye-banned/advanced-rag-qa/actions/workflows/build.yml)
[![static checks](https://github.com/baiye-banned/advanced-rag-qa/actions/workflows/static-checks.yml/badge.svg?branch=develop)](https://github.com/baiye-banned/advanced-rag-qa/actions/workflows/static-checks.yml)
[![e2e](https://github.com/baiye-banned/advanced-rag-qa/actions/workflows/e2e.yml/badge.svg?branch=develop)](https://github.com/baiye-banned/advanced-rag-qa/actions/workflows/e2e.yml)

## 目录

- [亮点](#亮点)
- [快速开始](#快速开始)
- [配置](#配置)
- [架构](#架构)
- [AI 调用与 Prompt](#ai-调用与-prompt)
- [测试](#测试)
- [部署](#部署)
- [技术栈](#技术栈)
- [目录结构](#目录结构)
- [CI](#ci)
- [协作与提交规范](#协作与提交规范)
- [分支与发布](#分支与发布)
- [安全说明](#安全说明)
- [文档](#文档)
- [License](#license)

## 亮点

- 支持登录、JWT 鉴权、用户资料、头像上传和路由守卫。
- 支持知识库创建、重命名、删除、切换、默认知识库兜底和文件管理。
- 支持文本、DOCX、PDF 上传解析、语义分块、Milvus Lite 向量索引和上传失败回滚。
- 隔离多知识库数据：以 `knowledge_base_id` 同时约束 MySQL 元数据和 Milvus 向量记录。
- 提供 `/api/chat/stream` SSE 流式问答，支持历史会话、Markdown 渲染、来源展示和 Trace 回放。
- 使用 Advanced RAG 检索链路，覆盖 RAG 路由判断、查询规划、HyDE、问题改写、关键词召回、向量召回、RRF 融合、rerank 和兜底检索。
- 支持图片附件问答，将视觉模型生成的图片描述与用户文本合并为 `effective_question`。
- 设计多轮会话记忆机制，包含短期滑窗、长期摘要、近期上下文压缩和长期摘要二次压缩。
- 在 assistant 消息保存后异步执行 RAGAS 在线评估，计算 faithfulness、response relevancy 和 context precision。
- 补充后端 RAG 检索链路和前端流式解析工具函数的回归测试。

## 快速开始

### 环境要求

- Node.js 22.15+（`npm test` 用到该版本起才提供的 `module.registerHooks`，见 [tests/README.md](tests/README.md)）
- Python 3.10+
- MySQL 8+
- Milvus Lite 本地文件或远程 Milvus 服务
- 可用的 DeepSeek / DashScope API Key
- 可选：阿里云 OSS，用于图片附件存储

浏览器基线（前端样式依赖）：Safari 16.4+ / Chrome 111+ / Firefox 128+。Tailwind CSS v4 依赖 `@property` 与 `color-mix()` 等较新的 CSS 特性，低于上述基线的浏览器上样式会整体失效（不是降级，是不生效）；旧浏览器需要继续支持时，前端需回退到 Tailwind 3.4.x。

### 安装依赖

```bash
npm install

cd backend
pip install -r requirements.txt
```

### 配置 `.env`

在仓库根目录复制示例配置：

```bash
cp .env.example .env
```

Windows PowerShell：

```powershell
Copy-Item .env.example .env
```

至少需要配置：

```env
MYSQL_USER=your_mysql_user
MYSQL_PASSWORD=your_mysql_password
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=rag_system
MYSQL_SSL_MODE=
MYSQL_SSL_CA=

DEEPSEEK_API_KEY=your_deepseek_api_key
DASHSCOPE_API_KEY=your_dashscope_api_key

SECRET_KEY=change-this-secret-key-in-production
```

`SECRET_KEY` 留空或仍保留 `.env.example` 中的公开占位值 `change-this-secret-key-in-production` 时，后端会拒绝启动；生成真实值：

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

`MYSQL_PASSWORD` 同样不能留空，也不能使用代码内置的公开占位值 `change-me`，否则启动阶段会拒绝连接。生成数据库口令并把同一口令设置到 MySQL 对应账号：

```bash
python -c "import secrets; print(secrets.token_urlsafe(24))"
```

仅本地开发可以分别用 `ALLOW_INSECURE_DEFAULT_SECRET=true`、`ALLOW_INSECURE_DEFAULT_MYSQL_PASSWORD=true` 临时放行，禁止用于任何他人可访问的部署。

默认账号播种由 `SEED_DEFAULT_USERS` / `SEED_ADMIN_PASSWORD` / `SEED_DEMO_PASSWORD` 控制：默认开启，只为尚不存在的 `admin`、`demo` 创建账号，不覆盖已有口令；未配置对应口令时会生成不可预测且不写入日志的随机口令，账号因此无法直接登录。重置方式、关闭播种与完整配置说明见 [部署指南的生产配置](docs/DEPLOYMENT.md#生产配置)。

### 初始化数据库

创建 MySQL 数据库：

```sql
CREATE DATABASE IF NOT EXISTS rag_system
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;
```

启动 FastAPI 时，`backend/main.py` 会调用 `init_db()` 创建表结构，并调用 `seed_default_users()` 初始化默认账号（播种规则见上一节）。

知识库与知识文件按归属用户隔离，历史数据（`user_id` 为 NULL）对任何用户都不可见，需要一次性回填：

```bash
python scripts/backfill_knowledge_owner.py --user-id <用户ID>            # 默认只打印回填计划
python scripts/backfill_knowledge_owner.py --user-id <用户ID> --apply    # 真正写入
```

### 启动后端

```bash
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8002
```

### 启动前端

```bash
npm run dev
```

Vite 默认地址：

```text
http://localhost:5173
```

### 健康检查

后端启动后检查根路径 `/health`（不在 `/api/` 前缀下）：

```bash
curl -i http://127.0.0.1:8002/health
```

期望返回 `200` 与 `{"status":"ok"}`。注意 `/api/health` 并不存在，请求它只会得到 `404`。

## 配置

完整配置以 [.env.example](.env.example) 为唯一事实来源；后端从仓库根目录读取 `.env` 与 `.env.development`。README 只列出首次启动真正要改的变量，长配置说明、SSL、远程 Milvus、离线缓存与账号播种细节见 [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md#生产配置)。

| 变量 | 用途 | 说明 |
| --- | --- | --- |
| `MYSQL_USER` / `MYSQL_PASSWORD` / `MYSQL_HOST` / `MYSQL_PORT` / `MYSQL_DATABASE` | MySQL 连接 | `MYSQL_PASSWORD` 必填；空值或代码内置占位值 `change-me` 会拒绝启动 |
| `MYSQL_SSL_MODE` / `MYSQL_SSL_CA` | MySQL TLS | 云数据库要求 SSL 时设置，详见部署指南 |
| `DEEPSEEK_API_KEY` | 回答、查询规划、记忆、重排兜底、RAGAS 的默认上游 | 未配置阶段级覆盖时继承到这里 |
| `DASHSCOPE_API_KEY` | Embedding、rerank、视觉理解、文本兜底 | 上述分项未单独配置密钥时回退到该值 |
| `SECRET_KEY` | JWT 签名 | 必填；空值或公开占位值 `change-this-secret-key-in-production` 会拒绝启动 |
| `MILVUS_LITE_URI` / `MILVUS_URI` | 本地或远程向量库 | 默认使用本地 `./milvus.db`；远程地址与凭据见部署指南 |
| `SEED_DEFAULT_USERS` / `SEED_ADMIN_PASSWORD` / `SEED_DEMO_PASSWORD` | 启动账号播种 | 默认只为缺失账号建号，不覆盖已有口令 |
| `RAGAS_ENABLED` | 异步在线评估开关 | 默认关闭；启用前需安装可选依赖 |

模型供应商：

- DeepSeek 走 OpenAI-compatible API，默认负责回答生成、查询规划、记忆、重排 LLM 兜底与 RAGAS。
- DashScope 负责 embedding、rerank、视觉理解和文本兜底；`DASHSCOPE_API_KEY` 可作为多条能力链路的公共密钥。
- 阿里云 OSS 仅图片附件需要；RAGAS 是可选评估能力，默认不安装依赖。

分阶段模型覆盖（留空即继承）：

| 阶段 | 覆盖变量 | 留空时继承 |
| --- | --- | --- |
| 路由 | `ROUTER_BASE_URL` / `ROUTER_API_KEY` / `ROUTER_MODEL` | `TEXT_FALLBACK_*` |
| 查询规划 | `PLANNER_BASE_URL` / `PLANNER_API_KEY` / `PLANNER_MODEL` | `DEEPSEEK_*` |
| 最终回答 | `ANSWER_BASE_URL` / `ANSWER_API_KEY` / `ANSWER_MODEL` | `DEEPSEEK_*` |
| 记忆摘要 | `MEMORY_BASE_URL` / `MEMORY_API_KEY` / `MEMORY_MODEL` | `DEEPSEEK_*` |
| 重排 LLM 兜底 | `RERANK_LLM_BASE_URL` / `RERANK_LLM_API_KEY` / `RERANK_LLM_MODEL` | `DEEPSEEK_*` |
| RAGAS | `RAGAS_BASE_URL` / `RAGAS_API_KEY` + `RAGAS_LLM_MODEL` | `DEEPSEEK_*` |

- 阶段级密钥留空时继承供应商密钥；只填阶段 `API_KEY` 时，该阶段的 `BASE_URL` 与 `MODEL` 仍分别继承对应的 `TEXT_FALLBACK_*` 或 `DEEPSEEK_*`。
- 不配置任何阶段变量时，现有 `.env` 的行为不变。
- RAGAS 的地址与密钥继承 `DEEPSEEK_*`，模型继续由 `RAGAS_LLM_MODEL` 控制，不存在 `RAGAS_MODEL` 变量。
- 合并后的 [.env.example](.env.example) 已包含全部阶段变量及继承注释，以该文件为准。

## 架构

### 整体架构

```text
Vue 3 前端
  -> Pinia / Vue Router / Element Plus
  -> Fetch SSE 调用 /api/chat/stream
  -> FastAPI 路由层
  -> JWT 鉴权与用户上下文解析
  -> 会话记忆 / 图片理解 / RAG 路由判断
  -> retrieve_knowledge 检索链路
  -> DeepSeek / DashScope / Milvus / MySQL
  -> SSE 流式返回答案、来源、Trace 和完成事件
```

### 后端分层

```text
backend/
  router/       API 路由挂载，负责把 HTTP 请求转到 service
  service/      业务编排，例如聊天流、知识库上传、删除回滚
  rag/          RAG 核心能力：LLM、Milvus、rerank、记忆、Trace、RAGAS、视觉理解
  database/     SQLAlchemy Session、MySQL 初始化、checkpointer
  model/        ORM 数据模型
  schema/       Pydantic 请求与响应结构
  crud/         数据访问逻辑
```

### 前端分层

```text
src/
  api/          Axios / Fetch API 封装，包含 streamChat
  stores/       Pinia 状态管理
  views/        Chat、Knowledge、Login、Profile 页面
  components/   Markdown、Trace、通用展示组件
  utils/        SSE 解析、文件校验、日期格式化等工具
```

### 数据与检索架构

- MySQL 保存用户、会话、消息、知识库、文件元数据和评估结果。
- Milvus / Milvus Lite 保存文档 chunk 向量，使用 `knowledge_base_id` 做多知识库隔离。
- 上传文件时先做文件校验、文本抽取和语义分块，再写入 MySQL 与 Milvus；失败时执行回滚。
- 删除知识库或文件时会清理对应 SQL 记录和向量数据，尽量保持元数据与向量索引一致。
- 聊天主链路不再经过 Agent / Tool 包装，当前核心检索入口是 `backend/rag/retrieval.py` 中的 `retrieve_knowledge()`。

### RAG 主链路

```text
用户问题
  -> streamChat()
  -> /api/chat/stream
  -> stream_chat()
  -> 解析 JWT 和当前知识库
  -> 构建 effective_question
  -> 构建短期记忆与长期摘要上下文
  -> decide_need_rag 判断是否需要检索
  -> build_query_plan 生成 HyDE、改写问题、关键词
  -> Milvus 向量召回 + 关键词召回
  -> RRF 融合
  -> rerank 重排
  -> 拼接知识库上下文
  -> stream_rag_answer 流式生成答案
  -> 保存 assistant 消息
  -> 异步 RAGAS 评估
```

更完整的模块说明见 [docs/PROJECT_ARCHITECTURE_FULL.md](docs/PROJECT_ARCHITECTURE_FULL.md)。

## AI 调用与 Prompt

### Prompt 角色划分

项目中的 Prompt 按职责拆成几类，而不是把所有要求塞进一个大 Prompt：

| Prompt 类型 | 作用 |
| --- | --- |
| RAG 路由 Prompt | 判断当前问题是否需要进入知识库检索 |
| 查询规划 Prompt | 生成 HyDE 文档、问题改写和关键词 |
| 回答生成 Prompt | 约束模型优先基于知识库上下文回答 |
| 直答 Prompt | 当问题不需要检索时，直接结合会话记忆回答 |
| 视觉理解 Prompt | 将图片附件转成可和文本问题融合的描述 |
| RAGAS 评估 Prompt | 对回答相关性、忠实度、上下文精度做在线评估 |

### 设计思路

这个项目的 Prompt 设计目标不是追求复杂 Agent 行为，而是让模型在企业知识库场景下更稳定：

- 明确模型身份：企业知识库问答助手。
- 明确证据优先级：知识库上下文优先，其次是当前问题，再其次是会话记忆。
- 明确禁止行为：不要编造知识库中没有出现的事实。
- 明确低置信度策略：信息不足时说明不足，而不是强行回答。
- 明确输出结构：路由和查询规划使用 JSON，方便后端解析和兜底。
- 保持链路可控：LLM 负责理解和生成，检索、融合、重排、保存、评估由确定性代码完成。

### 核心 Prompt 示例

回答生成阶段会将问题、会话记忆和知识库上下文分区传入模型：

```text
你是企业知识库智能问答助手。
请优先基于提供的企业知识库上下文回答用户问题。
如果知识库信息不足，请明确说明不足之处，并给出可验证的建议。
回答使用中文，结构清晰，避免编造未在上下文中出现的事实。
信息优先级为：企业知识库上下文 > 当前用户问题 > 会话记忆。
会话记忆只能用于理解指代、延续任务和用户偏好，不能替代知识库事实依据。
```

查询规划阶段要求模型只输出 JSON：

```text
请为用户问题生成：
1. 一段可能出现在企业文档里的假设答案文档 hyde_document；
2. 3 个语义不同但意图一致的检索改写 rewrites；
3. 不超过 8 个中文关键词 keywords。
只输出 JSON，不要输出 Markdown。
```

### AI 调用方式

项目当前主要使用三类 AI 调用：

| 调用类型 | 位置 | 说明 |
| --- | --- | --- |
| SSE 流式生成 | `/api/chat/stream`、`stream_rag_answer()` | 后端逐块读取模型输出，再通过 `text/event-stream` 返回前端 |
| JSON 结构化调用 | `decide_need_rag()`、`build_query_plan()` | 要求模型返回 JSON，用于路由判断和查询规划 |
| Embedding / rerank 调用 | Milvus 检索、DashScope rerank | 将文本转向量并对候选 chunk 重排 |

流式回答链路：

```text
前端 streamChat()
  -> fetch('/api/chat/stream')
  -> readStreamEvents(response.body)
  -> 后端 StreamingResponse(event_stream)
  -> stream_rag_answer()
  -> ChatOpenAI.astream()
  -> 前端边接收边渲染
```

SSE 事件主要包含：

- 普通文本 chunk：模型生成的回答片段。
- `sources`：最终引用的知识库片段。
- `trace`：本轮 RAG、记忆、评估等调试信息。
- `error`：模型、检索或服务异常。
- `[DONE]`：流式回答结束标记。

模型与服务：

- DeepSeek OpenAI-compatible API：主要用于回答生成和部分结构化文本调用。
- DashScope embedding：用于文档向量化和问题向量化。
- DashScope rerank：用于对候选 chunk 做相关性重排。
- DashScope / OpenAI-compatible vision：用于图片附件理解。
- RAGAS：用于 assistant 消息保存后的异步质量评估。

## 测试

| 目标 | 命令 |
| --- | --- |
| Node 测试 | `npm test` |
| Python 测试 | `pip install pytest`，然后 `python -m pytest -q tests` |
| 前端构建 | `npm run build` |

`pytest` 不在 `backend/requirements.txt` 里（该文件只列线上运行时依赖），测试工具要单独安装。RAGAS 是可选评估能力，默认部署不安装，启用时执行 `cd backend && pip install -r requirements-ragas.txt`；`Chat` chunk 体积提示只是 bundle size 提醒，不代表构建失败。

完整的四终端 Playwright 端到端验收、模型桩、产物处理与 CI/本地差异见 [docs/TESTING.md](docs/TESTING.md)。

## 部署

生产构建前端：

```bash
npm run build
```

构建产物位于 `dist/`。后端可使用 Uvicorn / Gunicorn + Uvicorn Worker 运行，例如：

```bash
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8002
```

Nginx 托管 `dist/`，并分别把 `/api/`、`/uploads/`、`/health` 反代到后端。三个必须注意的点：

- `proxy_buffering off` 对 SSE 很重要，否则流式回答会被 Nginx 缓冲，前端无法实时显示。
- 健康检查必须单独代理：后端是根路径 `/health`，不在 `/api/` 下。省略 `location = /health` 会让请求落进前端 `try_files`，返回 `200` + `text/html`；`/api/health` 不存在，只会得到 `404`。
- `/docs`、`/redoc`、`/openapi.json` 默认故意不代理，避免对外暴露完整接口结构；确需访问时补精确匹配 location，并配合 IP 白名单或额外鉴权。

`VITE_API_BASE_URL` 必须与 Nginx 代理路径一致（同源部署保持 `/api`）。完整 Nginx 配置、DNS 记录、HTTPS 与部署检查清单见 [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)。

## 技术栈

| 层级 | 技术 |
| --- | --- |
| 前端 | Vue 3、Vite、Element Plus、Pinia、Vue Router、TailwindCSS、Axios、Fetch streaming |
| 后端 | FastAPI、SQLAlchemy、MySQL、python-multipart、httpx |
| RAG / Retrieval | 查询规划、HyDE、多路召回、RRF 融合、rerank |
| 向量库 | Milvus Lite / Milvus |
| 模型能力 | DeepSeek OpenAI-compatible API、DashScope embedding / rerank / vision |
| 评估 | RAGAS |
| 对象存储 | 阿里云 OSS |

## 目录结构

```text
.
├── src/                         # Vue 前端
│   ├── api/                     # Axios / Fetch API 封装
│   ├── stores/                  # Pinia 状态管理
│   ├── utils/                   # 前端工具函数和流式解析
│   ├── views/                   # Chat、Knowledge、Login、Profile
│   └── components/              # Markdown、Trace 等组件
├── backend/                     # FastAPI 后端
│   ├── crud/                    # SQLAlchemy 数据访问
│   ├── database/                # DB session 和 checkpointer
│   ├── model/                   # SQLAlchemy 模型
│   ├── rag/                     # Milvus、LLM、记忆、Trace、RAGAS、视觉、检索
│   ├── router/                  # API 路由挂载
│   ├── schema/                  # Pydantic schema
│   └── service/                 # 业务服务
├── docs/                        # 架构、流程、部署与测试文档
├── tests/                       # Node 和 pytest 回归测试
│   └── e2e/                     # Playwright 浏览器验收（用例 / 夹具 / 模型桩服务）
├── package.json
├── pytest.ini
└── backend/requirements.txt
```

## CI

向 `develop`、`main` 提 PR，以及 push 到 `develop` 时会自动跑下面五个独立检查；PR 上另有 PR 标题校验、PR 描述校验与 Secret Scan 三道门禁。同一分支连续 push 时，上一次还在跑的运行会被自动取消（各 workflow 内的 `concurrency`）。

| Workflow | 检查项 | 内容 |
| --- | --- | --- |
| `.github/workflows/python-tests.yml` | 后端测试 pytest (Python 3.10) | 版本对齐 `runtime.txt`（`python-3.10.11`），`pip install -r backend/requirements.txt` + `pytest`，跑 `python -m pytest -q tests` |
| `.github/workflows/frontend-tests.yml` | 前端测试 node --test (Node 22) | `npm ci` 后跑 `npm test` |
| `.github/workflows/build.yml` | 前端构建 vite build (Node 22) | `npm run build`，产物 `dist/` 上传为 artifact |
| `.github/workflows/static-checks.yml` | 静态检查 (最低档) | 后端 `python -m compileall -q backend tests scripts` + `ruff check --select E4,E7,E9,F backend tests scripts`；前端 `node --check`（仓库根与 `src`、`tests`、`scripts` 下的 `.js`/`.mjs`）+ `npx eslint .` |
| `.github/workflows/e2e.yml` | 端到端验收 Playwright e2e (chromium) | 起 MySQL 8.0 服务容器 + 本地模型桩（`tests/e2e/stub_llm_server.py`，替掉全部模型上游）→ 构建前端并以 `vite preview` 托管 → 真实浏览器跑 `tests/e2e/chat.spec.mjs` → 用桩的请求日志确认检索链路真被走到（`tests/e2e/check_stub_calls.py`）→ 截图与各服务日志上传为 artifact |

说明：

- 静态检查是最低档，拦语法错误、未定义名、未使用导入这类确定性错误，**不是**完整规范；为什么不直接开 `ruff --select E,F` 以及后续怎么加严，写在 `static-checks.yml` 顶部注释里。
- 五个 workflow 都是 `permissions: contents: read`，不配置任何密钥，也不使用 `continue-on-error`：失败就是失败。
- 后端测试只需 `backend/requirements.txt` + `pytest`；RAGAS 等可选评估依赖是延迟导入，CI 不安装。
- e2e 不需要任何真实密钥：模型上游由仓库内的桩服务顶替，元数据库是一次性容器，`SECRET_KEY` 每次运行现场生成（写进仓库等于提交一个可用密钥）。审批用的占位口令只对本次运行的临时容器有效。

## 协作与提交规范

- 提交信息与 PR 标题遵循 [Conventional Commits](https://www.conventionalcommits.org/)，type 与 scope 取值、合并方式见 [COMMIT_CONVENTION.md](COMMIT_CONVENTION.md)。
- 新建 issue 必须选择模板；PR 描述按 [.github/pull_request_template.md](.github/pull_request_template.md) 逐节填写，必填节为空时 CI 会变红。
- CI 校验 PR 标题、PR 描述必填节与 issue 结构，脚本在 `scripts/` 下，可用 `node scripts/check_pr_body.mjs <文件>` 本地复现同一套规则。
- 日常 PR squash 合并进 `develop`，发布 PR（`develop` → `main`）用 merge commit。

## 分支与发布

- `main` 是发布分支，只接受来自 `develop` 的发布 PR 和紧急热修复；`develop` 是集成分支，日常改动都提到这里。
- 短分支从 `develop` 切出，命名为 `<type>/<topic>-<issue号>`，合入后删除；一律通过 PR 合入，不直接 push 公共分支。
- 合并方式：日常 PR 用 squash；发布 PR 和 `main → develop` 回同步 PR 用 merge commit（否则回同步会反复重现）。
- 发布：手动运行 `Release` 工作流（输入版本号）→ 评审并合并发布 PR 到 `main` → 手动打 tag；合并到 `main` 后 `Sync main into develop` 会自动开回同步 PR。
- 完整约定（分支命名、发布步骤、门禁、已知限制）见 [BRANCHING.md](BRANCHING.md)。

## 安全说明

- `.env`、本地数据库、上传文件、日志、PID 文件、缓存、`node_modules` 和构建产物都应加入 Git 忽略规则。
- `.env.example` 只保留占位配置，不应提交真实 API key、OSS 凭证、数据库密码或 JWT secret。
- 生产环境需要替换默认本地配置，配置 HTTPS / 反向代理，强化 JWT secret 管理，并进行外部模型与对象存储连通性检查。
- `SECRET_KEY` 缺失或仍是 `.env.example` 里的公开占位值 `change-this-secret-key-in-production` 时，后端会在启动阶段直接报错退出，不会带着这个人人可读的值对外提供服务；该值能让任何人伪造包括 `admin` 在内的任意用户登录态，不可用于任何对外可访问的部署。生成真实值：`python -c "import secrets; print(secrets.token_urlsafe(48))"`。
- 仅本地开发可用 `ALLOW_INSECURE_DEFAULT_SECRET=true` 放行启动：此时进程使用一次性随机密钥，每次重启已签发 token 全部失效，多进程/多副本之间互不认可对方签发的 token，禁止用于生产。
- `MYSQL_PASSWORD` 缺失或仍是仓库公开占位值 `change-me` 时，后端同样在启动阶段拒绝启动（`ensure_mysql_password_configured()`）；生成随机口令：`python -c "import secrets; print(secrets.token_urlsafe(24))"`，并把同一口令设置到 MySQL 账号。
- 仅本地开发可用 `ALLOW_INSECURE_DEFAULT_MYSQL_PASSWORD=true` 放行数据库连接：此时进程用仓库公开占位口令连库，任何拿到仓库的人都能用同一口令连上数据库；该开关与 `ALLOW_INSECURE_DEFAULT_SECRET` 相互独立，开启一个不会放宽另一个，禁止用于他人可访问的部署。
- 启动播种的演示账号不再是公开固定口令：账号口令由 `SEED_ADMIN_PASSWORD` / `SEED_DEMO_PASSWORD` 提供，未配置时随机生成且不写入日志；不需要演示账号时设置 `SEED_DEFAULT_USERS=false`。

## 文档

- [docs/PROJECT_ARCHITECTURE_FULL.md](docs/PROJECT_ARCHITECTURE_FULL.md)：完整架构和模块说明。
- [docs/PROJECT_CODE_READING_ROADMAP.md](docs/PROJECT_CODE_READING_ROADMAP.md)：代码阅读路线。
- [docs/PROJECT_FLOW_DIAGRAM.md](docs/PROJECT_FLOW_DIAGRAM.md)：聊天和知识库主流程 Mermaid 图。
- [docs/MAINTENANCE_GOAL_CLOSURE.md](docs/MAINTENANCE_GOAL_CLOSURE.md)：维护收束报告和剩余风险。
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)：生产配置、数据库初始化、构建、Nginx、DNS 与 HTTPS 部署细节。
- [docs/TESTING.md](docs/TESTING.md)：Node / pytest / 构建与四终端 Playwright 端到端验收。
- [DEVELOPING.md](DEVELOPING.md)：开发须知、提交前自检、密钥配置与 CI 说明。

## License

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE)。
