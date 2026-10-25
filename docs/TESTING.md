# 测试指南

本文件是测试与验收的完整细节：单元/回归测试、前端构建、端到端验收（含模型桩）以及 CI 与本地运行的差异。最短的日常命令见 [README 测试](../README.md#测试)；用例清单与维护约定见 [`tests/README.md`](../tests/README.md)。

## 测试分层

| 层 | 命令 | 关注点 |
| --- | --- | --- |
| 前端单元/回归 | `npm test` | 流式解析、状态管理、组件挂载、分页、Markdown 清洗、CI 门禁脚本 |
| 后端单元/回归 | `python -m pytest -q tests` | 检索、鉴权、配置一致性、启动守卫、上传、Trace、RAGAS、BM25、评估脚本 |
| 端到端验收 | `npm run test:e2e` | 真实浏览器走完登录、建库、上传、提问、流式回答、引用来源 |
| 静态检查 | 见 [CI](#ci-与本地差异) | 语法编译、ruff、`node --check`、ESLint |

`tests/e2e/*.spec.mjs` 不在 `npm test` 与 pytest 的收集范围内，必须用 Playwright 单独执行。

## 环境要求

- Python 3.10（CI 使用 3.10）。
- Node.js **>= 22.15**。`npm test` 会加载依赖 `module.registerHooks` 的用例：Node 22 需 22.15 及以上，Node 23 需 23.5 及以上，Node 24 及以上不受限；版本过低时整条 `npm test` 会直接以非零码退出。可用下面的命令自检：

  ```bash
  node -e "const [M,m]=process.versions.node.split('.').map(Number); process.exit((M===22&&m>=15)||(M===23&&m>=5)||M>=24?0:1)"
  ```

- 端到端验收额外需要 Playwright 的 chromium。CI 与本地都只安装这一个浏览器，结论才对得上：

  ```bash
  npx playwright install chromium
  ```

## 前端测试

在仓库根目录执行：

```bash
npm test
```

该命令等价于 `node --test "tests/**/*.test.js"`，覆盖前端流式响应解析、会话与知识库状态、已挂载组件、分页、Markdown 安全渲染、CI 门禁脚本等。用例文件名统一为 `*.test.js`。

## 后端测试

`pytest` 不在 `backend/requirements.txt` 中，首次运行前单独安装：

```bash
pip install pytest
python -m pytest -q tests
```

测试发现规则写在 `pytest.ini`：`testpaths = tests`，`python_files = test_*.py`。

以下验收用例完全离线，不需要网络、模型密钥或外部服务：

- `tests/test_milvus_acceptance.py`
- `tests/test_retrieval_acceptance.py`
- `tests/test_provider_smoke.py`（入口为 `scripts/smoke_providers.py`，只有显式传 `--live` 才会访问真实服务商）

RAGAS 相关测试依赖可选包。为缩短 Render 构建时间，`backend/requirements.txt` 默认不安装 RAGAS；需要运行这部分测试时：

```bash
cd backend
pip install -r requirements-ragas.txt
```

未安装可选包时，RAGAS 导入兼容用例会显式跳过；`RAGAS_ENABLED=false` 时在线评估只写入状态与轨迹事件，不会加载 RAGAS。

测试中可能出现 `Chat` 的 chunk 大小警告，该警告不致命，不影响测试结论。

## 前端构建

```bash
npm run build
```

构建完成后确认 `dist/index.html` 存在。构建日志中的 bundle 体积警告不致命，不会让构建失败；CI 会额外上传 `dist/` 供排查。

## 端到端验收

e2e 打的是「构建产物 + `vite preview`」，不是 dev server。dev server 的 `/api` 代理写在 `vite.config.js` 里，preview 不复用它，因此前端必须在**构建期**把 `VITE_API_BASE_URL` 固化成后端的绝对地址；后端需允许对应来源的跨域请求。

用例串行执行：所有用例共享同一个后端与数据库，并行只会制造假失败。默认单条用例超时 120 秒，断言超时 20 秒。

桩只替模型上游，不替后端检索，也不替前端渲染；但后端路由、检索规划、重排各自的兜底分支仍可能让浏览器用例通过，因此链路是否完整要以桩请求日志为准。

### 四终端完整流程

按顺序准备四个终端。所有命令都在仓库根目录执行，除非命令中另有 `cd`。

**终端 1：模型桩服务。** 日志目录必须先建好：桩启动时会校验日志文件可写，不可写会直接以退出码 2 失败。分片间隔与 CI 保持一致（见 `.github/workflows/e2e.yml`）。

```bash
# 0) 只装 chromium：CI 里也只装这一个浏览器，本地与 CI 的结论才对得上
npx playwright install chromium

# 1) 模型桩服务（日志目录先建好：桩启动时会校验日志文件可写，不可写直接以退出码 2 失败）
#    分片间隔与 CI 保持一致，见 .github/workflows/e2e.yml
mkdir -p tests/e2e/artifacts/logs
python tests/e2e/stub_llm_server.py --port 8899 --dim 1024 \
  --chunk-delay-ms 1200 --chunk-size 8 \
  --log-file tests/e2e/artifacts/logs/stub-requests.jsonl
```

**终端 2：后端。** 上游全部指向桩，MySQL 与向量库使用本地实例。

```bash
# 2) 后端：上游全部指向桩，MySQL / 向量库用本地实例
cd backend
export MYSQL_HOST=127.0.0.1 MYSQL_PORT=<端口> MYSQL_USER=<用户> \
  MYSQL_PASSWORD=<口令> MYSQL_DATABASE=<库名>
export SECRET_KEY="$(openssl rand -hex 32)"
export DEEPSEEK_BASE_URL=http://127.0.0.1:8899/v1 DEEPSEEK_API_KEY=<任意非空占位值>
export TEXT_FALLBACK_BASE_URL=http://127.0.0.1:8899/v1 TEXT_FALLBACK_API_KEY=<同上>
export EMBEDDING_BASE_URL=http://127.0.0.1:8899/v1 EMBEDDING_API_KEY=<同上>
export EMBEDDING_DIM=1024
export RERANK_BASE_URL=http://127.0.0.1:8899/v1/reranks RERANK_API_KEY=<同上>
export VISION_BASE_URL=http://127.0.0.1:8899/v1 VISION_API_KEY=<同上>
export SEED_DEFAULT_USERS=true SEED_DEMO_PASSWORD=<与下面的 E2E_PASSWORD 一致>
python -m uvicorn main:app --host 127.0.0.1 --port 8002
```

**终端 3：前端构建与托管。** 构建期固化后端绝对地址，再托管产物。末尾的 `/api` 与 dev 代理的目标一致：后端路由本身挂在 `/api` 下，前端请求 `${base}/chat/...`。

```bash
# 3) 前端：构建期固化后端绝对地址（preview 不复用 vite.config.js 的 dev 代理），再托管产物
#    末尾的 /api 与 dev 代理的目标一致：后端路由本身挂在 /api 下，前端请求 `${base}/chat/...`
VITE_API_BASE_URL=http://127.0.0.1:8002/api npm run build
npm run preview -- --port 4173 --strictPort --host 127.0.0.1
```

**终端 4：执行用例并校验链路。**

```bash
# 4) 跑用例
E2E_BASE_URL=http://127.0.0.1:4173 E2E_USERNAME=demo \
  E2E_PASSWORD=<与 SEED_DEMO_PASSWORD 一致> npm run test:e2e

# 5) 确认检索链路真被走到（CI 里也跑这一步，见 e2e.yml）
#    用例全绿不等于链路完整：后端路由 / 检索规划 / 重排都有兜底分支，
#    退化到兜底时用例照样绿。这一步读桩的请求日志，判定各角色是否都被调过。
python tests/e2e/check_stub_calls.py tests/e2e/artifacts/logs/stub-requests.jsonl
```

`E2E_USERNAME` 默认是 `demo`；`E2E_PASSWORD` 没有默认值，缺失时用例会给出明确失败而不是静默跳过。`E2E_RUN_ID` 默认取时间戳，用于给知识库命名，串行重复跑时也能区分批次。

主要用例只有一个完整场景：登录 → 新建知识库 → 上传夹具 `tests/e2e/fixtures/travel-expense-policy.md` → 选中知识库 → 提问「员工出差回来多久内必须提交报销？」→ 观察流式回答 → 打开引用来源抽屉。用例断言流式回答包含桩的固定标记 `E2E-STUB-ANSWER-OK`，来源抽屉展示文件名 `travel-expense-policy.md`、正文短语 `30 个工作日`，并包含 `planned #`、`rerank`、`RRF` 等链路信息。

### 模型桩与链路判定

桩是一个只依赖标准库的 OpenAI 兼容服务，按后端真实的系统提示词把请求分派成不同角色，并写入 JSONL：

| 角色 | 含义 |
| --- | --- |
| `router` | 路由：判断该不该检索 |
| `query_plan` | 检索规划：HyDE 文档、改写、关键词 |
| `embeddings` | 向量化：入库切分与查询向量 |
| `rerank` | 专用重排端点 |
| `answer` | 最终作答 |

`tests/e2e/check_stub_calls.py` 要求五个必需角色全部出现，并禁止出现 `generic` 与 `rerank_llm`：

- `generic` 说明后端换了系统提示词，桩只能给通用回复，后端随后会走自己的兜底分支。
- `rerank_llm` 说明专用重排端点没被用上，退化到了 LLM 后备重排。

缺少必需角色或出现任一禁用角色时脚本以退出码 1 失败，并打印各角色调用次数；日志文件不存在时以退出码 2 失败。日志中的坏行会直接报错，不做静默跳过。

**用例全绿不等于链路完整。** 后端在路由、检索规划、重排上各自都有兜底分支，任何一环退化到兜底，浏览器行为仍可能通过。桩的请求日志才是链路是否完整的硬断言，因此本地验收也应执行第 5 步。

### 构建产物与失败留痕

所有产物落在 `tests/e2e/artifacts/`，该目录已被 gitignore：

- 成功路径会附加 8 张全页截图：`1-logged-in`、`2-knowledge-base-created`、`3-file-uploaded`、`4-knowledge-base-selected`、`5-streaming`、`6-answer`、`7-sources-drawer`、`8-ring-selected`。
- 流式阶段还会附加 `streaming-partial.txt`，用于回看流式过程中的中间文本。
- HTML 报告输出到 `tests/e2e/artifacts/report`，查看方式：

  ```bash
  npx playwright show-report tests/e2e/artifacts/report
  ```

- 失败时额外保留 `trace.zip` 与失败截图；通过时按设计不保留 trace，避免产物膨胀。
- 失败日志、桩请求日志与后端/前端日志由 CI 一并收集并上传为 `e2e-artifacts`，保留 7 天。

## CI 与本地差异

CI 与本地的唯一差异是 workflow 注入的服务地址：MySQL 使用 CI 的 service 容器，模型上游全部指向本机桩，向量库与 checkpointer 使用临时文件。用例本身没有 `if (CI)` 分支，本地通过即代表 CI 应通过。

`.github/workflows/e2e.yml` 在 PR 指向 `develop` / `main`、push 到 `develop` 以及手动触发时运行，步骤与上面的四终端流程一致：

- 启动 MySQL 8.0 service，设置 `MILVUS_LITE_URI` / `MILVUS_URI` / `CHECKPOINTER_DB_PATH` 等本地路径。
- 生成随机 `SECRET_KEY` 并写入环境，播种 `demo` 账号。
- 启动桩、后端与 preview，并分别轮询健康检查确认就绪。
- 构建前端并确认 `dist/index.html` 存在。
- 执行 `npm run test:e2e`，随后执行 `check_stub_calls.py`。
- 无论成败都收集日志；`tests/e2e/artifacts/` 以 `e2e-artifacts` 上传，保留 7 天。

其余测试工作流：`python-tests.yml` 用 Python 3.10 安装后端依赖并运行 pytest；`frontend-tests.yml` 用 Node 22 执行 `npm ci` 与 `npm test`；`build.yml` 执行 `npm run build` 并验证 `dist/index.html`；`static-checks.yml` 执行 `compileall`、ruff、`node --check` 与 ESLint。这四项检查都没有 `continue-on-error`，失败即阻断。
