# RAG 评测基线（一阶段）

Issue #293 的离线评测 harness。目标是在**不访问网络、不读取 Key、不调用真实
embedding/LLM** 的前提下，固定问题集格式、校验规则、指标口径和报告结构，为后续切片 /
BM25 改动提供可比较的基线。二阶段才会接入 token 切片、父子切片、BM25 四配置的真实
检索对比。

与 #178 的分工：#178 负责真实 provider 的在线验收留档；本 harness 只做离线、可复现
的指标，不替代在线留档。

## 1. 使用方式

```bash
python scripts/eval_rag.py --dry-run \
  --questions tests/fixtures/rag_eval_questions.json \
  --out ../RAG_ops/reports/eval-dry-run.json
```

- `--dry-run`：一阶段唯一支持的模式，必须显式传入。
- `--questions`：问题集 JSON 路径。默认使用仓库内小型脱敏夹具；也可指向用户自己的
  65 题问题集。
- `--out`：JSON 报告路径。父目录不存在时会创建。
- `--md-out`：Markdown 报告路径，默认与 `--out` 同名、扩展名为 `.md`。

退出码：

| 退出码 | 含义 |
| ---: | --- |
| 0 | 问题集校验通过，报告已写出 |
| 1 | 问题集校验未通过（报告仍会写出，便于定位） |
| 2 | 输入无法读取、不是 JSON 数组，或使用了未支持的模式 |

报告应写到仓库外，不提交生成的 JSON/Markdown。

## 2. 问题集格式

根节点必须是 JSON 数组，每个元素是一道题。字段固定如下：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | string | 文件内唯一，建议使用稳定编号 |
| `question` | string | 问题文本 |
| `expected_answer` | string | 标准答案；不可答题写拒答标准 |
| `evidence` | string[] | 依据的章节/条款标签；`answerable=false` 时允许为空 |
| `category` | string | 题型；建议使用下方必备分类 |
| `answerable` | boolean | 是否可由知识库回答 |
| `exact_terms` | string[] | 可选；`category=exact_term` 时必填，供 #288 BM25 使用 |

必备分类（缺失时报告warning，不阻断外部问题集）：

- `single_fact`：单条事实
- `numeric_deadline`：数值 / 期限
- `cross_section`：跨章节
- `table`：表格
- `negation_exception`：否定 / 例外
- `unanswerable`：无答案、应拒答（同时 `answerable=false`）
- `exact_term`：20 条精确词项夹具，供 #288 使用（不计入上述六类覆盖）

校验规则：

- 必填字段缺失、`id` 重复、字段类型错误、可答题没有 `evidence`，均为 error；
- 缺少必备分类、`category=unanswerable` 却标为可回答（或相反），为 warning；
- 校验失败时仍写出报告，`validation.errors`、`validation.missing_fields`、
  `validation.duplicate_ids` 给出定位信息。

仓库内夹具 `tests/fixtures/rag_eval_questions.json` 覆盖六类必备题型各 2 条，另含
20 条 `exact_term`，共 32 条，全部为脱敏的示例制度数据，不含真实用户语料、Key 或
本机路径。

## 3. 指标定义

一阶段只计算/预留指标位置，不伪造分数；`--dry-run` 下所有指标值为 `null`，状态为
`not_run`。当调用方提供离线检索结果时，只计算 Recall@5 / Recall@20，状态变为
`partial_offline_retrieval`，其余指标仍为 `null`。

### Recall@5 / Recall@20

- 只有 `answerable=true` 且至少有一条 `evidence` 且提供了检索结果的题参与计算。
- 一条检索结果（chunk）与某条标准依据相关，当且仅当：
  1. chunk 的 `evidence` 元数据与标准依据标签规范化后相等；或
  2. 标准依据标签出现在 chunk 的 `content` 中（兜底）。
- 单题 `Recall@k = 前 k 条结果命中的标准依据数 / 该题标准依据总数`。
- 数据集 `Recall@k = 所有参与题的单题 Recall@k 的算术平均值`。
- 同一标签在答案中重复出现只计一次；分母是标准依据总数，不是结果条数。

### 引用命中（citation_hit）

答案引用的来源中至少一条命中标准依据的可答题数 / 参与判定的可答题数。
一阶段不生成答案，值为 `null`；二阶段在有引用输出的配置中计算。

### 答案正确性（answer_correctness）

答案判定正确的可答题数 / 参与判定的可答题数。判定器必须显式配置：二阶段默认使用
规范化文本匹配，可选接入 LLM judge；一阶段不运行。

### 拒答正确性（refusal_correctness）

正确拒答的不可答题数 / 参与判定的不可答题数。拒绝回答、明确说明知识库无依据都算
正确；给出无依据的具体答案算错误。一阶段不运行。

### 入库耗时 / 失败率

- `ingestion_latency_ms`：单文件入库耗时的平均值（毫秒）。
- `ingestion_failure_rate`：入库失败文件数 / 入库文件总数。
- 两者都只在真实入库运行时产生；一阶段为 `null`。

## 4. 字符切片基线

当前生产默认是 `backend/crud/knowledge_file.py::chunk_text` 的标题层级感知字符切片：

| 参数 | 值 |
| --- | ---: |
| `chunk_size` | 1200 字符 |
| `chunk_overlap` | 150 字符 |

一阶段只在报告的 `baseline` 段记录该配置（`status=configuration_only`），不运行真实
embedding/检索，因此不给出召回分数。`tests/test_eval_harness_293.py` 会从生产函数
的 AST 默认值反查这两个数字，配置漂移时测试失败。二阶段再把该配置与 token 切片、
父子切片、BM25 一起跑同一问题集。

## 5. 报告结构

JSON 报告的稳定字段：

```text
schema_version   报告结构版本
mode             本例为 dry-run
generated_at     UTC 时间戳（唯一的非确定性字段）
question_set     name / source / relative_path（仅仓库内文件）/ sha256 / count
validation       ok / question_count / category_counts / answerable / unanswerable
                 missing_fields / duplicate_ids / missing_required_categories
                 warnings / errors
metrics          status / reason / values / retrieval / per_question / definitions
baseline         character 切片配置与状态
```

Markdown 报告包含同一份分类计数、可回答分布、校验明细、指标状态和字符切片基线，
供 PR / CI 人工比对。

### 确定性

除 `generated_at` 外，同一问题集的报告内容稳定：

- 分类按名称排序；
- 问题集 SHA-256 基于规范化 JSON（键排序、紧凑分隔符）计算；
- 浮点指标保留 6 位小数；
- 外部问题集只记录文件名与 `source=external`，不写入本机绝对路径。

## 6. 一阶段边界

- 不调用真实 embedding / LLM / rerank，不联网。
- 不修改生产检索、切片、Milvus/MySQL 代码。
- 不做四配置对比；token / 父子 / BM25 对比在二阶段另开 issue。
- 不把真实 Key、本机绝对路径或用户语料写入仓库。
