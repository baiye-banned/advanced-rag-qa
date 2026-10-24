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
- `--questions`：问题集 JSON 路径。默认使用仓库内小型脱敏夹具；也可指向符合本文
  schema 的外部问题集。外部问题集缺少必备分类只产生 warning，不阻断运行。
- `--out`：JSON 报告路径。父目录不存在时会创建。
- `--md-out`：Markdown 报告路径，默认与 `--out` 同名、扩展名为 `.md`。
- `--duplicate-threshold`：问题/答案近似重复的 difflib 相似度阈值，取值 `(0, 1]`，
  默认 `0.9`。调低会报告更多相似对，便于清洗外部问题集。

退出码：

| 退出码 | 含义 |
| ---: | --- |
| 0 | 问题集校验通过，报告已写出 |
| 1 | 问题集校验未通过（报告仍会写出，便于定位；空数组也走此分支） |
| 2 | 输入无法读取、不是 JSON 数组，或使用了未支持的模式 |

报告应写到仓库外，不提交生成的 JSON/Markdown。

## 2. 问题集格式

根节点必须是 JSON 数组，每个元素是一道题。字段固定如下：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | string | 文件内唯一，建议使用稳定编号 |
| `question` | string | 问题文本 |
| `expected_answer` | string | 标准答案；不可答题写拒答标准 |
| `evidence` | string[] | 人工可读的章节/条款标签；`answerable=false` 时允许为空 |
| `expected_doc` | string \| string[] | 生产检索返回的 `file_name`；字符串对所有 `expected_sections` 生效，数组须与之一一对应 |
| `expected_sections` | string[] | 期望命中的章节 heading/条款文本，每项是一个标准依据单元 |
| `category` | string | 题型；建议使用下方必备分类 |
| `answerable` | boolean | 是否可由知识库回答 |
| `exact_terms` | string[] | 可选；`category=exact_term` 时必填，供 #288 BM25 使用 |

`expected_doc` + `expected_sections` 是可执行的依据映射：生产检索只返回
`file_name`/`chunk_id`/`content`，harness 按“文件名相同 + 章节文本出现在 `content`”
判定命中。旧问题集若只提供 `evidence`，仍会回退到标签文本匹配，但报告会给出
`missing_evidence_mapping` warning。`expected_sections` 的数量应与 `evidence` 一致；
不一致时给出 `evidence_mapping_length_mismatch` warning。

必备分类（缺失时报告warning，不阻断外部问题集）：

- `single_fact`：单条事实
- `numeric_deadline`：数值 / 期限
- `cross_section`：跨章节
- `table`：表格
- `negation_exception`：否定 / 例外
- `unanswerable`：无答案、应拒答（同时 `answerable=false`）
- `exact_term`：20 条精确词项夹具，供 #288 使用（不计入上述六类覆盖）；每道题的
  `question` 必须至少包含一个 `exact_terms` 字面词项，否则为 error。

校验规则：

- 必填字段缺失、`id` 重复、字段类型错误、可答题没有 `evidence`，均为 error；
- 空问题数组（0 条）为 error（`empty_question_set`），进程退出码为 1；
- `category=exact_term` 的 `question` 不含任何 `exact_terms` 字面词项，为 error
  （`exact_term_not_in_question`）；
- `expected_doc` / `expected_sections` 只给一半、类型错误或数组长度不一致，为 error；
- 缺少必备分类、`category=unanswerable` 却标为可回答（或相反）、可答题缺少可执行
  依据映射、`expected_sections` 与 `evidence` 数量不一致，为 warning；
- 问题或答案的 difflib 相似度达到 `--duplicate-threshold` 时，为
  `near_duplicate_question` / `near_duplicate_answer` warning；答案近似只在问题相似度
  至少 0.5 时判定，避免短拒答文案误报；
- 校验失败时仍写出报告，`validation.errors`、`validation.missing_fields`、
  `validation.duplicate_ids`、`validation.near_duplicates` 给出定位信息。

仓库内夹具 `tests/fixtures/rag_eval_questions.json` 覆盖六类必备题型各 2 条，另含
20 条 `exact_term`，共 32 条。所有可答题都提供了 `expected_doc` +
`expected_sections`，`exact_term` 题的 `question` 均包含对应字面词项；全部为脱敏的
示例制度数据，不含真实用户语料、Key 或本机路径。

## 3. 指标定义

一阶段只计算/预留指标位置，不伪造分数；`--dry-run` 下所有指标值为 `null`，状态为
`not_run`。当调用方提供离线检索结果时，只计算 Recall@5 / Recall@20 / MRR@5，状态
变为 `partial_offline_retrieval`，其余指标仍为 `null`。

### 标准依据单元与相关性

每个可答题的 `expected_sections` 是一个标准依据单元，并映射到同一位置的
`expected_doc`（字符串则对所有单元生效）。一条检索结果（chunk）命中一个单元，当且
仅当：

1. chunk 的 `file_name` 规范化后等于 `expected_doc`（忽略大小写、空白和常见扩展名），
   且 `expected_sections` 的章节文本出现在 chunk 的 `content` 中；或
2. chunk 没有 `file_name`（content-only）时，章节文本或完整 `evidence` 标签出现在
   `content` 中。

旧问题集没有 `expected_doc` / `expected_sections` 时，退回旧规则：chunk 的
`evidence` 元数据等于标签，或标签文本出现在 `content` 中。文件名相同但章节文本不在
同一 chunk 的，不算命中。

### Recall@5 / Recall@20

- 只有 `answerable=true`、至少有一个标准依据单元、且提供了检索结果的题参与计算。
- 单题 `Recall@k = 前 k 条结果命中的标准依据单元数 / 该题标准依据单元总数`。
- 数据集 `Recall@k = 所有参与题的单题 Recall@k 的算术平均值`。
- 同一单元在结果中重复出现只计一次；分母是标准依据单元总数，不是结果条数。
- `metrics.retrieval.unmatched_evidence_count` 汇总所有参与题中“在所有已提供结果里
  都未命中”的单元数；每题的具体标签见 `per_question.<id>.unmatched_evidence`。

### MRR@5

- 对每道参与计算的可答题，按返回顺序找到第一个命中任一标准依据单元的 chunk，
  记其 1-based 排名为 `r`（`per_question.<id>.first_relevant_rank`）。
- 单题 `RR@5 = 1/r`（当 `r <= 5`）；没有相关 chunk 或 `r > 5` 时取 `0`。
- 数据集 `MRR@5 = 所有参与题的单题 RR@5 的算术平均值`。
- 例：相关 chunk 分别在第 1、3、6 位和无相关时，`MRR@5 = (1/1 + 1/3 + 0 + 0) / 4
  = 0.333333`；第 6 位的题仍记录 `first_relevant_rank=6`，但不计入 MRR@5 分子。

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

## 4. 切片基线口径

报告里有两套基线，不能混用：

- `legacy-character`：`chunk_text(chunk_size=..., chunk_overlap=...)` 显式传参时的
  历史字符窗口（1200 字符 / 150 字符重叠）。它只用于和历史结果对比，**不是**当前
  生产默认口径。
- `current-token`：`chunk_text(file_id=...)` 的默认路径。issue #287 之后生产按 token
  预算切片（`CHUNK_TOKEN_LIMIT` / `TARGET` / `OVERLAP`，硬上限再乘
  `CHUNK_TOKEN_SAFETY_RATIO`）。

一阶段只在报告的 `baseline` 段记录这两套配置（`status=configuration_only`），不运行
真实 embedding/检索，因此不给出召回分数。`tests/test_eval_harness_293.py` 用 AST 钉住
基线口径：`chunk_text` 的 `chunk_size` / `chunk_overlap` 默认值必须是 `None`（生产默认
走 token 路径），`legacy-character` 的 1200/150 是显式的历史常量；同时读取
`backend/config.py` 中 `DEFAULT_CHUNK_TOKEN_*` 与 `_env_int` / `_env_float` 的字面量
默认值，逐项与 `current-token` 基线比对，任一处漂移都会失败。二阶段再把 token 切片与
父子切片、BM25 一起跑同一问题集。

## 5. 报告结构

JSON 报告的稳定字段：

```text
schema_version   报告结构版本
mode             本例为 dry-run
generated_at     UTC 时间戳（唯一的非确定性字段）
question_set     name / source / relative_path（仅仓库内文件）/ sha256 / count
validation       ok / question_count / category_counts / answerable / unanswerable
                 missing_fields / duplicate_ids / near_duplicates / duplicate_threshold
                 missing_required_categories / warnings / errors
metrics          status / reason / values（含 recall_at_5/20、mrr_at_5）/ retrieval
                 （含 unmatched_evidence_count）/ per_question / definitions
baseline         legacy-character 配置 + current-token 生产基线配置与状态
```

Markdown 报告包含同一份分类计数、可回答分布、校验明细、指标状态和两套基线口径，
供 PR / CI 人工比对。

### 确定性

除 `generated_at` 外，同一问题集的报告内容稳定：

- 分类按名称排序；
- 问题集 SHA-256 基于规范化 JSON（键排序、紧凑分隔符）计算；
- 近似重复对按题目的文件内顺序（前项索引、后项索引）输出；
- 浮点指标保留 6 位小数；
- 外部问题集只记录文件名与 `source=external`，不写入本机绝对路径。

## 6. 一阶段边界

- 不调用真实 embedding / LLM / rerank，不联网。
- 不修改生产检索、切片、Milvus/MySQL 代码。
- 不做四配置对比；token / 父子 / BM25 对比在二阶段另开 issue。
- 不把真实 Key、本机绝对路径或用户语料写入仓库。
