# 配置说明

这个仓库的每日更新由 GitHub Actions 调用 `python update_papers.py` 完成。脚本会抓取论文、生成 Markdown 页面、生成 API JSON、构建 MkDocs 静态站点，并在有配置邮件参数时发送通知。

## 默认抓取范围

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `PAPER_LIMIT` / `--limit` | `10` | 每个来源最多抓取 10 篇，因此默认上限是 arXiv 10 篇、bioRxiv 10 篇、medRxiv 10 篇。 |
| `PAPER_DATE` / `--date` | 昨天 | 默认抓取脚本运行当天的前一天。GitHub Actions 已设置 `TZ=Asia/Shanghai`，因此定时任务按北京时间计算昨天。 |
| `ARXIV_QUERY` / `--query` | 见下方 | 只作用于 arXiv。 |
| `ARXIV_LOOKBACK_DAYS` / `--arxiv-lookback-days` | `5` | arXiv 目标日期没有论文时，向前最多回看 5 天，避免周末或无发布日返回空列表。 |
| `ARXIV_PAGE_SIZE` | `50` | arXiv API 每次请求的分页大小。调小可以降低单次请求压力。 |
| `ARXIV_DELAY_SECONDS` | `8.0` | arXiv API 分页请求之间的等待秒数。遇到 429 时可调大。 |
| `ARXIV_NUM_RETRIES` | `8` | arXiv 客户端请求失败时的重试次数。 |
| `RXIV_KEYWORDS` / `--rxiv-keywords` | 空 | 可选，按标题和摘要过滤 bioRxiv / medRxiv，多个关键词用逗号、分号或换行分隔。 |
| `RXIV_EXCLUDE_KEYWORDS` / `--rxiv-exclude-keywords` | 空 | 可选，排除 bioRxiv / medRxiv 中命中的标题和摘要关键词。 |
| `API_INDEX_PAGE_SIZE` | `100` | API 索引分页大小，用于避免生成过大的 JSON 文件。 |

默认 arXiv 查询条件：

```text
cat:cs.NE OR cat:cs.MA OR cat:cs.LG OR cat:cs.CV OR cat:cs.CL OR cat:cs.AI OR cat:stat.ML OR cat:q-bio.BM OR cat:q-bio.CB OR cat:q-bio.GN OR cat:q-bio.MN OR all:"large language model" OR all:"large language models" OR all:LLM OR all:LLMs OR all:"foundation model" OR all:"foundation models" OR all:"generative AI" OR all:"mixture of experts" OR all:"retrieval augmented generation" OR all:"LLM agent" OR all:"LLM agents" OR all:"reasoning model" OR all:"post training" OR ((cat:cs.DC OR cat:cs.PF OR cat:cs.AR OR cat:cs.OS OR cat:cs.NI) AND (all:"machine learning" OR all:"deep learning" OR all:"neural network" OR all:"large language model" OR all:LLM OR all:"model serving" OR all:"LLM serving" OR all:"inference serving" OR all:"distributed training" OR all:"training system" OR all:"inference optimization" OR all:"KV cache" OR all:"speculative decoding" OR all:"GPU cluster" OR all:GPU OR all:accelerator))
```

默认查询现在覆盖四类关注方向：

| 方向 | 覆盖方式 |
|------|----------|
| LLM 算法 | `cs.CL`、`cs.LG`、`cs.AI`、`stat.ML`，以及 `large language model`、`LLM`、`foundation model`、`mixture of experts`、`retrieval augmented generation`、`reasoning model` 等关键词。 |
| ML Sys | `cs.DC`、`cs.PF`、`cs.AR`、`cs.OS`、`cs.NI` 中同时命中 `machine learning`、`deep learning`、`neural network` 等关键词的论文。 |
| AI Infra | 同上，额外关注 `model serving`、`LLM serving`、`inference serving`、`distributed training`、`KV cache`、`speculative decoding`、`GPU`、`accelerator` 等基础设施关键词。 |
| 既有方向 | 保留原有机器学习、计算机视觉、自然语言处理、人工智能和部分计算生物学分类。 |

arXiv API 分类检索使用 `cat:` 前缀。系统和基础设施分类被关键词约束，是为了避免把普通分布式系统、操作系统或网络论文大量混入每日结果。bioRxiv 和 medRxiv 当前按日期从对应来源抓取，再按 `RXIV_KEYWORDS` / `RXIV_EXCLUDE_KEYWORDS` 过滤，最后按 `PAPER_LIMIT` 截断。

GitHub Actions 里可以用仓库级 Variables 调整非敏感配置：

| Variable | 示例 |
|----------|------|
| `PAPER_DATE` | `2026-05-02` |
| `PAPER_LIMIT` | `20` |
| `ARXIV_QUERY` | `cat:cs.LG OR cat:cs.CL OR all:"large language model"` |
| `ARXIV_LOOKBACK_DAYS` | `5` |
| `ARXIV_DELAY_SECONDS` | `12` |
| `ARXIV_NUM_RETRIES` | `10` |
| `RXIV_KEYWORDS` | `machine learning, single cell, genomics` |
| `RXIV_EXCLUDE_KEYWORDS` | `case report` |

`PAPER_DATE` 适合临时手动补跑；如果长期保留这个变量，定时任务会一直抓同一天，因此日常定时任务建议留空。

## GitHub Actions Secrets

| Secret | 是否必需 | 用途 |
|--------|----------|------|
| `DEEPSEEK_API_KEY` | 可选 | 有值时用于翻译标题和摘要；没有值时保留原文。 |
| `SMTP_USERNAME` | 可选 | 发件邮箱账号。 |
| `SMTP_PASSWORD` | 可选 | 邮箱 SMTP 授权码。 |
| `EMAIL_RECIPIENT` | 可选 | 接收更新通知的邮箱。 |

SMTP 服务默认值在代码中配置为：

```text
SMTP_SERVER=smtp.163.com
SMTP_PORT=25
```

workflow 已经预留了 `SMTP_SERVER` 和 `SMTP_PORT` 两个可选 secret。没有配置时会继续使用 163 邮箱的默认 SMTP 地址；如果使用其他邮箱服务，可以在 GitHub Secrets 里添加这两个值。

## 常用手动命令

只重建索引和 API，不重新抓取：

```bash
python update_papers.py --rebuild-index-only
```

抓取指定日期，每个来源最多 20 篇：

```bash
python update_papers.py --date 2026-01-06 --limit 20
```

覆盖 arXiv 查询条件：

```bash
python update_papers.py --query "cat:cs.LG OR cat:cs.AI OR cat:cs.CL"
```

只看 LLM 算法、ML Sys 和 AI Infra 的示例：

```bash
python update_papers.py --query "cat:cs.LG OR cat:cs.CL OR cat:stat.ML OR all:\"large language model\" OR all:LLM OR all:\"foundation model\" OR all:\"mixture of experts\" OR all:\"retrieval augmented generation\" OR ((cat:cs.DC OR cat:cs.PF OR cat:cs.AR OR cat:cs.OS OR cat:cs.NI) AND (all:\"model serving\" OR all:\"LLM serving\" OR all:\"inference serving\" OR all:\"distributed training\" OR all:\"KV cache\" OR all:\"speculative decoding\" OR all:GPU OR all:accelerator))"
```

在 GitHub Actions 的 `Schedule Papers Update` 页面也可以手动触发 `workflow_dispatch`，直接填写日期、每个来源数量、arXiv 查询和 bioRxiv/medRxiv 关键词。留空时会使用仓库 Variables 或脚本默认值。

不发送邮件：

```bash
python update_papers.py --skip-email
```

## arXiv 返回 0 篇时的处理

arXiv 不是每天都有新论文发布。脚本默认先查 `PAPER_DATE` 指定日期；如果当天没有结果，会在 `ARXIV_LOOKBACK_DAYS` 范围内向前查找最近可用论文，并在 API 和页面中保留每篇论文的 `published_date`，避免把补取日期和实际发布日期混淆。

## 来源临时失败时的处理

如果某个来源临时失败，例如 arXiv API 返回 429，脚本会先按 `ARXIV_DELAY_SECONDS` 和 `ARXIV_NUM_RETRIES` 的设置进行重试。仍然失败时，只要其他来源成功，任务会继续生成当天页面，并在失败来源的独立页面中写入“抓取失败”提示，避免 bioRxiv / medRxiv 等已抓到的结果被整次丢弃。只有所有来源都失败时，脚本才会让 GitHub Actions 失败。
