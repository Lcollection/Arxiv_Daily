# 项目结构

当前项目是一个静态文档站点，核心思路是把每日论文先沉淀成结构化 JSON，再生成适合阅读和部署的 MkDocs 页面。

## 目录职责

| 路径 | 职责 |
|------|------|
| `update_papers.py` | 抓取、翻译、归档、API 生成、MkDocs 构建和邮件通知的主入口。 |
| `docs/daily-papers/` | 每日论文 Markdown 页面，包含每日汇总页和各来源独立页。 |
| `docs/api/` | 前端和外部 Agent 可读取的结构化 JSON。大索引已经拆分为分页 JSON。 |
| `docs/new_style.css` | MkDocs 前端样式覆盖。 |
| `docs/assets/javascripts/paper_export.js` | 前端一键导出 Markdown、CSV、JSON 的脚本。 |
| `hooks/search_exclude.py` | 排除大量历史论文页进入搜索索引，避免 `search_index.json` 过大。 |
| `.github/workflows/schedule.yml` | 定时和手动触发的 GitHub Actions workflow。 |
| `site/` | MkDocs 构建产物，由 GitHub Actions 临时生成并部署，不再作为主分支提交重点。 |

## 数据流

```text
GitHub Actions
  -> python update_papers.py
  -> 抓取 arXiv / bioRxiv / medRxiv
  -> 标题与摘要归一化、可选翻译
  -> 写入 docs/daily-papers/
  -> 写入 docs/api/
  -> python -m mkdocs build
  -> 提交 docs/ 和配置变更
  -> 部署 site/ 到 GitHub Pages
```

## 可继续借鉴 Paperclip 类项目的结构

如果后续要向 Paperclip 类阅读工具靠拢，可以把现在的静态归档继续拆成以下层次：

| 层次 | 建议结构 | 当前对应 |
|------|----------|----------|
| Provider | 每个来源一个抓取适配器 | `get_arxiv_papers()`、`get_rxiv_papers()` |
| Normalizer | 统一论文字段 | `normalize_paper()`、`_paper_api_record()` |
| Index | 日期、来源、分页索引 | `docs/api/index.json` 和分页文件 |
| Reader UI | 列表、详情、导出、筛选 | MkDocs 页面、CSS、导出 JS |
| Personal Workflow | 收藏、稍后读、Obsidian | `export_obsidian()` |

目前还没有直接引入外部 Paperclip 项目代码。要做精确结构对齐，需要先确定你指的是哪个 Paperclip 仓库或产品链接。
