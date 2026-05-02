# Agent API

本站会在构建时生成一组静态 JSON 文件，Agent 可以直接通过 HTTP GET 调用。接口不下载 PDF，只返回论文元数据和唯一的 `pdf_url` 链接。

| 用途 | 路径 |
|------|------|
| 最新一天论文 | `api/latest.json` |
| 日期索引清单 | `api/index.json` |
| 日期索引分页 | `api/index/0001.json` |
| 指定日期 | `api/daily/YYYY-MM-DD.json` |
| arXiv 日期索引清单 | `api/source/arxiv.json` |
| bioRxiv 日期索引清单 | `api/source/biorxiv.json` |
| medRxiv 日期索引清单 | `api/source/medrxiv.json` |
| 来源日期索引分页 | `api/source/{source}/0001.json` |

`api/index.json` 和 `api/source/{source}.json` 只保存分页清单，避免历史归档增长后生成单个大体积 JSON。客户端应先读取清单中的 `pages[].api_path`，再按需读取对应分页。

`api/index.json` 的主体结构：

```json
{
  "version": 1,
  "generated_at": "2026-01-06T00:00:00Z",
  "latest_date": "2026-01-06",
  "count": 300,
  "chunked": true,
  "chunk_key": "dates",
  "page_size": 100,
  "page_count": 3,
  "pages": [
    {
      "page": 1,
      "count": 100,
      "api_path": "api/index/0001.json"
    }
  ]
}
```

`api/index/0001.json` 的主体结构：

```json
{
  "version": 1,
  "generated_at": "2026-01-06T00:00:00Z",
  "page": 1,
  "page_size": 100,
  "count": 100,
  "total_count": 300,
  "dates": [
    {
      "date": "2026-01-06",
      "count": 30,
      "api_path": "api/daily/2026-01-06.json",
      "page_path": "daily-papers/2026-01-06.md"
    }
  ]
}
```

## 返回字段

`api/daily/YYYY-MM-DD.json` 的主体结构：

```json
{
  "version": 1,
  "date": "2026-01-06",
  "count": 30,
  "sources": {
    "arxiv": {
      "source_label": "arXiv",
      "count": 10,
      "page_path": "daily-papers/2026-01-06-arxiv.md"
    }
  },
  "papers": [
    {
      "id": "stable-paper-id",
      "date": "2026-01-06",
      "source": "arxiv",
      "source_label": "arXiv",
      "title": "Original English title",
      "translated_title": "中文标题",
      "author": "First Author",
      "summary": "Original abstract",
      "translated_summary": "中文摘要",
      "pdf_url": "https://...",
      "paper_url": "https://...",
      "rank": 1
    }
  ]
}
```

## Obsidian

本地 Obsidian 不会在 GitHub Actions 中自动写入。需要在本机运行并指定 vault：

```bash
python update_papers.py --rebuild-index-only --obsidian-vault "D:\Your\ObsidianVault"
```

导出全部历史日期：

```bash
python update_papers.py --rebuild-index-only --obsidian-vault "D:\Your\ObsidianVault" --export-obsidian-all
```

默认会写入 `Arxiv Daily/YYYY/YYYY-MM-DD.md`，可通过 `--obsidian-folder` 或环境变量 `OBSIDIAN_FOLDER` 修改目录名。
