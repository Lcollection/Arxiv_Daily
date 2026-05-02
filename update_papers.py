import argparse
import hashlib
import json
import os
import random
import re
import smtplib
import subprocess
import tempfile
import time
from datetime import UTC, date, datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path


DEEPSEEK_API_URL = "https://api.deepseek.com/v1"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.163.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "25"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
EMAIL_RECIPIENT = os.getenv("EMAIL_RECIPIENT")

SOURCE_ORDER = ("arxiv", "biorxiv", "medrxiv")
SOURCE_LABELS = {
    "arxiv": "arXiv",
    "biorxiv": "bioRxiv",
    "medrxiv": "medRxiv",
}
RXIV_DOMAINS = {
    "biorxiv": "www.biorxiv.org",
    "medrxiv": "www.medrxiv.org",
}

DEFAULT_QUERY = (
    "cs.NE OR cs.MA OR cs.LG OR cs.CV OR cs.CL OR cs.AI "
    "OR q-bio.BM OR q-bio.CB OR q-bio.GN OR q-bio.MN"
)

FAMOUS_QUOTES = [
    "The only way to do great work is to love what you do. - Steve Jobs",
    "Believe you can and you're halfway there. - Theodore Roosevelt",
    "Success is not final, failure is not fatal: It is the courage to continue that counts. - Winston S. Churchill",
    "The best way to predict the future is to invent it. - Alan Kay",
    "Do not wait to strike till the iron is hot; but make it hot by striking. - William Butler Yeats",
]

DATE_SOURCE_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})-(?P<source>arxiv|biorxiv|medrxiv)$"
)
PDF_LINK_RE = re.compile(r"\[PDF\]\((?P<url>[^)]+)\)")
LEGACY_HEADING_RE = re.compile(
    r"^#\s+(?P<source>Arxiv|arXiv|BioRxiv|bioRxiv|MedRxiv|medRxiv)\s+"
    r"(?P<date>\d{4}-\d{2}-\d{2})\s+Papers\s*$",
    re.MULTILINE,
)

_translation_client = None


def default_target_date() -> str:
    return (date.today() - timedelta(days=1)).isoformat()


def validate_date(value: str) -> str:
    datetime.strptime(value, "%Y-%m-%d")
    return value


def clean_text(value) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\s+", " ", text).strip()


def markdown_cell(value) -> str:
    text = clean_text(value)
    if not text:
        return ""
    return text.replace("|", r"\|")


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def source_slug(source: str) -> str:
    normalized = source.lower()
    if normalized == "arxiv":
        return "arxiv"
    if normalized == "biorxiv":
        return "biorxiv"
    if normalized == "medrxiv":
        return "medrxiv"
    raise ValueError(f"Unsupported source: {source}")


def first_author(authors) -> str:
    if not authors:
        return ""
    if isinstance(authors, (list, tuple)):
        return clean_text(authors[0]) if authors else ""
    return clean_text(str(authors).split(";")[0])


def doi_link(doi: str) -> str:
    doi = clean_text(doi)
    if not doi:
        return ""
    if doi.startswith("http://") or doi.startswith("https://"):
        return doi
    return f"https://doi.org/{doi}"


def rxiv_pdf_link(source: str, doi: str) -> str:
    doi = clean_text(doi)
    if not doi:
        return ""
    if doi.startswith("http://") or doi.startswith("https://"):
        return doi

    domain = RXIV_DOMAINS[source_slug(source)]
    suffix = ".full.pdf" if re.search(r"v\d+$", doi) else "v1.full.pdf"
    return f"https://{domain}/content/{doi}{suffix}"


def extract_pdf_url(value: str) -> str:
    text = clean_text(value)
    match = PDF_LINK_RE.search(text)
    if match:
        return clean_text(match.group("url"))
    if text.startswith("http://") or text.startswith("https://"):
        return text
    return ""


def paper_id(date_str: str, source: str, paper: dict) -> str:
    payload = "|".join(
        [
            date_str,
            source,
            clean_text(paper.get("pdf_link")),
            clean_text(paper.get("title") or paper.get("translated_title")),
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def json_dump(data: dict | list) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def split_markdown_table_row(line: str) -> list:
    text = line.strip()
    if not text.startswith("|") or not text.endswith("|"):
        return []
    cells = re.split(r"(?<!\\)\|", text.strip("|"))
    return [cell.strip().replace(r"\|", "|") for cell in cells]


def safe_obsidian_filename(value: str) -> str:
    text = clean_text(value)
    text = re.sub(r'[<>:"/\\|?*]', "-", text)
    return text.strip(". ") or "untitled"


def get_translation_client():
    global _translation_client
    if not DEEPSEEK_API_KEY:
        return None
    if _translation_client is None:
        from openai import OpenAI

        _translation_client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_API_URL,
        )
    return _translation_client


def translate(text: str, enabled: bool = True) -> str:
    text = clean_text(text)
    if not text:
        return ""
    if not enabled:
        return text

    client = get_translation_client()
    if client is None:
        print("DEEPSEEK_API_KEY is not set; using original text without translation.")
        return text

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是中英学术翻译专家。只输出中文译文，"
                        "保留必要的英文专有名词，不要解释，不要添加 Markdown。"
                    ),
                },
                {"role": "user", "content": text},
            ],
            timeout=30,
        )
        translated = response.choices[0].message.content
        return clean_text(translated) or text
    except Exception as exc:
        print(f"Translation failed; using original text. Error: {exc}")
        return text


def normalize_paper(raw: dict, translate_enabled: bool = True) -> dict:
    title = clean_text(raw.get("title") or raw.get("translated_title"))
    summary = clean_text(raw.get("summary") or raw.get("abstract") or raw.get("translated_summary"))
    translated_title = clean_text(raw.get("translated_title")) or translate(title, translate_enabled)
    translated_summary = clean_text(raw.get("translated_summary"))
    if not translated_summary and summary:
        translated_summary = translate(summary, translate_enabled)

    return {
        "title": title,
        "translated_title": translated_title,
        "author": first_author(raw.get("author") or raw.get("authors") or raw.get("auther")),
        "pdf_link": clean_text(raw.get("pdf_link") or raw.get("pdf_url") or raw.get("url") or doi_link(raw.get("doi", ""))),
        "paper_url": clean_text(raw.get("paper_url") or raw.get("entry_url") or doi_link(raw.get("doi", ""))),
        "summary": summary,
        "translated_summary": translated_summary,
    }


def valid_paper(paper: dict) -> bool:
    return bool(clean_text(paper.get("translated_title") or paper.get("title"))) and bool(
        clean_text(paper.get("pdf_link"))
    )


def get_arxiv_papers(query: str, target_date: str, limit: int, translate_enabled: bool) -> list:
    import arxiv

    target = datetime.strptime(target_date, "%Y-%m-%d").date()
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max(200, limit * 30),
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers = []
    for result in client.results(search):
        published = result.published.date()
        if published > target:
            continue
        if published < target:
            break

        title = clean_text(result.title)
        summary = clean_text(getattr(result, "summary", ""))
        raw = {
            "title": title,
            "author": str(result.authors[0]) if result.authors else "",
            "pdf_link": result.pdf_url,
            "paper_url": result.entry_id,
            "summary": summary,
        }
        paper = normalize_paper(raw, translate_enabled)
        if valid_paper(paper):
            papers.append(paper)
        if len(papers) >= limit:
            break
        time.sleep(1)
    return papers


def get_rxiv_papers(source: str, target_date: str, limit: int, translate_enabled: bool) -> list:
    from paperscraper.get_dumps import biorxiv, medrxiv

    fetcher = biorxiv if source == "biorxiv" else medrxiv
    fd, tmp_name = tempfile.mkstemp(prefix=f"{source}-{target_date}-", suffix=".jsonl")
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        fetcher(start_date=target_date, end_date=target_date, save_path=str(tmp_path))
        papers = []
        if not tmp_path.exists():
            return papers

        with tmp_path.open("r", encoding="utf-8") as file:
            for line in file:
                if len(papers) >= limit:
                    break
                if not line.strip():
                    continue
                item = json.loads(line)
                raw = {
                    "title": item.get("title", ""),
                    "author": first_author(item.get("authors", "")),
                    "pdf_link": rxiv_pdf_link(source, item.get("doi", "")),
                    "paper_url": doi_link(item.get("doi", "")),
                    "summary": item.get("abstract", ""),
                }
                paper = normalize_paper(raw, translate_enabled)
                if valid_paper(paper):
                    papers.append(paper)
        return papers
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


class PaperManager:
    def __init__(self, target_date: str):
        self.target_date = validate_date(target_date)
        self.docs_dir = Path("docs")
        self.daily_dir = self.docs_dir / "daily-papers"
        self.api_dir = self.docs_dir / "api"
        self.api_daily_dir = self.api_dir / "daily"
        self.api_raw_dir = self.api_dir / "raw"
        self.api_source_dir = self.api_dir / "source"
        self.daily_dir.mkdir(parents=True, exist_ok=True)
        self.api_daily_dir.mkdir(parents=True, exist_ok=True)
        self.api_raw_dir.mkdir(parents=True, exist_ok=True)
        self.api_source_dir.mkdir(parents=True, exist_ok=True)

    def daily_source_path(self, date_str: str, source: str) -> Path:
        return self.daily_dir / f"{date_str}-{source}.md"

    def daily_summary_path(self, date_str: str) -> Path:
        return self.daily_dir / f"{date_str}.md"

    def source_raw_path(self, date_str: str, source: str) -> Path:
        return self.api_raw_dir / f"{date_str}-{source}.json"

    def save_daily_papers(self, papers: list, source: str) -> None:
        source = source_slug(source)
        label = SOURCE_LABELS[source]
        path = self.daily_source_path(self.target_date, source)
        normalized = []
        for paper in papers:
            normalized_paper = normalize_paper(paper, translate_enabled=False)
            if valid_paper(normalized_paper):
                normalized.append(normalized_paper)

        lines = [
            f"# {label} {self.target_date}",
            "",
        ]
        if not normalized:
            lines.extend(["> 当天没有抓取到有效论文。", ""])
        else:
            lines.extend(
                [
                    "| 标题 | 作者 | PDF链接 | 摘要 |",
                    "|------|------|--------|------|",
                ]
            )
            for paper in normalized:
                pdf_link = f"[PDF]({paper['pdf_link']})"
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            markdown_cell(paper.get("translated_title") or paper.get("title")),
                            markdown_cell(paper.get("author")),
                            pdf_link,
                            markdown_cell(paper.get("translated_summary") or paper.get("summary")),
                        ]
                    )
                    + " |"
                )
            lines.append("")

        atomic_write(path, "\n".join(lines))
        self.write_source_raw(self.target_date, source, normalized)

    def collect_dates(self) -> dict:
        dates = {}
        for path in self.daily_dir.glob("*.md"):
            match = DATE_SOURCE_RE.match(path.stem)
            if not match:
                continue
            date_str = match.group("date")
            source = match.group("source")
            dates.setdefault(date_str, {})[source] = path
        return dict(sorted(dates.items(), reverse=True))

    def migrate_legacy_source_pages(self) -> int:
        migrated = 0
        for source in SOURCE_ORDER:
            legacy_path = self.docs_dir / f"{source}_papers.md"
            if not legacy_path.exists():
                continue

            content = legacy_path.read_text(encoding="utf-8")
            matches = list(LEGACY_HEADING_RE.finditer(content))
            if not matches:
                content = self._read_legacy_from_git(source) or content
                matches = list(LEGACY_HEADING_RE.finditer(content))
            if not matches:
                continue

            for idx, match in enumerate(matches):
                heading_source = source_slug(match.group("source"))
                if heading_source != source:
                    continue
                date_str = match.group("date")
                start = match.start()
                end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
                section = content[start:end].strip() + "\n"
                target_path = self.daily_source_path(date_str, source)
                if not target_path.exists():
                    atomic_write(target_path, section)
                    migrated += 1
        return migrated

    def _read_legacy_from_git(self, source: str) -> str:
        """Read the pre-migration source archive from git when the working file is already compact."""
        try:
            safe_dir = Path.cwd().as_posix()
            result = subprocess.run(
                [
                    "git",
                    "-c",
                    f"safe.directory={safe_dir}",
                    "show",
                    f"HEAD:docs/{source}_papers.md",
                ],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            return result.stdout
        except Exception:
            return ""

    def _source_link(self, date_str: str, source: str, from_daily_dir: bool = False) -> str:
        path = self.daily_source_path(date_str, source)
        if not path.exists():
            return "-"
        href = f"{date_str}-{source}.md" if from_daily_dir else f"daily-papers/{date_str}-{source}.md"
        return f"[{SOURCE_LABELS[source]}]({href})"

    def _daily_link(self, date_str: str, from_daily_dir: bool = False) -> str:
        href = f"{date_str}.md" if from_daily_dir else f"daily-papers/{date_str}.md"
        return f"[每日汇总]({href})"

    def _source_body(self, date_str: str, source: str) -> str:
        path = self.daily_source_path(date_str, source)
        if not path.exists():
            return "> 当天没有该来源的论文页面。"
        lines = path.read_text(encoding="utf-8").splitlines()
        if lines and lines[0].startswith("# "):
            lines = lines[1:]
        while lines and not lines[0].strip():
            lines.pop(0)
        body = "\n".join(lines).strip()
        return body or "> 当天没有抓取到有效论文。"

    def _paper_api_record(self, date_str: str, source: str, paper: dict, index: int) -> dict:
        normalized = normalize_paper(paper, translate_enabled=False)
        return {
            "id": paper_id(date_str, source, normalized),
            "date": date_str,
            "source": source,
            "source_label": SOURCE_LABELS[source],
            "title": clean_text(normalized.get("title")),
            "translated_title": clean_text(normalized.get("translated_title")),
            "author": clean_text(normalized.get("author")),
            "summary": clean_text(normalized.get("summary")),
            "translated_summary": clean_text(normalized.get("translated_summary")),
            "pdf_url": clean_text(normalized.get("pdf_link")),
            "paper_url": clean_text(normalized.get("paper_url")),
            "rank": index + 1,
        }

    def write_source_raw(self, date_str: str, source: str, papers: list) -> None:
        records = [
            self._paper_api_record(date_str, source, paper, index)
            for index, paper in enumerate(papers)
            if valid_paper(normalize_paper(paper, translate_enabled=False))
        ]
        payload = {
            "version": 1,
            "date": date_str,
            "source": source,
            "source_label": SOURCE_LABELS[source],
            "count": len(records),
            "papers": records,
        }
        atomic_write(self.source_raw_path(date_str, source), json_dump(payload))

    def read_source_raw(self, date_str: str, source: str) -> list:
        path = self.source_raw_path(date_str, source)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        return data.get("papers", [])

    def parse_source_papers(self, date_str: str, source: str) -> list:
        path = self.daily_source_path(date_str, source)
        if not path.exists():
            return []

        header = []
        papers = []
        for line in path.read_text(encoding="utf-8").splitlines():
            cells = split_markdown_table_row(line)
            if not cells:
                continue
            if not header:
                if any("PDF" in cell for cell in cells):
                    header = cells
                continue
            if all(re.fullmatch(r":?-{2,}:?", cell.strip()) for cell in cells):
                continue

            def cell_at(name: str, default_index: int | None = None) -> str:
                for idx, column in enumerate(header):
                    if name in column and idx < len(cells):
                        return cells[idx]
                if default_index is not None and default_index < len(cells):
                    return cells[default_index]
                return ""

            translated_title = cell_at("标题", 0)
            author = cell_at("作者", 1)
            pdf_url = extract_pdf_url(cell_at("PDF", 2))
            if not translated_title or not pdf_url or pdf_url.lower() == "none":
                continue

            title = cell_at("Title")
            summary = cell_at("摘要")
            papers.append(
                {
                    "title": title,
                    "translated_title": translated_title,
                    "author": author,
                    "pdf_link": pdf_url,
                    "translated_summary": summary,
                }
            )
        return papers

    def source_api_payload(self, date_str: str, source: str) -> dict:
        raw_papers = self.read_source_raw(date_str, source)
        if raw_papers:
            papers = [
                self._paper_api_record(date_str, source, paper, index)
                for index, paper in enumerate(raw_papers)
            ]
        else:
            papers = [
                self._paper_api_record(date_str, source, paper, index)
                for index, paper in enumerate(self.parse_source_papers(date_str, source))
            ]
        return {
            "version": 1,
            "date": date_str,
            "source": source,
            "source_label": SOURCE_LABELS[source],
            "count": len(papers),
            "page_path": f"daily-papers/{date_str}-{source}.md",
            "papers": papers,
        }

    def daily_api_payload(self, date_str: str) -> dict:
        sources = {}
        papers = []
        for source in SOURCE_ORDER:
            source_payload = self.source_api_payload(date_str, source)
            sources[source] = {
                "source_label": SOURCE_LABELS[source],
                "count": source_payload["count"],
                "page_path": source_payload["page_path"],
            }
            papers.extend(source_payload["papers"])

        return {
            "version": 1,
            "date": date_str,
            "count": len(papers),
            "sources": sources,
            "page_path": f"daily-papers/{date_str}.md",
            "papers": papers,
        }

    def build_api_files(self) -> None:
        dates = self.collect_dates()
        generated_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        date_entries = []
        source_entries = {source: [] for source in SOURCE_ORDER}
        latest_payload = None

        for date_str in dates:
            daily_payload = self.daily_api_payload(date_str)
            daily_payload["generated_at"] = generated_at
            atomic_write(self.api_daily_dir / f"{date_str}.json", json_dump(daily_payload))
            if latest_payload is None:
                latest_payload = daily_payload

            date_entries.append(
                {
                    "date": date_str,
                    "count": daily_payload["count"],
                    "api_path": f"api/daily/{date_str}.json",
                    "page_path": f"daily-papers/{date_str}.md",
                    "sources": daily_payload["sources"],
                }
            )
            for source in SOURCE_ORDER:
                source_entries[source].append(
                    {
                        "date": date_str,
                        "count": daily_payload["sources"][source]["count"],
                        "api_path": f"api/daily/{date_str}.json",
                        "page_path": f"daily-papers/{date_str}-{source}.md",
                    }
                )

        index_payload = {
            "version": 1,
            "generated_at": generated_at,
            "latest_date": date_entries[0]["date"] if date_entries else "",
            "count": len(date_entries),
            "dates": date_entries,
            "endpoints": {
                "latest": "api/latest.json",
                "index": "api/index.json",
                "daily": "api/daily/{YYYY-MM-DD}.json",
                "source_index": "api/source/{arxiv|biorxiv|medrxiv}.json",
            },
        }
        atomic_write(self.api_dir / "index.json", json_dump(index_payload))
        atomic_write(
            self.api_dir / "latest.json",
            json_dump(latest_payload or {"version": 1, "generated_at": generated_at, "count": 0, "papers": []}),
        )

        for source in SOURCE_ORDER:
            payload = {
                "version": 1,
                "generated_at": generated_at,
                "source": source,
                "source_label": SOURCE_LABELS[source],
                "count": len(source_entries[source]),
                "dates": source_entries[source],
            }
            atomic_write(self.api_source_dir / f"{source}.json", json_dump(payload))

    def export_obsidian(self, vault_path: str, folder: str = "Arxiv Daily", export_all: bool = False) -> int:
        vault = Path(vault_path).expanduser()
        if not vault.exists():
            raise FileNotFoundError(f"Obsidian vault does not exist: {vault}")
        if not vault.is_dir():
            raise NotADirectoryError(f"Obsidian vault is not a directory: {vault}")

        dates = list(self.collect_dates().keys()) if export_all else [self.target_date]
        exported = 0
        for date_str in dates:
            payload = self.daily_api_payload(date_str)
            if not payload["papers"]:
                continue
            year_dir = vault / folder / date_str[:4]
            target = year_dir / f"{safe_obsidian_filename(date_str)}.md"
            atomic_write(target, self.obsidian_note(payload))
            exported += 1
        return exported

    def obsidian_note(self, payload: dict) -> str:
        date_str = payload["date"]
        lines = [
            "---",
            f'date: "{date_str}"',
            "tags:",
            "  - arxiv-daily",
            "  - papers",
            "sources:",
        ]
        for source in SOURCE_ORDER:
            if payload["sources"][source]["count"]:
                lines.append(f"  - {source}")
        lines.extend(
            [
                "---",
                "",
                f"# {date_str} 每日论文",
                "",
                f"- API: `api/daily/{date_str}.json`",
                f"- Web: `daily-papers/{date_str}.md`",
                "",
            ]
        )

        for source in SOURCE_ORDER:
            papers = [paper for paper in payload["papers"] if paper["source"] == source]
            if not papers:
                continue
            lines.extend([f"## {SOURCE_LABELS[source]}", ""])
            for paper in papers:
                title = paper["translated_title"] or paper["title"]
                lines.append(f"- [{title}]({paper['pdf_url']})")
                if paper["author"]:
                    lines.append(f"  - 作者: {paper['author']}")
                if paper["title"] and paper["title"] != title:
                    lines.append(f"  - 原题: {paper['title']}")
                if paper["paper_url"]:
                    lines.append(f"  - 页面: {paper['paper_url']}")
                summary = paper["translated_summary"] or paper["summary"]
                if summary:
                    lines.append(f"  - 摘要: {summary}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def build_daily_summary_pages(self) -> None:
        dates = self.collect_dates()
        for date_str, source_paths in dates.items():
            lines = [
                f"# {date_str} 每日论文",
                "",
                "| 来源 | 独立页面 |",
                "|------|----------|",
            ]
            for source in SOURCE_ORDER:
                lines.append(f"| {SOURCE_LABELS[source]} | {self._source_link(date_str, source, True)} |")
            lines.append("")

            for source in SOURCE_ORDER:
                if source not in source_paths:
                    continue
                lines.extend(
                    [
                        f"## {SOURCE_LABELS[source]}",
                        "",
                        self._source_body(date_str, source),
                        "",
                    ]
                )
            atomic_write(self.daily_summary_path(date_str), "\n".join(lines).rstrip() + "\n")

    def build_root_index(self) -> None:
        dates = self.collect_dates()
        latest_date = next(iter(dates), None)
        lines = [
            "# 每日论文索引",
            "",
            "论文已按日期拆分归档；首页只保留日期入口，避免所有历史文章挤在单个页面。",
            "",
        ]
        if latest_date:
            lines.extend([f"最近更新：[{latest_date}](daily-papers/{latest_date}.md)", ""])

        lines.extend(
            [
                "| 日期 | 每日汇总 | arXiv | bioRxiv | medRxiv |",
                "|------|----------|-------|---------|---------|",
            ]
        )
        for date_str in dates:
            lines.append(
                f"| {date_str} | {self._daily_link(date_str)} | "
                f"{self._source_link(date_str, 'arxiv')} | "
                f"{self._source_link(date_str, 'biorxiv')} | "
                f"{self._source_link(date_str, 'medrxiv')} |"
            )
        lines.append("")
        atomic_write(self.docs_dir / "index.md", "\n".join(lines))

    def build_daily_archive(self) -> None:
        dates = self.collect_dates()
        lines = [
            "# 按日期归档",
            "",
            "| 日期 | 每日汇总 | arXiv | bioRxiv | medRxiv |",
            "|------|----------|-------|---------|---------|",
        ]
        for date_str in dates:
            lines.append(
                f"| {date_str} | {self._daily_link(date_str, True)} | "
                f"{self._source_link(date_str, 'arxiv', True)} | "
                f"{self._source_link(date_str, 'biorxiv', True)} | "
                f"{self._source_link(date_str, 'medrxiv', True)} |"
            )
        lines.append("")
        atomic_write(self.daily_dir / "archive.md", "\n".join(lines))

    def build_latest_page(self) -> None:
        dates = self.collect_dates()
        latest_date = next(iter(dates), None)
        if not latest_date:
            atomic_write(self.daily_dir / "latest.md", "# 最近更新\n\n暂无论文归档。\n")
            return

        lines = [
            "# 最近更新",
            "",
            f"最近更新日期：[{latest_date}]({latest_date}.md)",
            "",
            "| 来源 | 页面 |",
            "|------|------|",
        ]
        for source in SOURCE_ORDER:
            lines.append(f"| {SOURCE_LABELS[source]} | {self._source_link(latest_date, source, True)} |")
        lines.append("")
        atomic_write(self.daily_dir / "latest.md", "\n".join(lines))

    def build_source_archives(self) -> None:
        dates = self.collect_dates()
        for source in SOURCE_ORDER:
            label = SOURCE_LABELS[source]
            lines = [
                f"# {label} 按日期归档",
                "",
                "历史论文已按日期拆分；这里保留日期入口，不再把所有文章堆在一个页面。",
                "",
                "| 日期 | 页面 |",
                "|------|------|",
            ]
            for date_str, source_paths in dates.items():
                if source not in source_paths:
                    continue
                lines.append(f"| {date_str} | [查看](daily-papers/{date_str}-{source}.md) |")
            lines.append("")
            atomic_write(self.docs_dir / f"{source}_papers.md", "\n".join(lines))

    def rebuild_indexes(self) -> None:
        self.build_daily_summary_pages()
        self.build_root_index()
        self.build_daily_archive()
        self.build_latest_page()
        self.build_source_archives()
        self.build_api_files()


def build_mkdocs_site() -> None:
    subprocess.run(["mkdocs", "build"], check=True)


def get_balance():
    if not DEEPSEEK_API_KEY:
        return None

    import requests

    try:
        response = requests.get(
            "https://api.deepseek.com/user/balance",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        print(f"Failed to read DeepSeek balance: {exc}")
        return None


def send_email(subject: str, body: str) -> bool:
    missing = [
        name
        for name, value in {
            "SMTP_USERNAME": SMTP_USERNAME,
            "SMTP_PASSWORD": SMTP_PASSWORD,
            "EMAIL_RECIPIENT": EMAIL_RECIPIENT,
        }.items()
        if not value
    ]
    if missing:
        print(f"Email skipped; missing environment variables: {', '.join(missing)}")
        return False

    msg = MIMEMultipart()
    msg["From"] = SMTP_USERNAME
    msg["To"] = EMAIL_RECIPIENT
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30) as server:
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.sendmail(SMTP_USERNAME, EMAIL_RECIPIENT, msg.as_string())
    return True


def build_email_body(target_date: str, counts: dict) -> str:
    total_papers = sum(counts.values())
    category_summary = "\n".join(
        f"{SOURCE_LABELS[source]}: {counts.get(source, 0)}" for source in SOURCE_ORDER
    )
    random_quote = random.choice(FAMOUS_QUOTES)
    balance = get_balance()

    balance_line = "DeepSeek account balance: unavailable."
    if balance and balance.get("balance_infos"):
        info = balance["balance_infos"][0]
        currency = info.get("currency", "")
        balance_line = (
            "DeepSeek account balance: "
            f"{info.get('total_balance')} {currency}; "
            f"granted: {info.get('granted_balance')} {currency}; "
            f"topped up: {info.get('topped_up_balance')} {currency}."
        )

    return (
        f"Date: {target_date}\n"
        f"Collected {total_papers} papers.\n\n"
        f"Breakdown by source:\n{category_summary}\n\n"
        f"{balance_line}\n\n"
        "New papers have been updated. Check the website for details.\n\n"
        f"{random_quote}"
    )


def fetch_all_sources(target_date: str, query: str, limit: int, translate_enabled: bool) -> dict:
    fetchers = {
        "arxiv": lambda: get_arxiv_papers(query, target_date, limit, translate_enabled),
        "biorxiv": lambda: get_rxiv_papers("biorxiv", target_date, limit, translate_enabled),
        "medrxiv": lambda: get_rxiv_papers("medrxiv", target_date, limit, translate_enabled),
    }

    fetched = {}
    failures = {}
    for source, fetcher in fetchers.items():
        try:
            fetched[source] = fetcher()
        except Exception as exc:
            failures[source] = str(exc)

    if failures:
        details = "; ".join(f"{source}: {error}" for source, error in failures.items())
        raise RuntimeError(f"Paper fetch failed; no daily files were written. {details}")
    return fetched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update daily paper pages.")
    parser.add_argument("--date", default=os.getenv("PAPER_DATE", default_target_date()))
    parser.add_argument("--limit", type=int, default=int(os.getenv("PAPER_LIMIT", "10")))
    parser.add_argument("--query", default=os.getenv("ARXIV_QUERY", DEFAULT_QUERY))
    parser.add_argument("--no-translate", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-email", action="store_true")
    parser.add_argument("--rebuild-index-only", action="store_true")
    parser.add_argument("--migrate-legacy", action="store_true")
    parser.add_argument("--obsidian-vault", default=os.getenv("OBSIDIAN_VAULT", ""))
    parser.add_argument("--obsidian-folder", default=os.getenv("OBSIDIAN_FOLDER", "Arxiv Daily"))
    parser.add_argument("--export-obsidian-all", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_date = validate_date(args.date)
    if args.limit < 1:
        raise ValueError("--limit must be greater than 0")

    manager = PaperManager(target_date)

    if args.migrate_legacy:
        migrated = manager.migrate_legacy_source_pages()
        print(f"Migrated {migrated} legacy daily pages.")

    if args.rebuild_index_only:
        manager.rebuild_indexes()
        if args.obsidian_vault:
            exported = manager.export_obsidian(
                args.obsidian_vault,
                folder=args.obsidian_folder,
                export_all=args.export_obsidian_all,
            )
            print(f"Exported {exported} Obsidian notes.")
        return

    fetched = fetch_all_sources(
        target_date=target_date,
        query=args.query,
        limit=args.limit,
        translate_enabled=not args.no_translate,
    )

    for source in SOURCE_ORDER:
        manager.save_daily_papers(fetched[source], source)

    manager.rebuild_indexes()

    if args.obsidian_vault:
        exported = manager.export_obsidian(
            args.obsidian_vault,
            folder=args.obsidian_folder,
            export_all=args.export_obsidian_all,
        )
        print(f"Exported {exported} Obsidian notes.")

    if not args.skip_build:
        build_mkdocs_site()

    if not args.skip_email:
        body = build_email_body(target_date, {source: len(fetched[source]) for source in SOURCE_ORDER})
        try:
            send_email("Papers Update", body)
        except Exception as exc:
            print(f"Email failed after successful site generation: {exc}")


if __name__ == "__main__":
    main()
