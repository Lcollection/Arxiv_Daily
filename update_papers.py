import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from pathlib import Path
import time
import arxiv
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random
from paperscraper.get_dumps import biorxiv,medrxiv
from openai import OpenAI

# Get today date
# 问题：未导入relativedelta且错误使用datetime.date
# 修正方案：
from datetime import date, datetime
from dateutil.relativedelta import relativedelta  # 添加导入

now = date.today()  # 正确调用date类
today = now + relativedelta(days=-1)
today = today.strftime("%Y-%m-%d")

# DeepSeek API endpoint and key
DEEPSEEK_API_URL = "https://api.deepseek.com/v1"
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

# Email settings
SMTP_SERVER = "smtp.163.com"
SMTP_USERNAME = os.getenv('SMTP_USERNAME')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
EMAIL_RECIPIENT = os.getenv('EMAIL_RECIPIENT')

# List of famous quotes
FAMOUS_QUOTES = [
    "The only way to do great work is to love what you do. - Steve Jobs",
    "Believe you can and you're halfway there. - Theodore Roosevelt",
    "Success is not final, failure is not fatal: It is the courage to continue that counts. - Winston S. Churchill",
    "The best way to predict the future is to invent it. - Alan Kay",
    "Do not wait to strike till the iron is hot; but make it hot by striking. - William Butler Yeats"
]

class PaperManager:
    def __init__(self):
        self.today = datetime.now().strftime("%Y-%m-%d")
        self.daily_dir = Path("docs/daily-papers")
        self.daily_dir.mkdir(parents=True, exist_ok=True)
        
    # 修改PaperManager的表格列定义
    def save_daily_papers(self, papers: list, source: str):
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# {source} {self.today}\n\n")
            f.write("| 标题 | 作者 | PDF链接 | 代码链接 | 摘要 |\n")  # 新增代码链接列
            f.write("|------|------|---------|---------|------|\n")
            for paper in papers:
                code_link = f"[代码]({paper['code_link']})" if paper.get('code_link') else ""
                f.write(f"| {paper.get('translated_title', '')} | "
                        f"{paper.get('author', '')} | "
                        f"[PDF]({paper['pdf_link']}) | "
                        f"{code_link} | "  # 添加代码链接列
                        f"{paper.get('translated_summary', '')} |\n")
    
    def update_index(self, sources: list):
        """更新主索引页面"""
        index_file = Path("docs/index.md")
        existing_content = index_file.read_text(encoding="utf-8") if index_file.exists() else ""
        
        # 生成新的索引内容
        new_content = [
            "# 每日论文索引\n",
            "| 日期 | arXiv | bioRxiv | medRxiv |",
            "|------|-------|---------|---------|"
        ]
        
        # 查找已有日期
        dates = sorted({
            f.stem.split("-")[0] 
            for f in self.daily_dir.glob("*.md")
        }, reverse=True)[:30]  # 保留最近30天
        
        for date_str in dates:
            arxiv = f"[查看](daily-papers/{date_str}-arxiv.md)" if (self.daily_dir / f"{date_str}-arxiv.md").exists() else "-"
            biorxiv = f"[查看](daily-papers/{date_str}-biorxiv.md)" if (self.daily_dir / f"{date_str}-biorxiv.md").exists() else "-"
            medrxiv = f"[查看](daily-papers/{date_str}-medrxiv.md)" if (self.daily_dir / f"{date_str}-medrxiv.md").exists() else "-"
            new_content.append(f"| {date_str} | {arxiv} | {biorxiv} | {medrxiv} |")
        
        # 保留旧内容中的非索引部分
        preserved_content = []
        if existing_content:
            for line in existing_content.split("\n"):
                if not line.startswith("|") and not line.startswith("# 每日论文索引"):
                    preserved_content.append(line)
        
        # 合并内容
        full_content = "\n".join(new_content + ["\n"] + preserved_content)
        index_file.write_text(full_content, encoding="utf-8")


# Function to get today's papers from Arxiv
# 修改arxiv论文处理中的代码链接提取逻辑
def get_arxiv_papers(query, delay=3):
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=200,  # Increase max_results to get more papers
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    results = client.results(search)
    papers = []
    
    # 在arxiv论文处理循环中添加翻译字段
    for result in results:
        # 同时从链接和摘要提取代码链接
        code_link = None
        # 方法1：检查论文链接
        for link in result.links:
            if "github" in link.href or "gitlab" in link.href:
                code_link = link.href
                break
        # 方法2：从摘要文本提取
        if not code_link and hasattr(result, 'summary'):
            code_link = extract_code_link(result.summary)
            
        papers.append({
            "title": result.title,
            "author": result.authors[0],
            "pdf_link": result.pdf_url,
            "code_link": code_link,  # 使用合并后的代码链接
            "category": result.categories[0],
            "translated_title": translate(result.title),  # 新增翻译字段
            "translated_summary": translate(result.summary) if hasattr(result, 'summary') else ""
        })
        if len(papers) >= 10:  # 测试模式下限制10篇
            break
        time.sleep(3)
    return papers[:10]  # 双重保险确保数量限制
    
    # papers = []
    # for result in client.results(search):  # Use client.results
    #     # Ensure the paper is published today
    #     if result.published.strftime("%Y-%m-%d") == today:
    #         code_link = None
    #         # Try to find a code link in the links
    #         for link in result.links:
    #             if "github" in link.href or "gitlab" in link.href:
    #                 code_link = link.href
    #                 break
    #         papers.append({
    #             "title": result.title,
    #             "summary": result.summary,
    #             "pdf_link": result.pdf_url,
    #             "code_link": code_link,
    #             "category": result.categories[0]  # Assume the first category is the primary one
    #         })
    #     time.sleep(delay)  # Add delay between requests
    
    return papers

# Function to get today's papers from biorxiv
def get_biorxiv_papers():
    # 修正参数名称 begin_date -> start_date
    biorxiv(start_date=today, end_date=today, save_path="biorxiv.jsonl")
    jsonl_file = "biorxiv.jsonl"
    papers = []
    with open(jsonl_file, "r") as file:
        for line in file:
            if len(papers) >= 10:  # 限制10篇
                break
            json_obj = json.loads(line)
            papers.append({
                "title": json_obj["title"],
                "author": json_obj["authors"].split(';')[0],
                "pdf_link": "https://doi.org/" + json_obj["doi"],  # 修正.org缺失
                "code_link": extract_code_link(json_obj["abstract"]),  # 新增代码链接提取
                "translated_title": translate(json_obj["title"]),  # 新增翻译字段
                "translated_summary": ""  # 占位符
            })
    return papers[:10]  # 返回前10条

def get_medrxiv_papers():
    # 相同修正
    medrxiv(start_date=today, end_date=today, save_path="medrxiv.jsonl") 
    jsonl_file = "medrxiv.jsonl"
    papers = []
    with open(jsonl_file, "r") as file:
        for line in file:
            json_obj = json.loads(line)
            papers.append({
                "title": json_obj["title"],
                "auther":json_obj["authors"].split(';')[0],
                "pdf_link": "https://doi.org/" + json_obj["doi"],
                "code_link": None,
            })
    os.remove("medrxiv.jsonl")
    return papers

    
# Function to translate and summarize using DeepSeek API

# 问题：翻译指令不清晰
def translate(text):
    cc = "请将以下学术论文内容翻译成中文，保持专业术语准确性：\n" + text  # 优化指令
    response = chat.chat.completions.create(
        model="deepseek-chat",
        temperature=0.7,  # 降低随机性
        messages=[
            {"role": "system","content":"You are a helpful translator"},
            {"role": "user", "content": cc},
            ],
        stream=False
    )
    return response.choices[0].message.content


# Function to save papers as Markdown tables
# 问题：save_as_markdown使用"r+"模式可能导致文件不存在错误
# 修正方案：
def save_as_markdown(papers, filename, topic):
    with open(filename, "w", encoding="utf-8") as file:  # 改为写入模式
        file.write(f"# {topic} {today} Papers\n\n")
        file.write("| 标题 | 作者 | PDF链接 | 代码仓库 | Title | \n")
        file.write("|-------|----------|-----------|---------|--------------| \n")
        for paper in papers:
            title = paper['title']
            pdf_link = f"[PDF]({paper['pdf_link']})"
            code_link = f"[Code]({paper['code_link']})" if paper['code_link'] else "N/A"
            translated_title = translate(title)
            auther = paper['auther']
            file.write(f"| {translated_title} | {auther} | {pdf_link} | {code_link} | {title} |\n")        
        file.write(old)

# Function to send email notification
def send_email(subject, body):
    msg = MIMEMultipart()
    msg['From'] = SMTP_USERNAME
    msg['To'] = EMAIL_RECIPIENT
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    server = smtplib.SMTP(SMTP_SERVER)
    server.starttls()
    server.login(SMTP_USERNAME, SMTP_PASSWORD)
    text = msg.as_string()
    server.sendmail(SMTP_USERNAME, EMAIL_RECIPIENT, text)
    server.quit()

# Function to build MkDocs site
def build_mkdocs_site():
    os.system("mkdocs build")

# Function to get DeepSeek account balance
def get_balance():
    url = "https://api.deepseek.com/user/balance"
    payload={}
    Token = "Bearer " + DEEPSEEK_API_KEY
    headers = {
        'Accept': 'application/json',
        'Authorization': Token
    }
    response = requests.request("GET", url, headers=headers, data=payload)
    print(response.text)
    return response.text


def main():
    TEST_MODE = True  # 测试模式开关
    query = "cs.NE OR cs.MA OR cs.LG OR cs.CV OR cs.CL OR cs.AI OR q-bio.BM OR q-bio.CB OR q-bio.GN OR q-bio.MN"

    manager = PaperManager()
    arxiv_papers = get_arxiv_papers(query)[:10] if TEST_MODE else get_arxiv_papers(query)
    biorxiv_papers = get_biorxiv_papers()[:10] if TEST_MODE else get_biorxiv_papers()
    medrxiv_papers = get_medrxiv_papers()[:10] if TEST_MODE else get_medrxiv_papers()

    manager.save_daily_papers(arxiv_papers, "arxiv")
    manager.save_daily_papers(biorxiv_papers, "biorxiv")
    manager.save_daily_papers(medrxiv_papers, "medrxiv")
    
    manager.update_index(["arxiv", "biorxiv", "medrxiv"])

    build_mkdocs_site()

    # Prepare email content
    total_papers = len(arxiv_papers) + len(biorxiv_papers) + len(medrxiv_papers)
    categories = {
        "Arxiv": len(arxiv_papers),
        "BioRxiv": len(biorxiv_papers),
        "MedRxiv": len(medrxiv_papers)
    }
    category_summary = "\n".join([f"{category}: {count}" for category, count in categories.items()])
    random_quote = random.choice(FAMOUS_QUOTES)
    bal = get_balance()
    j = json.loads(bal)
    Type = j["balance_infos"][0]["currency"]
    Total_balance = j["balance_infos"][0]["total_balance"]
    Granted_balance = j["balance_infos"][0]["granted_balance"]
    Topped_up_balance = j["balance_infos"][0]["topped_up_balance"]

    subject = "Papers Update"
    body = f"Today, we have collected {total_papers} papers.\n\n" \
           f"Breakdown by source:\n{category_summary}\n\n" \
           f"In your deepseek account balance: {Total_balance} {Type} ; In the acount Grandted: {Granted_balance} {Type}, Topped up: {Topped_up_balance} {Type}.\n\n" \
           f"New papers have been updated. Check the website for details.\n\n" \
           f"{random_quote}"

    # Send email notification
    send_email(subject, body)

if __name__ == "__main__":
    main()


# 新增代码链接提取函数
def extract_code_link(text):
    import re
    # 匹配GitHub/GitLab链接
    patterns = [
        r'https?://github\.com/[^\s]+',
        r'https?://gitlab\.com/[^\s]+',
        r'Code is available at (http\S+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None
