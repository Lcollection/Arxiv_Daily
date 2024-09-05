import json
import requests
from bs4 import BeautifulSoup
import datetime
from dateutil.relativedelta import relativedelta
import time
import arxiv
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random
from openai import OpenAI

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

# Function to get today's papers from Arxiv
def get_arxiv_papers(query, delay=3):
    now = datetime.date.today()
    today = now + relativedelta(days=-1)
    print(today)
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=2,  # Increase max_results to get more papers
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    results = client.results(search)
    papers = []
    
    for result in results:
        if result.published.strftime("%Y-%m-%d") == today:
            code_link = None
            for link in result.links:
                if "github" in link.href or "gitlab" in link.href:
                    code_link = link.href
                    break
            papers.append({
                "title": result.title,
                "summary": result.summary.replace('\n', ' '),
                "auther": result.authors[0],
                "pdf_link": result.pdf_url,
                "code_link": code_link,
                "category": result.categories[0]
            })
        time.sleep(3)
    
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
    today = datetime.now().strftime("%Y-%m-%d")
    url = f"https://www.biorxiv.org/archive/{today}"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        papers = []
        for article in soup.find_all("article", class_="highwire-article"):
            title = article.find("span", class_="highwire-article-title").text.strip()
            summary = article.find("p", class_="highwire-article-summary").text.strip()
            pdf_link = article.find("a", class_="highwire-article-pdf-download")["href"]
            code_link = None
            if "artificial intelligence" in title.lower() or "machine learning" in title.lower():
                papers.append({
                    "title": title,
                    "summary": summary,
                    "pdf_link": pdf_link,
                    "code_link": code_link
                })
        return papers
    else:
        raise Exception("Failed to fetch papers from biorxiv")

# Function to get today's papers from medrxiv
def get_medrxiv_papers():
    today = datetime.now().strftime("%Y-%m-%d")
    url = f"https://www.medrxiv.org/archive/{today}"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        papers = []
        for article in soup.find_all("article", class_="highwire-article"):
            title = article.find("span", class_="highwire-article-title").text.strip()
            summary = article.find("p", class_="highwire-article-summary").text.strip()
            pdf_link = article.find("a", class_="highwire-article-pdf-download")["href"]
            code_link = None
            if "artificial intelligence" in title.lower() or "machine learning" in title.lower():
                papers.append({
                    "title": title,
                    "summary": summary,
                    "pdf_link": pdf_link,
                    "code_link": code_link
                })
        return papers
    else:
        raise Exception("Failed to fetch papers from medrxiv")

# Function to translate and summarize using DeepSeek API

def translate(text):
    chat = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_URL)
    cc = "帮我把这段翻译成中文," + text
    # print(cc)
    response = chat.chat.completions.create(
        model="deepseek-chat",
        temperature=1.1,
        messages=[
            {"role": "system","content":"You are a helpful translator"},
            {"role": "user", "content": cc},
            ],
        stream=False
    )
    return response.choices[0].message.content


# Function to save papers as Markdown tables
def save_as_markdown(papers, filename, topic):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(f"# {topic} Papers\n\n")
        file.write("| 标题  | 摘要 | 作者 | PDF链接 | 代码仓库 | Title | Abstract | \n")
        file.write("|-------|---------|----------|-----------|------------------|--------------------|---------|\n")
        for paper in papers:
            title = paper['title']
            summary = paper['summary']
            pdf_link = f"[PDF]({paper['pdf_link']})"
            code_link = f"[Code]({paper['code_link']})" if paper['code_link'] else "N/A"
            translated_title = translate(title)
            translated_summary = translate(summary)
            auther = paper['auther']
            file.write(f"| {translated_title} | {translated_summary} | {auther} | {pdf_link} | {code_link} | {title} | {summary} |\n")

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

# Main function
def main():
    query = "cs.NE OR cs.MA OR cs.LG OR cs.IT OR cs.GT OR cs.GR OR cs.ET OR cs.DS OR cs.CV OR cs.CR OR cs.CL OR cs.CE OR cs.AI"

    arxiv_papers = get_arxiv_papers(query)
    biorxiv_papers = get_biorxiv_papers()
    medrxiv_papers = get_medrxiv_papers()

    save_as_markdown(arxiv_papers, "docs/arxiv_papers.md", "Arxiv")
    save_as_markdown(biorxiv_papers, "docs/biorxiv_papers.md", "BioRxiv")
    save_as_markdown(medrxiv_papers, "docs/medrxiv_papers.md", "MedRxiv")

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
