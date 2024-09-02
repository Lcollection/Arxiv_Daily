import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
import arxiv
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# DeepSeek API endpoint and key
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/translate"
DEEPSEEK_API_KEY = "your_deepseek_api_key"

# Email settings
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = "your_email@gmail.com"
SMTP_PASSWORD = "your_email_password"
EMAIL_RECIPIENT = "your_email@gmail.com"

# Function to get today's papers from Arxiv
def get_arxiv_papers(query, delay=3):
    today = datetime.now().strftime("%Y-%m-%d")
    search = arxiv.Search(
        query=query,
        max_results=1000,  # Increase max_results to get more papers
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    papers = []
    for result in search.results():
        if result.published.strftime("%Y-%m-%d") == today:
            code_link = None
            # Try to find a code link in the links
            for link in result.links:
                if "github" in link.href or "gitlab" in link.href:
                    code_link = link.href
                    break
            papers.append({
                "title": result.title,
                "summary": result.summary,
                "pdf_link": result.pdf_url,
                "code_link": code_link,
                "category": result.categories[0]  # Assume the first category is the primary one
            })
        time.sleep(delay)  # Add delay between requests
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
def translate_and_summarize(text):
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "text": text,
        "target_language": "zh",  # Translate to Chinese
        "summarize": True
    }
    response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        return result.get("translated_text"), result.get("summary")
    else:
        return None, None

# Function to save papers as Markdown tables
def save_as_markdown(papers, filename, topic):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(f"# {topic} Papers\n\n")
        file.write("| Title | Summary | PDF Link | Code Link | Translated Title | Translated Summary | Summary |\n")
        file.write("|-------|---------|----------|-----------|------------------|--------------------|---------|\n")
        for paper in papers:
            title = paper['title']
            summary = paper['summary']
            pdf_link = f"[PDF]({paper['pdf_link']})"
            code_link = f"[Code]({paper['code_link']})" if paper['code_link'] else "N/A"
            translated_title, translated_summary = translate_and_summarize(title)
            summary_text, summary_summary = translate_and_summarize(summary)
            file.write(f"| {title} | {summary} | {pdf_link} | {code_link} | {translated_title} | {translated_summary} | {summary_summary} |\n")

# Function to send email notification
def send_email(subject, body):
    msg = MIMEMultipart()
    msg['From'] = SMTP_USERNAME
    msg['To'] = EMAIL_RECIPIENT
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    server.starttls()
    server.login(SMTP_USERNAME, SMTP_PASSWORD)
    text = msg.as_string()
    server.sendmail(SMTP_USERNAME, EMAIL_RECIPIENT, text)
    server.quit()

# Function to build MkDocs site
def build_mkdocs_site():
    os.system("mkdocs build")

# Main function
def main():
    query = "cat:cs.AI OR cat:cs.LG"

    arxiv_papers = get_arxiv_papers(query)
    biorxiv_papers = get_biorxiv_papers()
    medrxiv_papers = get_medrxiv_papers()

    save_as_markdown(arxiv_papers, "docs/arxiv_papers.md", "Arxiv")
    save_as_markdown(biorxiv_papers, "docs/biorxiv_papers.md", "BioRxiv")
    save_as_markdown(medrxiv_papers, "docs/medrxiv_papers.md", "MedRxiv")

    build_mkdocs_site()

    # Send email notification
    subject = "Papers Update"
    body = f"New papers have been updated. Check the website for details."
    send_email(subject, body)

if __name__ == "__main__":
    main()
