# breitbart.py
import requests
from bs4 import BeautifulSoup
# from get_data import process_article_text
# REDACTED HEADERS
headers = {

}

response = requests.get('https://www.breitbart.com/', headers=headers)
soup = BeautifulSoup(response.content, 'html.parser')
# find <a> with data-testid="internal-link" and link includes /news/article
links = soup.find_all('a', href=lambda href: href and '2024-election/2024' in href)
x=0
for link in links:
    print(link)

    articleLink = (link['href'])
    response = requests.get(articleLink, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    articleTitle = soup.find('h1').text

    try:
    
        # find div with class entry-content and add all p tags text
        article = soup.find_all('div', {'class': 'entry-content'})
        

        text = ""

        for articleText in article:
            text += articleText.text

        process_article(f"<Article: {articleTitle}> {articleText}")
    except Exception as e:
        print(e)
        print(f"Error processing article: {articleLink}")