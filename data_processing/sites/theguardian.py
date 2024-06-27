# TheGuardian.py
import requests
from bs4 import BeautifulSoup
# from get_data import process_article_text

# REDACTED HEADERS
headers = {

}

response = requests.get('https://www.theguardian.com/us', headers=headers)
soup = BeautifulSoup(response.content, 'html.parser')
# find <a> with href including article/2024
links = soup.find_all('a', href=lambda href: href and 'article/2024' in href)
for linkElement in links:
    try:
        link = linkElement['href']

        
        # find article title div data-gu-name="headline


        # reformat link to include https://www.theguardian.com
        if 'https://www.theguardian.com' not in link:
            link = 'https://www.theguardian.com' + link

        response = requests.get(link, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')

        articleTitle = soup.find('div', {'data-gu-name': 'headline'}).text

        # find div with class article-body-viewer-selector
        article = soup.find('div', class_='article-body-viewer-selector')

        # print all text in article

        articleText = article.text

        # find author <a> with rel="author"
        author = soup.find('a', rel='author')
        authorName = author.text

        # Remove all non-ascii, non-alphanumeric characters, or if not in range of 32-128
        articleTitle = ''.join([c for c in articleTitle if 32 <= ord(c) <= 127])
        authorName = ''.join([c for c in authorName if 32 <= ord(c) <= 127])
        articleText = ''.join([c for c in articleText if 32 <= ord(c) <= 127])
        

        # Remove any non-alpha-numeric characters
        # exit()
        # process_article_text(articleTitle, authorName, articleText)
    except Exception as e:
        print(e)
        continue
