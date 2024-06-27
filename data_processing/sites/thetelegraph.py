# TheTelegraph.py
import requests
from bs4 import BeautifulSoup
# from get_data import process_article_text

# REDACTED HEADERS
headers = {

}

response = requests.get('https://www.telegraph.co.uk/us/', headers=headers)
soup = BeautifulSoup(response.content, 'html.parser')
# find <a> with class u-clickable-area__link
links = soup.find_all('a', href=lambda href: href and 'news' in href, class_='u-clickable-area__link')
x=0
for link in links:
    x+=1
    if x == 1:
        continue

    articleLink = "https://www.telegraph.co.uk" + (link['href'])
    print(articleLink)
    response = requests.get(articleLink, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    articleTitle = soup.find('h1').text

    try:
        # find div with data-component="text-block"
        article = soup.find_all('div', class_='article-body-text')
        # extract all text
        text = ""
        for articleText in article:
            text += articleText.text

        # process_article(articleTitle, text)
    except Exception as e:
        print(e)
        print(f"Error processing article: {articleLink}")