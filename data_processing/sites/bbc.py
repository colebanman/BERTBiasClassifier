# BBC.py
import requests
from bs4 import BeautifulSoup

# REDACTED HEADERS
headers = {

}

response = requests.get('https://www.bbc.com/news/world', headers=headers)
soup = BeautifulSoup(response.content, 'html.parser')
# find <a> with data-testid="internal-link" and link includes /news/article
links = soup.find_all('a', {'data-testid': 'internal-link'}, href=lambda href: href and '/news/article' in href)
x=0
for link in links:
    x+=1
    if x == 1:
        continue

    articleLink = "https://www.bbc.com" + (link['href'])
    response = requests.get(articleLink, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    articleTitle = soup.find('h1').text

    try:
    
        # find div with data-component="text-block"
        article = soup.find_all('div', {'data-component': 'text-block'})

        text = ""

        for articleText in article:
            text += articleText.text

    except Exception as e:
        print(e)
        print(f"Error processing article: {articleLink}")