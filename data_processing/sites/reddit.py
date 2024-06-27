# reddit.py
import re
import requests
from bs4 import BeautifulSoup

# REDACTED COOKIES AND HEADERS
cookies = {

}

headers = {

}

params = {
    'render-mode': 'partial',
    'is_lit_ssr': 'false',
    'top-level': '0',
    'comments-remaining': '1000',
}

data = {
    'cursor': 'xxx',
    'csrf_token': 'xxx',
}

response = requests.post(
    'https://www.reddit.com/svc/shreddit/more-comments/politics/xxxx',
    params=params,
    cookies=cookies,
    headers=headers,
    data=data,
)
soup = BeautifulSoup(response.text, 'html.parser')
target_divs = soup.find_all('div', id=lambda x: x and '-post-rtjson-content' in x)

extracted_text = ""
for div in target_divs:
    paragraphs = div.find_all('p')
    for paragraph in paragraphs:
        extracted_text += paragraph.get_text() + "\n"
# Remove links
extracted_text = re.sub(r'\[.*?\]\(.*?\)', '', extracted_text)

# Replace unicode characters, remove invalid ones
extracted_text = extracted_text.encode('ascii', 'ignore').decode()

print(extracted_text)

# Append text to reddit.txt
with open("reddit.txt", "a") as file:
    file.write(extracted_text)