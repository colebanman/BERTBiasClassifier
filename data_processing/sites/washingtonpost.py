# wash.py
import requests
from bs4 import BeautifulSoup

params = {
    'td_theme_name': 'Newspaper',
    'v': '12.6.3',
}

for x in range(1,5):
    data = {
        'action': 'td_ajax_loop',
        'loopState[sidebarPosition]': '',
        'loopState[moduleId]': '1',
        'loopState[currentPage]': '0',
        'loopState[max_num_pages]': '50000',
        'loopState[atts][category_id]': '44',
        'loopState[atts][offset]': x,
        'loopState[ajax_pagination_infinite_stop]': '3',
        'loopState[server_reply_html_data]': '',
    }

    response = requests.post(
        'https://www.washingtonexaminer.com/wp-admin/admin-ajax.php',
        params=params,
        data=data,
    )

    html = response.json()['server_reply_html_data']
    soup = BeautifulSoup(html, 'html.parser')
    links = soup.find_all('a', href=lambda href: href and 'news/campaigns' in href)

    sets = []
    for link in links:
        print(link.get("href"))
        sets.append(link.get("href"))

    sets = list(set(sets))

    # For every third link, extract the text and write it to a file
    for i, link in enumerate(sets):
        if i % 3 != 0:
            continue
    
        text = ""

        response = requests.get(link)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        extracted_text = ""
        for paragraph in paragraphs:
            extracted_text += paragraph.get_text() + "\n"
        
        text += extracted_text
        with open(f"wash.txt", "a", encoding="utf-8") as file:
            file.write(text)