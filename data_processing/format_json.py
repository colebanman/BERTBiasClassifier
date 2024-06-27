# process_articles.py
import re
import json

# Combine array from allSentences with array from article1
with open("allData.json", "r", encoding="utf-8") as file:
    allSentences = json.load(file)
    print(f"Number of sentences: {len(allSentences)}")

with open("reddit.json", "r", encoding="utf-8") as file:
    article1 = json.load(file)
    print(f"Number of sentences: {len(article1)}")

allSentences.extend(article1)
print(f"Number of sentences: {len(allSentences)}")

# write the sentences to a new file
with open("allData.json", "w", encoding="utf-8") as file:
    json.dump(allSentences, file, indent=4, ensure_ascii=False)


# with open("wash.json", "r", encoding="utf-8") as file:
#     data = json.load(file) # [{"sentence": "sentence1", "label": "bias1"}, {"sentence": "sentence2", "label": "bias2"}, ...]
#     print(f"Number of sentences: {len(data)}")

#     # remove duplicate sentences
#     data = list({sentence.get('sentence', ''): sentence for sentence in data}.values())

#     print(f"Number of sentences after removing duplicates: {len(data)}")

#     # remove unlabelled sentences
#     data = [sentence for sentence in data if sentence.get('label')]
#     print(f"Number of labelled sentences: {len(data)}")

#     # # write the sentences to a new file
#     with open("wash.json", "w", encoding="utf-8") as file:
#         json.dump(data, file, indent=4, ensure_ascii=False)

# with open("articles.txt", "r") as file:
#     articles = file.readlines()

# # print number of articles
# print(f"Number of articles: {len(articles)}")
# # print(f"Total characters: {sum([len(article) for article in articles])}")

# # remove any html tags
# articles = [re.sub(r'<[^>]*>', '', article) for article in articles]

# # remove duplicate lines
# articles = list(set(articles))

# print(f"Number of articles after removing duplicates: {len(articles)}")

# # split into three seperate articles, 33% each
# article1 = articles[:int(len(articles)/3)]
# article2 = articles[int(len(articles)/3):int(2*len(articles)/3)]
# article3 = articles[int(2*len(articles)/3):]

# # write the articles to seperate files
# with open("article1.txt", "w") as file:
#     file.writelines(article1)
# with open("article2.txt", "w") as file:
#     file.writelines(article2)
# with open("article3.txt", "w") as file:
#     file.writelines(article3)