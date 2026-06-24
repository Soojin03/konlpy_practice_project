import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import torch
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from konlpy.tag import Kkma
from PIL import Image
import numpy as np

# ============================================
# 1. 텍스트 수집
# ============================================
url = "https://m.sports.naver.com/fifaworldcup2026/article/117/0004078155"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

res = requests.get(url, headers=headers)
res.encoding = "utf-8"
soup = BeautifulSoup(res.text, "html.parser")

content_area = soup.select_one("._article_content, .article_body, #articleBodyContents, .news_end")
if content_area:
    text = content_area.get_text(strip=True)
else:
    paragraphs = soup.select("p")
    text = "\n".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])

print("본문:\n", text)

with open("data/news_article.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("\n저장 완료: news_article.txt")


# ============================================
# 2. 텍스트 정제
# ============================================
with open("data/news_article.txt", "r", encoding="utf-8") as f:
    text = f.read()

clean_text = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣\s]", " ", text)
clean_text = re.sub(r"\s+", " ", clean_text).strip()

print("\n정제된 텍스트:")
print(clean_text)


# ============================================
# 3. 형태소 분석
# ============================================
kkma = Kkma()
morphs = kkma.pos(clean_text)

print("\n형태소 분석 결과:")
print(morphs)


# ============================================
# 4. 불용어 제거
# ============================================
words = [word for word, pos in morphs]

stopwords = ["게티이미지코리아", "마이데일리", "기자", "데일리", "마이"]
words = [word for word in words if word not in stopwords]
words = [word for word in words if len(word) > 1]

print("\n불용어 제거 후:")
print(words)


# ============================================
# 5. 단어 빈도 계산 + PyTorch Tensor 변환
# ============================================
vocab = sorted(set(words))
word_to_id = {word: idx for idx, word in enumerate(vocab)}
word_ids = [word_to_id[word] for word in words]

word_tensor = torch.tensor(word_ids, dtype=torch.long)
print("\nPyTorch Tensor:")
print(word_tensor)

word_counts_tensor = torch.bincount(word_tensor)
word_freq = {
    vocab[i]: int(word_counts_tensor[i].item())
    for i in range(len(vocab))
}
word_freq = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True))

print("\n단어 빈도:")
print(word_freq)


# ============================================
# 6. 워드클라우드 생성 + 상위 20개 단어 시각화
# ============================================
font_path = "./fonts/malgunsl.ttf"

font_prop = fm.FontProperties(fname=font_path)
plt.rcParams["font.family"] = font_prop.get_name()

# 상위 20개 단어만 추출
top20 = dict(pd.Series(word_freq).head(20))
print('상위 20개 단어:',top20)
# 원형 마스크
circle_mask = np.array(Image.open("images/circle.png"))

# 연한 초록~연두 컬러 함수
def green_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    import random
    colors = [
        "hsl(90, 60%, 60%)",
        "hsl(100, 50%, 65%)",
        "hsl(120, 45%, 65%)",
        "hsl(80, 55%, 70%)",
        "hsl(110, 40%, 70%)",
    ]
    return random.choice(colors)

wordcloud = WordCloud(
    font_path=font_path,
    width=1200,
    height=1200,
    background_color="white",
    max_words=20,
    mask=circle_mask,
    color_func=green_color_func
).generate_from_frequencies(top20)

plt.figure(figsize=(24, 24))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("월드컵 뉴스 기사 워드 클라우드 (상위 20개)", fontsize=18)
plt.show()