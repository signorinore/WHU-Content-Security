import requests
from bs4 import BeautifulSoup
import re

r = requests.get("http://www.whu.edu.cn/")
r.encoding = r.apparent_encoding
demo = r.text
soup = BeautifulSoup(demo, 'html.parser')
for tag in soup.find_all('a', string=re.compile('樱')):
    print(tag.string)


