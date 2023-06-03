import requests
from bs4 import BeautifulSoup

r = requests.get("https://top.baidu.com/board?tab=movie")
r.encoding = r.apparent_encoding
demo = r.text
soup = BeautifulSoup(demo, 'html.parser')
ulist = []
print('序号\t片名')
it = iter(soup.find_all('div', 'c-single-text-ellipsis'))
for tag in it:
    ulist.append(tag.string)
    print(ulist.index(tag.string) + 1, ulist[ulist.index(tag.string)])
    next(it)
