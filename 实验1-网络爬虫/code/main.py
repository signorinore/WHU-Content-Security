import requests
from bs4 import BeautifulSoup as bs
import re
import bs4

# 获取网页html文本
def getHTMLText(url):
    try:
        kv = {
            'user-agent': 'Mozilla/5.0'
        }
        r = requests.get(url, headers=kv, timeout = 30)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text
    except:
        return ""

def fillUnivList(ulist, html):
    soup = bs(html, "html.parser")
    for tr in soup.find('tbody').children:
        if isinstance(tr, bs4.element.Tag):
            tds = tr('td')
            ulist.append([tds[0].text.replace('\n', '').replace(' ', ''), re.split(' ', tds[1].text)[1], re.search(r'\w{2}', tds[2].text).
                          group(0), re.search(r'\w{2}', tds[3].text).group(0), tds[4].text.replace('\n', '').replace(' ', '')])

def printUnivList(ulist, num):
    print("{:^10}\t{:^10}\t{:^6}\t{:^6}\t{:^10}".format("排名", "学校名称", "省市", "类型", "总分"))
    for i in range(num):
        u = ulist[i]
        print("{:^10}\t{:^10}\t{:^6}\t{:^6}\t{:^10}".format(u[0], u[1], u[2], u[3], u[4]))

if __name__=='__main__':
    uinfo = []
    url = 'https://www.shanghairanking.cn/rankings/bcur/2021'
    html = getHTMLText(url)
    fillUnivList(uinfo, html)
    printUnivList(uinfo, 20)

