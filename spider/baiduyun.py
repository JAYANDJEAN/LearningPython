# coding:utf-8
'''
Created on 2016年2月27日
@author: Jay
'''
import urllib
import urllib2
import re
import time

uk = 2214641459
url0 = 'http://yun.baidu.com/share'
url1 = 'http://yun.baidu.com/wap/share/home?third=0&uk=' + str(uk) + '&start='
'''
proxy = {'http':'27.24.158.155:84'}
proxy_support = urllib2.ProxyHandler(proxy)
opener = urllib2.build_opener(proxy_support)
urllib2.install_opener(opener)

exemples:
pages:
url='http://yun.baidu.com/wap/share/home?third=0&uk=2214641459&start='
links:
url='http://yun.baidu.com/wap/link?uk=2214641459&shareid=640316896&third=0'
'''


def getResponse(url):
    headers = {"User-Agent": "Mozilla/5.0 (Linux; Android 4.4.2; Nexus 4 Build/KOT49H)"}
    request = urllib2.Request(url, headers=headers)
    res = urllib2.urlopen(request).read()
    return res


def getNames(response):
    pattern = re.compile('<h3>(.+?)</h3>')
    names = re.findall(pattern, response)
    return names


def getLinks(response):
    response = re.sub('amp;', '', response)
    pattern = re.compile('"list-item"\shref="/wap(.+?)"')
    links = re.findall(pattern, response)
    return [url0 + i for i in links]


def getTitle(response):
    pattern = re.compile('<title>(.+?)</title>')
    title = re.search(pattern, response)
    return title


def getTotalNum(response):
    pattern = re.compile('totalCount:"(\d+)"')
    num = re.search(pattern, response)
    return num.group(1)


res = getResponse(url1)
# print res
f = open(r'/Volumes/File/Baidu_Share.txt', 'w')
num = getTotalNum(res)
print num
MAX = (int(num) / 20 + 1) * 20
urls = [url1 + str(i) for i in range(0, MAX, 20)]

for url in urls:
    time.sleep(2)
    res = getResponse(url)
    names = getNames(res)
    # print names
    links = getLinks(res)
    com = zip(names, links)
    for c in com:
        f.write(c[0] + '：' + '\n')
        f.write(c[1] + '\n' + '\n')
f.close()
