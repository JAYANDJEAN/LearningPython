# coding:utf-8
'''
Created on 2016年11月2日

@author: Jay
'''

import urllib
import urllib2
from bs4 import BeautifulSoup

url = 'http://www.mmjpg.com/mm/'
number = '701'
_number = number


def getRes(_url):
    headers = {"User-Agent": "Mozilla/5.0 (Linux; Android 4.4.2; Nexus 4 Build/KOT49H)"}
    request = urllib2.Request(_url, headers=headers)
    res = urllib2.urlopen(request).read()
    return res


next = url + number
count = 0

while next != None and _number == number:
    soup = BeautifulSoup(getRes(next), "lxml")
    main_image = soup.find("div", "content")
    next = main_image.a['href']
    _number = next.split('/')[4]
    img = main_image.img['src']
    title = main_image.img['alt'].split(' ')[0]
    count += 1
    name = title + str(count) + '.jpg'
    urllib.urlretrieve(img, './test/%s' % name)
