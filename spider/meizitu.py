# coding:utf-8
'''
Created on 2016年10月25日

@author: Jay
'''

import urllib
import urllib2
import os
from bs4 import BeautifulSoup

url = 'http://www.mzitu.com/'
number = '170307'
_number = number


def getRes(_url):
    headers = {"User-Agent": "Mozilla/5.0 (Linux; Android 4.4.2; Nexus 4 Build/KOT49H)"}
    request = urllib2.Request(_url, headers=headers)
    res = urllib2.urlopen(request).read()
    return res


nexturl = url + number
count = 0

while nexturl is not None and _number == number:
    soup = BeautifulSoup(getRes(nexturl), "lxml")
    main_image = soup.find("div", "main-image")
    nexturl = main_image.a['href']
    _number = nexturl.split('/')[3]
    img = main_image.img['src']
    print img
    title = main_image.img['alt']
    count += 1
    name = title + str(count) + '.jpg'
    urllib.urlretrieve(img, './test/%s' % name)
