### crawler
```
#!/bin/python
coding: utf-8

import os
import re
import sys
import urllib2

import socket
socket.setdefaulttimeout(10.0)

MAXNUM = 10

def download(index):
  url = 'http://dongman.2345.com/lt/%d' % (index)
  try:
    html = urllib2.urlopen(url).read()
  except Exception:
    html = "None"
  return html

def parse(html):
  videos = []
  html = html.strip()
  pattern = r'<a title="(.+?)"' 
  find_re = re.compile(pattern, re.DOTALL) 
  for item in find_re.findall(html): 
    result = dict( title = item.decode('gbk') ) 
    videos.append(result) 
    return videos

if __name__ == '__main__':
  for i in range(MAXNUM):
    html = download(i+1)
    videos = parse(html)
    
  for item in videos:
    print item['title'].encode('utf-8')
