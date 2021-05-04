"""
Download all the image from a certain webpage into a folder
"""
import re
import requests
from bs4 import BeautifulSoup
import os

site = 'https://thegradient.pub/gaussian-process-not-quite-for-dummies/'
folder = '../gaussian-process/assets/'
use_order = True

response = requests.get(site)

soup = BeautifulSoup(response.text, 'html.parser')
img_tags = soup.find_all('img')

urls = [img['src'] for img in img_tags]


for i, url in enumerate(urls):
    filename = re.search(r'/([\w_-]+[.](jpg|gif|png))$', url)
    if not filename:
         print("Regex didn't match with the url: {}".format(url))
         continue
    else:
        if use_order:
            filename = str(i) + '.' + filename.group(1).split('.')[-1]
        else:
            filename = filename.group(1)
        
        filename = os.path.join(folder, filename)

    with open(filename, 'wb') as f:
        if 'http' not in url:
            # sometimes an image source can be relative 
            # if it is provide the base url which also happens 
            # to be the site variable atm. 
            url = '{}{}'.format(site, url)
        response = requests.get(url)
        f.write(response.content)
