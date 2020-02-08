def get_random_cat_image():
    """ A useless but wonderful function """
    import requests
    import urllib
    from PIL import Image
    
    cat_url = 'https://api.thecatapi.com/v1/images/search'
    img_url = requests.get(cat_url).json()[0]['url']
    
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
               'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
               'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
               'Accept-Encoding': 'none',
               'Accept-Language': 'en-US,en;q=0.8',
               'Connection': 'keep-alive'}
    req = urllib.request.Request(img_url, headers=headers)

    with urllib.request.urlopen(req) as f:
        img = Image.open(f)
    return img
