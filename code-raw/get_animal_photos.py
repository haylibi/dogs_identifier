import requests
import tqdm

URL = 'https://api.bing.microsoft.com/v7.0/images/search'
API_KEY = '996f57c6ba74426d951a64445a2f4bf8'

     


def save_photo(url, save_location, session=requests.Session()):
    response = session.get(url)
    if response.status_code != 200:
        return False
    
    with open(save_location + '/' + url.split("/")[-1], 'wb') as f:
        f.write(response.content)
    return True



query = 'Panda Animal'
mkt = 'en-US'


headers = { 'Ocp-Apim-Subscription-Key': API_KEY }
params = { 'q': query, 'mkt': mkt, 'count':150, 'offset':1000 }

photos = []


session = requests.Session()

response = session.get(URL, headers=headers, params=params)
r_json = response.json()

photos.extend([x['contentUrl'] for x in r_json['value']])

counter = 1
while len(photos) < 7000 and counter < 900:
    params['offset'] = r_json['nextOffset']
    
    response = session.get(URL, headers=headers, params=params)
    r_json = response.json()
    
    photos.extend([x['contentUrl'] for x in r_json['value']])
    counter += 1


photos.extend([x['contentUrl'] for x in r_json['value']])
session.close()



# Downloading all Photos
session = requests.Session()
error = []
for url in tqdm.tqdm(photos, desc='Downloading all photos'):
    if save_photo(url, 'data/pandas'): continue
    error.append(url)

session.close()