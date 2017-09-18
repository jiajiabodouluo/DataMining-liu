"""

@author: Jiaxing liu

"""

import sys
import requests
import bs4
from bs4 import BeautifulSoup
import time
import json
import random


details = []

url = "http://blog.feedspot.com/video_game_news/"                # the websit
print(url)

# Beautiful soup way
html = requests.get(url).text
soup = BeautifulSoup(html,'html.parser')
data = soup.find_all('tr',{'class':"trow"})

for computer in data:
    
    rank = computer.findNext('td').text                          # find rank
    
    namer  = computer.findNext('div',{'class':'trow-wrap'})      # find name   
    name = namer.findNext('strong').text
    
    facebook  = computer.findNext('td',{'class':'stats'})        # find facebook fans number         
    facebook_fans = facebook.findNext('span').text
       
    Alexa  = computer.findNext('td',{'class':'stats alx'})       # find Alexa rank
    Alexa_Rank = Alexa.findNext('span').text
    
    person_dict = {}                                             # save all data
    person_dict['Rank'] = rank.strip()
    person_dict['Name'] = name.strip()
    person_dict['Facebookfans'] = facebook_fans.strip()
    person_dict['Alexa rank'] = Alexa_Rank.strip()
    details.append(person_dict) 
#    time.sleep(random.randint(0,3))       

with open('output.json', 'w+') as f:                             # write data into json file
    output = json.dumps(details,indent=1)     
    f.write(output)
print(details)