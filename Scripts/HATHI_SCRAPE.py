import time
import urllib.request
from bs4 import BeautifulSoup
import os
import csv
import sys

lst = []

with open('RC.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        if 'id' not in row:
            lst.append(row[0])
            
last = []

# Set the directory you want to start from
rootDir = 'hathi'
for dirName, subdirList, fileList in os.walk(rootDir):
    # print('Found directory: %s' % dirName)
    for fname in fileList:
        dir_path = os.path.dirname(os.path.abspath(fname))
        if 'Crusoe' not in fname:
            fname = fname.replace('+',':')
            fname = fname.replace('=','/')
            last.append(fname[:-4])

ids = []
print(len(lst), len(last))
for id in lst:
    if id in last:
        continue
    else:
        ids.append(id)
        
books = []
pref = 'https://babel.hathitrust.org/cgi/ssd?id='

def getText(zoup):
    x = zoup.find('p', class_='Text') #finds text of page
    if x:
        return(x.contents[0])#.replace('\n', ' ')) #removes newline characters
    else:
        return('')

def findNext(zoup):
    y = zoup.find_all('a') #finds all links on page
    for link in y:
        if len(link.contents) > 0:
            if 'Next Page' == link.contents[0].replace('\n','').strip():
                return(pref.split('/cgi')[0] + link.get('href'))
            
def checkLast(zoup):
    y = zoup.find_all('p')
    for foo in y:
        z = foo.contents
        for foos in z:
            if 'This is the last page' in foos:
                return(False)
    return(True)

def getSoup(soup, lmao):
    try:
        time.sleep(.5)
        if lmao%10 == 0:
            print(lmao)
        lmao+=1
        page = urllib.request.urlopen(findNext(soup))
        soup = BeautifulSoup(page, 'html.parser')
        return([soup, lmao])
    except KeyboardInterrupt:
        sys.exit()
    except:
        return(getSoup(soup, lmao))

def getBook(id):
    bookText = ''
    time.sleep(.5)
    page = urllib.request.urlopen(pref + id)
    lmao = 0
    #page = urllib.request.urlopen('https://babel.hathitrust.org/cgi/ssd?id=aeu.ark%3A%2F13960%2Ft82j81z5j;page=ssd;view=plaintext;seq=354')
    soup = BeautifulSoup(page, 'html.parser')
    while checkLast(soup):
        bookText += getText(soup)
        bookText += "orgil"
        newInfo = getSoup(soup, lmao)
        soup = newInfo[0]
        lmao = newInfo[1]
    newBook = [id, bookText]
    books.append(newBook)
    id = id.replace(':', '+')
    id = id.replace('/', '=')
    with open(id+".txt", 'w', encoding='utf8') as bookFile:
        #bookFile.write(id + '\n')
        bookFile.write(bookText)
    print('finished book')

def httpLoop(id):
    try:
        getBook(id)
    #except KeyboardInterrupt:
        sys.exit()
    #except:
        httpLoop(id)

crount = 0

for idx in ids[60:]:

    print(idx)
    print(crount)
    crount += 1
    httpLoop(idx)
    