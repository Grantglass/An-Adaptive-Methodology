
import sys
import nltk
from nltk import word_tokenize
import pickle
from book import book 
import string
import re
import xml.etree.ElementTree as ET
bookList=[]
with open("Total.txt", "rb") as fp:
        bookList=pickle.load(fp)
whiteList=string.ascii_letters+ string.digits+ " "+"."+"\'"+"\n"
with open("C.txt",encoding='utf-8', errors='ignore') as myfile:
    text=myfile.read()
root = ET.fromstring(text)
count=0
for child in root:
    count+=1
    print(child.find("documentName").text)
    segments=[]
    for segment in child.find("segments"):
        segInfo=(segment.find("depthScore").text,segment.find("text").text)
        segments.append(segInfo)
        segment.find("depthScore").text
    for b in bookList:
        if(b.code==child.find("documentName").text):
            b.segList=segments
            print(b.segList[2])
print(count)
