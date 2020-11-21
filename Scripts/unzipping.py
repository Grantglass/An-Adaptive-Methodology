
import os, shutil, zipfile
# Set the directory you want to start from
rootDir = './Downloads/my_dataset'
for dirName, subdirList, fileList in os.walk(rootDir):
    for fname in fileList:
        if fname.endswith(".zip"):
            zip_ref = zipfile.ZipFile(dirName + "\\" + fname)
            book = './Downloads/Texts' + dirName.split('\\')[6] +'.'+dirName.split('\\')[-1]
            zip_ref.extractall(book)
            zip_ref.close()


    



        
    