import os
from fnmatch import fnmatch
import zipfile
##location of database
root = './Texts'
pattern = "*.zip"
count=0

for path, subdirs, files in os.walk(root):
    for name in files:
        if fnmatch(name, pattern):
            print(os.path.join(path, name));
            library=os.path.join(path, name).split("\\")[1]
            fileNum=os.path.basename(os.path.join(path, name)).split(".")[0]
            zip_ref = zipfile.ZipFile(os.path.join(path, name), 'r')
            ##location where you want unzipped files to go, use same location for concatenateFiles
            zip_ref.extractall("./Text/unzipped/"+library+"."+fileNum)
            zip_ref.close()
            
            count+=1
            
print(count)
