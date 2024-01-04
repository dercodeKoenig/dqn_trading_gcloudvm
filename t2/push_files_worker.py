import os
import time
import random
import sys

if len(sys.argv) == 2:
    if sys.argv[1] == "github":
        print("updading github...")
        os.system("git pull")
        os.system("git add *")
        os.system("git commit -m .")
        os.system("git push")
        exit()


while True:
    print("running update commands...")
    os.system("git pull")
    os.system("git add *")
    os.system("git commit -m .")
    os.system("git push")
    os.system("kaggle datasets version -m . -p .")
    
    hour = 60*60
    sleeptime = random.randint(hour*3, hour*6)
    print("sleep for", sleeptime, "seconds")
    time.sleep(sleeptime)