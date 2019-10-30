import pymongo
from pymongo import MongoClient

cluster = MongoClient("mongodb+srv://shon615:laghate8@gettingstarted-heozl.mongodb.net/test?retryWrites=true&w=majority")
db = cluster["test"]
collection = db["test"]

post1 = {"name": "Josh", "Score": 7}
post2 = {"name": "Aditya", "Score":10}

#collection.insert_many([post1, post2])


results = collection.find({"name":"Josh"})

for result in results:
    print(result['Score'])