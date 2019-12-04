import pymongo
from pymongo import MongoClient

cluster = MongoClient("mongodb+srv://shon615:laghate8@gettingstarted-heozl.mongodb.net/test?retryWrites=true&w=majority")
db = cluster["test"]

list_of_names = db.list_collection_names()

print(list_of_names)