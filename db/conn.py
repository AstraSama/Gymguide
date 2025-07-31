from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)

# Substitua pelos nomes do seu banco e coleção
db = client["gymguide"]
collection = db["analises"]
