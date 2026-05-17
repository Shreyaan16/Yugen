import os
from dotenv import load_dotenv

load_dotenv()

########### JWT CONSTANTS ############
SECRET_KEY = os.getenv("JWT_SECRET")
ALGORITHM = os.getenv("JWT_ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES"))
DATABASE_URL = os.getenv("DATABASE_URL")

########### RECOMMENDATION CONSTANTS ############
TOP_K = 10
UNKNOWN_GENRE_ID = 0