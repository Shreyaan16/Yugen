import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

########### JWT CONSTANTS ############
SECRET_KEY = os.getenv("JWT_SECRET")
ALGORITHM = os.getenv("JWT_ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES"))


########### RECOMMENDATION CONSTANTS ############
TOP_K = 10
TOP_K_SIMILAR = 10
UNKNOWN_GENRE_ID = 0
ALL_ANIME_DEFAULT_LIMIT = 50
ALL_ANIME_MAX_LIMIT = 200