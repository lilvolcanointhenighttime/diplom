from dotenv import load_dotenv
import os


load_dotenv()

FNS_API_KEY = os.getenv("FNS_API_KEY")
REPUTATION_API_KEY = os.getenv("REPUTATION_API_KEY")