import logging
import sys
from pathlib import Path


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)
fileHandler = logging.FileHandler("logs/service.log", mode="a+")
stdoutHandler = logging.StreamHandler(sys.stdout)

formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
fileHandler.setFormatter(formatter)
stdoutHandler.setFormatter(formatter)

logger.addHandler(fileHandler)
logger.addHandler(stdoutHandler)