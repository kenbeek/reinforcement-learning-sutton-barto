from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os

load_dotenv(find_dotenv())


project_dir = Path(os.getenv("PROJECT_DIR"))
output_dir = project_dir.joinpath("output")
