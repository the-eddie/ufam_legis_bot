import os

# Diretório base
BASE_DIR = os.path.expanduser("~/ufam/src/nlp_ufam/ufam_legis_bot")

# Diretórios de entrada e saída
INPUT_FOLDER = os.path.join(BASE_DIR, "data/source")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "data/pre-processed")

# Caminho do arquivo CSV
CSV_FILE = os.path.join(INPUT_FOLDER, "download_summary.csv")

# Configurações do web crawler
SOURCE_LINK = "https://proeg.ufam.edu.br/normas-academicas/57-proeg/146-legislacao-e-normas.html"
QUERY_SELECTOR = "#content-section"
FILE_EXTENSIONS = ["pdf"]
TIMEOUT = 30
MAX_RETRIES = 3

# Formato de log
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s'