import os
from dotenv import load_dotenv

load_dotenv()

# Diretório base
BASE_DIR = os.path.expanduser("~/ufam/src/nlp_ufam/ufam_legis_bot")

# Diretórios de entrada e saída
CRAWLING_FOLDER = os.path.join(BASE_DIR, "data/source")
PRE_PROCESSING_FOLDER = os.path.join(BASE_DIR, "data/pre-processed")

# Caminho dos arquivos CSV
DOWNLOAD_SUMMARY_CSV_FILE = os.path.join(CRAWLING_FOLDER, "download_summary.csv")
PRE_PROCESSING_SUMMARY_CSV_FILE = os.path.join(PRE_PROCESSING_FOLDER, "processing_summary.csv")

# Configurações do web crawler
SOURCE_LINK = "https://proeg.ufam.edu.br/normas-academicas/57-proeg/146-legislacao-e-normas.html"
QUERY_SELECTOR = "#content-section"
FILE_EXTENSIONS = ["pdf"]
TIMEOUT = 30
MAX_RETRIES = 3

# Configurações do Chroma DB
CHROMA_PERSIST_DIRECTORY = os.path.join(BASE_DIR, "data/db")
CHROMA_COLLECTION_NAME = "documentos_segmentados"
SENTENCE_TRANSFORMER_MODEL = "distiluse-base-multilingual-cased-v1"

# Formato de log
LOG_FORMAT = '%(asctime)s [%(levelname)s] [%(name)s] %(module)s::%(funcName)s(%(lineno)d) -- %(message)s'

# Synthetic Dataset para Fine-tuning
FINE_TUNING_DIR = os.path.join(BASE_DIR, "data", "synthetic_dataset")
SYNTHETIC_DATASET_OUTPUT_FILE = os.path.join(FINE_TUNING_DIR, "synthetic_dataset.json")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NUM_PAIRS_TO_GENERATE = 1000
BATCH_SIZE = 10

# Configurações para geração de conjunto de dados sintéticos
MAX_CONTEXT_LENGTH = 2000
GPT_MODEL = "gpt-4o-mini"
NUM_PAIRS_TO_GENERATE = 1000
SAVE_INTERVAL = 10  # Salva o conjunto de dados a cada 10 pares gerados