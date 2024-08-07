import json
import os
from typing import List, Dict
import spacy
from spacy.lang.pt import Portuguese
from chromadb import Client, PersistentClient
from chromadb import Settings as ChromaSettings
from chromadb.utils import embedding_functions
import logging
from tqdm import tqdm

from ufam_legis_bot.config import *

# Configuração do logging
logging.getLogger('chromadb').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class TextSplitter:
    def __init__(self):
        self.nlp = Portuguese()
        self.nlp.add_pipe("sentencizer")

        chroma_settings = ChromaSettings(
            allow_reset=True,
            anonymized_telemetry=False
        )

        self.chroma_client = PersistentClient(path=CHROMA_PERSIST_DIRECTORY, settings=chroma_settings)

        # Reinicializa o database, apagando completamente o conteúdo existente.
        self.chroma_client.reset()

        self.sentence_embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=SENTENCE_TRANSFORMER_MODEL
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=self.sentence_embedder
        )

    def rule_based_splitter(self, text: str) -> List[str]:
        """
        Divide o texto em sentenças usando o sentencizer do spaCy.

        Args:
            text: O texto a ser dividido.

        Returns:
            Uma lista de sentenças.
        """
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]

    def process_file(self, file_path: str) -> Dict:
        """
        Processa um arquivo JSON e retorna os chunks de texto.

        Args:
            file_path: O caminho para o arquivo JSON.

        Returns:
            Um dicionário contendo informações do documento e chunks.
        """
        logger.info(f"Processando arquivo: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            document = json.load(file)

        text = document["text_content"]
        document_metadata = document["metadata"]

        sentences = self.rule_based_splitter(text)

        chunks = []
        for i, sentence in enumerate(sentences):
            chunk = {
                "id": f"{document['file_name']}_{i}",
                "text": sentence,
                "metadata": {
                    "document_title": ', '.join(document_metadata["anchor_texts"]) if isinstance(document_metadata["anchor_texts"], list) else str(document_metadata["anchor_texts"]),
                    "document_date": document_metadata["processing_date"],
                    "chunk_index": i,
                    "total_chunks": len(sentences),
                    "file_name": document['file_name']
                }
            }
            chunks.append(chunk)

        logger.info(f"Arquivo processado: {
                    file_path}. Total de chunks: {len(chunks)}")
        return {
            "document": {
                "anchor_texts": ', '.join(document["metadata"]["anchor_texts"]) if isinstance(document["metadata"]["anchor_texts"], list) else str(document["metadata"]["anchor_texts"]),
                "date": document["metadata"]["processing_date"],
                "file_name": document["file_name"]
            },
            "chunks": chunks
        }

    def process_folder(self, folder_path: str):
        """
        Processa todos os arquivos JSON em uma pasta e salva os chunks no Chroma.

        Args:
            folder_path: O caminho para a pasta contendo os arquivos JSON.
        """
        logger.info(f"Iniciando processamento da pasta: {folder_path}")
        files = sorted([f for f in os.listdir(folder_path) if f.endswith(".json")])
        total_files = len(files)

        for index, filename in enumerate(files, start=1):
            file_path = os.path.join(folder_path, filename)
            logger.info(f"Processando arquivo {index}/{total_files}: {filename}")
            processed_data = self.process_file(file_path)

            # Adicionar chunks ao Chroma
            for chunk in processed_data["chunks"]:
                metadata = {k: ', '.join(v) if isinstance(v, list) else v for k, v in chunk["metadata"].items()}
                self.collection.add(
                    ids=[chunk["id"]],
                    documents=[chunk["text"]],
                    metadatas=[metadata]
                )

        logger.info("Processamento concluído. Dados salvos no Chroma DB.")


def main():
    """
    Função principal para executar o processamento de texto e indexação.
    """
    try:
        splitter = TextSplitter()
        splitter.process_folder(PRE_PROCESSING_FOLDER)
    except Exception as e:
        logger.error(
            f"Ocorreu um erro durante o processamento de texto: {str(e)}")
        raise


if __name__ == "__main__":
    main()
