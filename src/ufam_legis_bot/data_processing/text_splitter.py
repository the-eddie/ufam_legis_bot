from ufam_legis_bot.config import *
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

import sys

# Adiciona o diretório pai ao PYTHONPATH
sys.path.append("/Users/edisson/ufam/src/nlp_ufam/ufam_legis_bot/src")

# Configuração do logging
logging.getLogger('chromadb').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class TextSplitter:
    """
    Divide documentos em chunks, armazenando-os no ChromaDB juntamente com embeddings para semantic search,
    """
    def __init__(self):
        self.nlp = Portuguese()
        self.nlp.add_pipe("sentencizer")

        chroma_settings = ChromaSettings(
            allow_reset=True,
            anonymized_telemetry=False
        )

        self.chroma_client = PersistentClient(
            path=CHROMA_PERSIST_DIRECTORY, settings=chroma_settings)

        # Reinicializa o database, apagando completamente o conteúdo existente.
        self.chroma_client.reset()

        self.sentence_embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=SENTENCE_TRANSFORMER_MODEL
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=self.sentence_embedder
        )

    def rule_based_splitter(self, text: str, target_size: int = 1000, overlap: int = 1) -> List[str]:
        """
        Divide o texto em chunks baseados em sentenças, com tamanho alvo e sobreposição.

        Args:
            text: O texto a ser dividido.
            target_size: O tamanho alvo para cada chunk (em caracteres).
            overlap: O número de sentenças de sobreposição entre chunks.

        Returns:
            Uma lista de chunks de texto.
        """
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        chunks = []
        current_chunk = []
        current_size = 0

        for i, sentence in enumerate(sentences):
            current_chunk.append(sentence)
            current_size += len(sentence)

            if current_size >= target_size or i == len(sentences) - 1:
                chunks.append(" ".join(current_chunk))
                current_chunk = current_chunk[-overlap:]
                current_size = sum(len(s) for s in current_chunk)

        # Tratamento do caso de borda para o último chunk
        if len(chunks) > 1 and len(chunks[-1]) < target_size / 2:
            last_chunk = chunks.pop()
            chunks[-1] += " " + last_chunk

        return chunks

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

        chunks = self.rule_based_splitter(text)

        processed_chunks = []
        for i, chunk in enumerate(chunks):
            processed_chunk = {
                "id": f"{document['file_name']}_{i}",
                "text": chunk,
                "metadata": {
                    "document_title": ', '.join(document_metadata["anchor_texts"]) if isinstance(document_metadata["anchor_texts"], list) else str(document_metadata["anchor_texts"]),
                    "document_date": document_metadata["processing_date"],
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "file_name": document['file_name']
                }
            }
            processed_chunks.append(processed_chunk)

        logger.info(f"Arquivo processado: {file_path}. Total de chunks: {len(processed_chunks)}")
        return {
            "document": {
                "anchor_texts": ', '.join(document["metadata"]["anchor_texts"]) if isinstance(document["metadata"]["anchor_texts"], list) else str(document["metadata"]["anchor_texts"]),
                "date": document["metadata"]["processing_date"],
                "file_name": document["file_name"]
            },
            "chunks": processed_chunks
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
        all_documents = []

        for index, filename in enumerate(files, start=1):
            file_path = os.path.join(folder_path, filename)
            logger.info(f"Processando arquivo {index}/{total_files}: {filename}")
            processed_data = self.process_file(file_path)
            all_documents.append(processed_data)

            # Adicionar chunks ao Chroma
            for chunk in processed_data["chunks"]:
                metadata = {
                    k: ', '.join(v) if isinstance(v, list) else v for k, v in chunk["metadata"].items()
                }
                self.collection.add(
                    ids=[chunk["id"]],
                    documents=[chunk["text"]],
                    metadatas=[metadata]
                )

        logger.info("Processamento concluído. Dados salvos no Chroma DB.")

        # Salvar todos os documentos processados em um único arquivo JSON
        output_file = os.path.join(CHROMA_PERSIST_DIRECTORY, "documents_chunks.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_documents, f, ensure_ascii=False, indent=2)
        logger.info(f"Todos os documentos processados foram salvos em: {output_file}")

def main():
    """
    Função principal para executar o processamento de texto e indexação.
    """
    try:
        splitter = TextSplitter()
        splitter.process_folder(PRE_PROCESSING_FOLDER)
    except Exception as e:
        logger.error(f"Ocorreu um erro durante o processamento de texto: {str(e)}")
        raise

if __name__ == "__main__":
    main()