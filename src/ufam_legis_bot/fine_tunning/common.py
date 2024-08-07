from typing import List, Dict, Any
import json
import os
import logging
import time
from pydantic import BaseModel, Field
from typing import List

from ufam_legis_bot.config import LOG_FORMAT

# Configuração do logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Processador de documentos JSON."""

    @staticmethod
    def load_documents(folder_path: str) -> List[Dict[str, Any]]:
        """
        Carrega todos os documentos JSON de uma pasta.

        Args:
            folder_path: Caminho da pasta contendo os arquivos JSON.

        Returns:
            Lista de dicionários, cada um representando um documento.
        """
        documents = []
        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith('.json'):
                with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                    documents.append(json.load(file))
        return documents

    @staticmethod
    def extract_metadata(document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extrai metadados relevantes de um documento.

        Args:
            document: Dicionário contendo os dados do documento.

        Returns:
            Dicionário com os metadados extraídos.
        """
        return {
            "text_content": document.get("text_content", ""),
            "original_file_names": document.get("metadata", {}).get("original_file_names", []),
            "anchor_texts": document.get("metadata", {}).get("anchor_texts", [])
        }


class LLMLogger:
    """Registrador de chamadas e respostas do LLM."""

    def __init__(self, log_file_path: str):
        """
        Inicializa o registrador do LLM.

        Args:
            log_file_path: Caminho para o arquivo de log.
        """
        self.log_file_path = log_file_path

    def log_interaction(self, prompt: str, response: str):
        """
        Registra uma interação com o LLM.

        Args:
            prompt: O prompt enviado ao LLM.
            response: A resposta recebida do LLM.
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f"=== Interação em {timestamp} ===\n")
            log_file.write(f"Prompt:\n{prompt}\n\n")
            log_file.write(f"Resposta:\n{response}\n\n")
            log_file.write("=" * 50 + "\n\n")
            log_file.flush()


class FactualStatement(BaseModel):
    """Modelo para representar uma declaração factual extraída."""
    statement: str = Field(...,
                           description="Declaração factual extraída do documento")
    context: str = Field(
        ..., description="Contexto ou seção do documento de onde a declaração foi extraída")


class ExtractedFacts(BaseModel):
    """Modelo para representar a lista de declarações factuais extraídas de um documento."""
    facts: List[FactualStatement] = Field(
        ..., description="Lista de declarações factuais extraídas")
