import os
import json
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from tqdm import tqdm

from ufam_legis_bot.config import *
from ufam_legis_bot.fine_tunning.common import logger, LLMLogger, FactualStatement, ExtractedFacts, DocumentProcessor

class GPTLLMClient:
    """Cliente para interação com a API do OpenAI GPT."""

    def __init__(self, llm_logger: LLMLogger):
        """
        Inicializa o cliente GPT (OpenAI).

        Args:
            llm_logger: Instância do LLMLogger para registrar interações.
        """
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.llm_logger = llm_logger

    def extract_facts(self, document: Dict[str, Any]) -> ExtractedFacts:
        """
        Extrai declarações factuais relevantes de um documento usando GPT (OpenAI).

        Args:
            document: Dicionário contendo os metadados do documento.

        Returns:
            ExtractedFacts contendo a lista de declarações factuais extraídas.
        """
        system_prompt, user_prompt = self._generate_fact_extraction_prompt(document)
        response = self._query_gpt(system_prompt, user_prompt)
        self.llm_logger.log_interaction(f"System: {system_prompt}\nUser: {user_prompt}", response)
        return response

    @staticmethod
    def _generate_fact_extraction_prompt(document: Dict[str, Any]) -> Tuple[str, str]:
        """
        Gera prompts para extração de declarações factuais.

        Args:
            document: Dicionário contendo os metadados do documento.

        Returns:
            Tupla contendo o prompt do sistema e o prompt do usuário.
        """
        system_prompt = """Você é um especialista em análise de legislação acadêmica.
        O usuário irá fornecer documentos, que são parte da legislação acadêmica da
        Universidade Federal do Amazonas (UFAM). Sua tarefa é extrair declarações factuais
        relevantes do texto fornecido, garantindo uma cobertura abrangente de todas as áreas
        importantes da legislação. Identifique fatos relevantes para diferentes personas
        (estudantes, professores, servidores da universidade, etc.), em diferentes níveis de
        especificidade, incluindo políticas gerais, procedimentos específicos, exceções e
        definições importantes. Certifique-se que a informação é de fato relevante e que ela
        contenha contexto suficiente para que seja compreendida sem informações adicionais"""

        user_prompt = f"""Analise o seguinte texto de legislação acadêmica e extraia declarações factuais relevantes:

        Texto:
        {document['text_content'][:100000]}

        Informações adicionais que podem ajudar no entendimento do documento:
        * Nomes dos arquivos originais: {', '.join(document['original_file_names'])}
        * Textos âncora: {', '.join(document['anchor_texts'])}

        Extraia declarações factuais importantes, incluindo o contexto ou seção de onde foram extraídas."""

        return system_prompt, user_prompt

    def _query_gpt(self, system_prompt: str, user_prompt: str) -> ExtractedFacts:
        """
        Consulta o GPT (OpenAI) e retorna a resposta estruturada.

        Args:
            system_prompt: Prompt do sistema para o GPT (OpenAI).
            user_prompt: Prompt do usuário para o GPT (OpenAI).

        Returns:
            ExtractedFacts contendo a lista de declarações factuais extraídas.
        """
        try:
            completion = self.client.beta.chat.completions.parse(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=ExtractedFacts,
            )
            return completion.choices[0].message.parsed
        except Exception as e:
            logger.error(f"Erro ao consultar GPT (OpenAI): {str(e)}")
            return ExtractedFacts(facts=[])

class FactExtractor:
    """Extrator de declarações factuais de documentos."""

    def __init__(self, llm_logger: LLMLogger):
        """
        Inicializa o extrator de declarações factuais.

        Args:
            llm_logger: Instância do LLMLogger para registrar interações.
        """
        self.document_processor = DocumentProcessor()
        self.gpt_llm_client = GPTLLMClient(llm_logger)

    def extract_facts_from_documents(self, folder_path: str) -> Dict[str, ExtractedFacts]:
        """
        Extrai declarações factuais de todos os documentos em uma pasta.

        Args:
            folder_path: Caminho da pasta contendo os documentos JSON.

        Returns:
            Dicionário mapeando nomes de arquivos para ExtractedFacts.
        """
        documents = self.document_processor.load_documents(folder_path)
        facts_by_document = {}

        for document in tqdm(documents, desc="Extraindo declarações factuais dos documentos"):
            metadata = self.document_processor.extract_metadata(document)
            facts = self.gpt_llm_client.extract_facts(metadata)
            facts_by_document[document['file_name']] = facts
            logger.info(f"Declarações factuais extraídas para o documento: {document['file_name']}")

        return facts_by_document

    @staticmethod
    def save_facts(facts_by_document: Dict[str, ExtractedFacts], output_file: str):
        """
        Salva as declarações factuais extraídas em um arquivo JSON.

        Args:
            facts_by_document: Dicionário mapeando nomes de arquivos para ExtractedFacts.
            output_file: Caminho do arquivo de saída.
        """
        serializable_facts = {
            file_name: facts.dict() for file_name, facts in facts_by_document.items()
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_facts, f, ensure_ascii=False, indent=2)
        logger.info(f"Declarações factuais salvas em {output_file}")

def main():
    os.makedirs(FINE_TUNING_DIR, exist_ok=True)

    log_file_path = os.path.join(FINE_TUNING_DIR, "topics_extraction.log")
    llm_logger = LLMLogger(log_file_path)

    extractor = FactExtractor(llm_logger)
    facts_by_document = extractor.extract_facts_from_documents(PRE_PROCESSING_FOLDER)

    output_file = os.path.join(FINE_TUNING_DIR, "extracted_facts.json")
    extractor.save_facts(facts_by_document, output_file)

    logger.info("Extração de declarações factuais concluída")
    logger.info(f"Log de interações com o LLM salvo em: {log_file_path}")

if __name__ == "__main__":
    main()