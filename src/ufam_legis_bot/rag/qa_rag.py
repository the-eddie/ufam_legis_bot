import logging
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

from ufam_legis_bot.config import *

# Configuração do logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class LLaMARAGSystem:
    """
    Sistema de Perguntas e Respostas usando LLaMA 3 e RAG com Chroma DB.
    """

    def __init__(self, model_name: str = "openai-community/gpt2"): #meta-llama/Llama-2-7b-chat-hf"):
        """
        Inicializa o sistema de Q&A.

        Args:
            model_name: Nome do modelo LLaMA no Hugging Face.
        """
        logger.info("Inicializando o sistema LLaMA RAG")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, max_length=512)

        self.chroma_client = PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
        self.collection = self.chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)

        self.sentence_embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=SENTENCE_TRANSFORMER_MODEL
        )

        logger.info("Sistema LLaMA RAG inicializado com sucesso")

    def retrieve_relevant_context(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Recupera o contexto relevante do Chroma DB com base na consulta.

        Args:
            query: A pergunta do usuário.
            n_results: Número de resultados a serem recuperados.

        Returns:
            Lista de documentos relevantes.
        """
        logger.info(f"Recuperando contexto para a consulta: {query}")
        query_embedding = self.sentence_embedder([query])
        results = self.collection.query(query_embeddings=query_embedding, n_results=n_results)

        documents = []
        for i in range(len(results['ids'][0])):
            doc = {
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i]
            }
            documents.append(doc)

        logger.info(f"Recuperados {len(documents)} documentos relevantes")
        return documents

    def generate_answer(self, query: str, context: List[Dict[str, Any]]) -> str:
        """
        Gera uma resposta para a consulta com base no contexto recuperado.

        Args:
            query: A pergunta do usuário.
            context: Lista de documentos relevantes.

        Returns:
            Resposta gerada pelo modelo.
        """
        logger.info("Gerando resposta com base no contexto recuperado")
        context_text = " ".join([doc['text'] for doc in context])
        prompt = f"Com base no seguinte contexto:\n\n{context_text}\n\nResponda à seguinte pergunta: {query}\n\nResposta:"

        response = self.pipe(prompt, max_length=512, num_return_sequences=1)
        answer = response[0]['generated_text'].split("Resposta:")[-1].strip()

        logger.info("Resposta gerada com sucesso")
        return answer

    def answer_question(self, query: str) -> str:
        """
        Processa uma pergunta e retorna uma resposta.

        Args:
            query: A pergunta do usuário.

        Returns:
            Resposta gerada pelo sistema.
        """
        logger.info(f"Processando pergunta: {query}")
        relevant_context = self.retrieve_relevant_context(query)
        answer = self.generate_answer(query, relevant_context)
        logger.info("Resposta gerada e pronta para retorno")
        return answer

def main():
    """
    Função para demonstrar o uso do sistema de Q&A.
    """
    try:
        qa_system = LLaMARAGSystem()

        while True:
            query = input("Digite sua pergunta (ou 'sair' para encerrar): ")
            if query.lower() == 'sair':
                break

            answer = qa_system.answer_question(query)
            print(f"Resposta: {answer}\n")

    except Exception as e:
        logger.error(f"Ocorreu um erro durante a execução do sistema de Q&A: {str(e)}")
        raise

if __name__ == "__main__":
    main()