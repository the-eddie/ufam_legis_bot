import json
import os
import random
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from tqdm import tqdm
from pydantic import BaseModel, Field

from ufam_legis_bot.config import *
from ufam_legis_bot.fine_tunning.common import logger, LLMLogger, ExtractedFacts, FactualStatement, DocumentProcessor


class QuestionAnswerPair(BaseModel):
    """Modelo para representar um par de pergunta e resposta."""
    question: str = Field(..., description="Pergunta gerada")
    answer: str = Field(..., description="Resposta correspondente à pergunta")
    document_source: str = Field(...,
                                 description="Nome do documento de origem")
    topics: List[str] = Field(default_factory=list,
                              description="Tópicos relacionados à pergunta")


class GeneratedQAPairs(BaseModel):
    """Modelo para representar múltiplos pares de pergunta e resposta gerados."""
    pairs: List[QuestionAnswerPair] = Field(
        ..., description="Lista de pares de pergunta e resposta gerados")


class SyntheticDatasetGenerator:
    """Gerador de conjunto de dados sintético para fine-tuning."""

    def __init__(self, llm_logger: LLMLogger):
        """
        Inicializa o gerador de conjunto de dados sintético.

        Args:
            llm_logger: Instância do LLMLogger para registrar interações.
        """
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.llm_logger = llm_logger
        self.extracted_facts: Dict[str, ExtractedFacts] = {}
        self.generated_pairs: List[QuestionAnswerPair] = []
        self.used_pairs_for_refinement: set = set()
        self.document_processor = DocumentProcessor()

    def load_extracted_facts(self, file_path: str) -> None:
        """
        Carrega as declarações factuais extraídas de um arquivo JSON.

        Args:
            file_path: Caminho do arquivo JSON contendo as declarações factuais.
        """
        logger.info(f"Carregando declarações factuais de {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.extracted_facts = {
            file_name: ExtractedFacts(**facts_data)
            for file_name, facts_data in data.items()
        }
        logger.info(f"Carregadas declarações factuais de {
                    len(self.extracted_facts)} documentos")

    def generate_question_answer_pairs(self, document: Dict[str, Any], facts: List[FactualStatement]) -> List[QuestionAnswerPair]:
        """
        Gera pares de pergunta e resposta usando GPT-4o-mini com Structured Outputs.

        Args:
            document: Documento contendo o texto completo e metadados.
            facts: Lista de declarações factuais para gerar os pares de pergunta e resposta.

        Returns:
            Uma lista de pares de pergunta e resposta gerados.
        """
        system_prompt = """Você é um especialista em legislação acadêmica da Universidade Federal do Amazonas (UFAM).
        Sua tarefa é gerar perguntas e respostas relevantes com base em declarações factuais fornecidas e no contexto completo do documento.
        Siga estas diretrizes:

        1. Crie uma pergunta clara, específica e relevante para cada declaração factual fornecida.
        2. As perguntas devem ser formuladas como se fossem feitas por estudantes, professores ou funcionários da universidade. Elas devem refletir dúvidas reais que essas pessoas podem ter no dia-a-dia e gostariam da ajuda de aum assistente virtual para responder.
        3. As respostas devem ser precisas, baseadas nas declarações fornecidas e no contexto do documento.
        4. Varie o nível de complexidade das perguntas, incluindo tanto questões simples quanto mais elaboradas.
        5. Inclua perguntas sobre procedimentos, prazos, requisitos, exceções e definições importantes mencionadas nas declarações.
        6. Os tópicos devem refletir as principais áreas temáticas abordadas na pergunta e resposta.
        7. Evite criar perguntas que possam ser respondidas com um simples "sim" ou "não".
        8. Sempre que apropriado, inclua referências específicas a artigos, seções ou parágrafos mencionados nas declarações ou no documento.
        9. Gere as respostas sob o ponto de vista de um assistente útil e amigável.

        Exemploes de perguntas ruins:
        [ "Como a Câmara de Ensino de Graduação lidará com casos omissos na Resolução 070/2011?",
          "O que acontece se os requisitos da transferência ex officio não forem atendidos conforme a Resolução 070/2011?".
          "Como é definida a afinidade entre o curso de origem e o de destinação segundo a Resolução 070/2011?" ]

        Exemploes de perguntas boas:
        [ "Qual é o procedimento que um aluno deve seguir para solicitar afinidade de curso na UFAM?",
          "Qual é a limitação para a atuação de voluntários nas atividades de docência?".
          "O que acontece se um aluno realizar atividades acadêmicas sem estar oficialmente matriculado?" ]

        Lembre-se: seu objetivo é criar um conjunto diversificado de perguntas e respostas que cubram amplamente o conteúdo da legislação acadêmica da UFAM."""

        metadata = document.get('metadata', {})
        original_file_names = metadata.get('original_file_names', [])
        anchor_texts = metadata.get('anchor_texts', [])

        user_prompt = f"""Com base no seguinte documento da legislação acadêmica da UFAM e nas declarações factuais fornecidas,
        gere um par de pergunta e resposta relevante para cada declaração factual:

        Documento:
        {document.get('text_content', '')[:100000]}

        Informações adicionais:
        * Nomes dos arquivos originais: {', '.join(original_file_names) if isinstance(original_file_names, list) else str(original_file_names)}
        * Textos âncora: {', '.join(anchor_texts) if isinstance(anchor_texts, list) else str(anchor_texts)}

        Declarações factuais:
        {' '.join([f"- {fact.statement} (Contexto: {fact.context})" for fact in facts])}

        Gere um par de pergunta e resposta para cada declaração factual, explorando aspectos importantes destas declarações e do documento, seguindo as diretrizes fornecidas."""

        try:
            completion = self.client.beta.chat.completions.parse(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=GeneratedQAPairs,
            )

            generated_pairs = completion.choices[0].message.parsed.pairs
            self.llm_logger.log_interaction(f"System: {system_prompt}\nUser: {user_prompt}", str(generated_pairs))

            # Check if the generation was cut off
            if completion.choices[0].finish_reason == "length":
                logger.warning("A geração foi interrompida devido ao limite máximo de tokens. Continuando a geração...")
                remaining_facts = len(facts) - len(generated_pairs)
                if remaining_facts > 0:
                    # Generate the remaining pairs
                    additional_pairs = self.generate_question_answer_pairs(document, facts[len(generated_pairs):])
                    generated_pairs.extend(additional_pairs)

            return [
                QuestionAnswerPair(
                    question=pair.question,
                    answer=pair.answer,
                    document_source=document['file_name'],
                    topics=pair.topics
                )
                for pair in generated_pairs
            ]
        except Exception as e:
            logger.error(
                f"Erro ao gerar pares de pergunta e resposta: {str(e)}")
            return []

    def generate_initial_dataset(self) -> None:
        """
        Gera o conjunto de dados inicial com um par de pergunta e resposta por fato.
        """
        logger.info("Iniciando geração do conjunto de dados inicial")
        documents = self.document_processor.load_documents(
            PRE_PROCESSING_FOLDER)

        for document in tqdm(documents, desc="Processando documentos"):
            document_name = document['file_name']
            facts = self.extracted_facts[document_name].facts

            new_pairs = self.generate_question_answer_pairs(document, facts)
            self.generated_pairs.extend(new_pairs)

            if len(self.generated_pairs) % SAVE_INTERVAL == 0:
                self._save_intermediate_results()

        logger.info(f"Geração inicial concluída. Total de pares gerados: {
                    len(self.generated_pairs)}")

    def refine_dataset(self, target_pairs: int, batch_size: int = 10) -> None:
        """
        Refina o conjunto de dados para atingir o número alvo de pares.

        Args:
            target_pairs: Número alvo de pares de pergunta e resposta.
            batch_size: Tamanho do lote para geração e salvamento.
        """
        logger.info(f"Iniciando refinamento do conjunto de dados para atingir {
                    target_pairs} pares")
        initial_pairs_count = len(self.generated_pairs)
        pbar = tqdm(total=target_pairs - initial_pairs_count,
                    desc="Refinando pares de pergunta e resposta")

        while len(self.generated_pairs) < target_pairs:
            base_pairs = random.sample(self.generated_pairs[:initial_pairs_count], min(
                batch_size, target_pairs - len(self.generated_pairs)))
            new_pairs = self.generate_refined_pairs(base_pairs)

            for pair in new_pairs:
                if len(self.generated_pairs) < target_pairs:
                    self.generated_pairs.append(pair)
                    pbar.update(1)

            if len(self.generated_pairs) % SAVE_INTERVAL == 0:
                self._save_intermediate_results()

        pbar.close()
        logger.info(f"Refinamento concluído. Total de pares após refinamento: {
                    len(self.generated_pairs)}")

    def generate_refined_pairs(self, base_pairs: List[QuestionAnswerPair]) -> List[QuestionAnswerPair]:
        """
        Gera pares refinados de pergunta e resposta baseados em pares existentes.

        Args:
            base_pairs: Lista de pares de pergunta e resposta base para refinamento.

        Returns:
            Uma lista de novos pares de pergunta e resposta refinados.
        """
        system_prompt = """Você é um especialista em legislação acadêmica da Universidade Federal do Amazonas (UFAM).
        Sua tarefa é gerar novas perguntas e respostas baseadas em pares existentes, mudando a perspectiva ou expandindo o tópico.
        Siga estas diretrizes:

        1. Crie uma nova pergunta que explore um aspecto diferente ou uma perspectiva alternativa do tópico original.
        2. A nova pergunta deve ser substancialmente diferente da original, mas ainda relacionada ao mesmo tema geral.
        3. As respostas devem ser precisas e coerentes com a nova pergunta, mantendo a integridade da informação original.
        4. Varie o nível de complexidade, incluindo tanto questões mais simples quanto mais elaboradas.
        5. Considere diferentes perspectivas (estudantes, professores, funcionários) ao formular as novas perguntas.
        6. Evite criar perguntas que possam ser respondidas com um simples "sim" ou "não".
        7. Mantenha a relevância para o contexto acadêmico da UFAM.

        Lembre-se: seu objetivo é expandir e diversificar o conjunto de perguntas e respostas, mantendo a precisão e relevância para a legislação acadêmica da UFAM."""

        user_prompt = "Com base nos seguintes pares de pergunta e resposta, gere novos pares relacionados, mas com perspectivas ou aspectos diferentes:\n\n"
        for pair in base_pairs:
            user_prompt += f"Pergunta original: {
                pair.question}\nResposta original: {pair.answer}\n\n"
        user_prompt += "Gere um novo par de pergunta e resposta para cada par fornecido, seguindo as diretrizes fornecidas."

        try:
            completion = self.client.beta.chat.completions.parse(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=GeneratedQAPairs,
            )

            refined_pairs = completion.choices[0].message.parsed.pairs
            self.llm_logger.log_interaction(f"System: {system_prompt}\nUser: {
                                            user_prompt}", str(refined_pairs))

            return [
                QuestionAnswerPair(
                    question=pair.question,
                    answer=pair.answer,
                    document_source=base_pairs[i].document_source,
                    topics=pair.topics
                )
                for i, pair in enumerate(refined_pairs)
            ]
        except Exception as e:
            logger.error(
                f"Erro ao gerar pares refinados de pergunta e resposta: {str(e)}")
            return []

    def _save_intermediate_results(self) -> None:
        """Salva resultados intermediários em um arquivo JSON."""
        output_file = os.path.join(
            FINE_TUNING_DIR, "synthetic_dataset_intermediate.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([pair.dict() for pair in self.generated_pairs],
                      f, ensure_ascii=False, indent=2)
        logger.info(f"Resultados intermediários salvos em {output_file}")

    def save_final_dataset(self) -> None:
        """Salva o conjunto de dados final em um arquivo JSON."""
        output_file = SYNTHETIC_DATASET_OUTPUT_FILE
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([pair.dict() for pair in self.generated_pairs],
                      f, ensure_ascii=False, indent=2)
        logger.info(f"Conjunto de dados final salvo em {output_file}")


def main():
    """Função principal para executar a geração do conjunto de dados sintético."""
    os.makedirs(FINE_TUNING_DIR, exist_ok=True)

    log_file_path = os.path.join(
        FINE_TUNING_DIR, "synthetic_dataset_generation.log")
    llm_logger = LLMLogger(log_file_path)

    generator = SyntheticDatasetGenerator(llm_logger)

    extracted_facts_file = os.path.join(
        FINE_TUNING_DIR, "extracted_facts.json")
    generator.load_extracted_facts(extracted_facts_file)

    # Primeira parte: gerar um par por fato
    generator.generate_initial_dataset()

    # Segunda parte: refinar o conjunto de dados até atingir o número alvo
    generator.refine_dataset(NUM_PAIRS_TO_GENERATE, BATCH_SIZE)

    generator.save_final_dataset()

    logger.info("Geração do conjunto de dados sintético concluída")
    logger.info(f"Log de interações com o LLM salvo em: {log_file_path}")


if __name__ == "__main__":
    main()
