import os
import json
from datetime import datetime
import time
from typing import Dict, Tuple, List
from dataclasses import dataclass

import pandas as pd
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import cv2
import numpy as np
from pypdf import PdfReader
import logging
from tqdm import tqdm

from ufam_legis_bot.config import *

# Configuração do logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

@dataclass
class PDFInfo:
    file_name: str
    original_file_names: List[str]
    anchor_texts: List[str]
    links: List[str]

class PDFProcessor:
    def __init__(self, file_path: str, pdf_info: PDFInfo):
        self.file_path = file_path
        self.pdf_info = pdf_info

    def process(self) -> Dict:
        """Processa um único arquivo PDF."""
        try:
            start_time = time.time()
            text, num_pages, ocr_tool, ocr_confidence = self._extract_text()
            clean_text = self._clean_text(text)
            processing_time = time.time() - start_time

            word_count = len(clean_text.split())
            char_count = len(clean_text)

            return {
                "file_name": self.pdf_info.file_name,
                "text_content": clean_text,
                "metadata": {
                    "page_count": num_pages,
                    "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "ocr_tool": ocr_tool,
                    "original_file_names": self.pdf_info.original_file_names,
                    "anchor_texts": self.pdf_info.anchor_texts,
                    "links": self.pdf_info.links,
                    "word_count": word_count,
                    "char_count": char_count,
                    "processing_time": round(processing_time, 2),
                    "ocr_confidence": round(ocr_confidence, 2)
                }
            }
        except Exception as e:
            logger.error(f"Erro ao processar {self.file_path}: {e}")
            return {
                "file_name": self.pdf_info.file_name,
                "error": str(e),
                "metadata": {
                    "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "original_file_names": self.pdf_info.original_file_names,
                    "anchor_texts": self.pdf_info.anchor_texts,
                    "links": self.pdf_info.links
                }
            }

    def _extract_text(self) -> Tuple[str, int, str, float]:
        """
        Extrai texto de um arquivo PDF.

        Returns:
            Uma tupla contendo o texto extraído, número de páginas, a ferramenta de OCR utilizada e a confiança do OCR.
        """
        try:
            with open(self.file_path, 'rb') as file:
                reader = PdfReader(file)
                num_pages = len(reader.pages)
                text = "".join(page.extract_text() or "" for page in reader.pages)

            if not text.strip():
                logger.info(f"Texto não extraído diretamente. Tentando OCR para {self.file_path}")
                return self._perform_ocr()

            logger.info(f"Texto extraído diretamente de {self.file_path}")
            return text, num_pages, "PyPDF2", 100.0
        except Exception as e:
            logger.error(f"Erro ao extrair texto de {self.file_path}: {e}")
            raise

    def _perform_ocr(self) -> Tuple[str, int, str, float]:
        """
        Realiza OCR em um arquivo PDF.

        Returns:
            Uma tupla contendo o texto extraído por OCR, número de páginas, a ferramenta de OCR utilizada e a confiança média do OCR.
        """
        try:
            images = convert_from_path(self.file_path)
            total_text = ""
            total_confidence = 0
            for image in images:
                image_np = np.array(image)
                gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                clean_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                ocr_result = pytesseract.image_to_data(Image.fromarray(clean_image), lang='por', output_type=pytesseract.Output.DICT)

                page_text = " ".join(ocr_result['text'])
                total_text += page_text + "\n"

                confidences = [int(conf) for conf in ocr_result['conf'] if conf != '-1']
                total_confidence += sum(confidences) / len(confidences) if confidences else 0

            average_confidence = total_confidence / len(images) if images else 0
            logger.info(f"OCR realizado com sucesso para {self.file_path}")
            return total_text, len(images), "Tesseract OCR", average_confidence
        except Exception as e:
            logger.error(f"Erro ao realizar OCR em {self.file_path}: {e}")
            raise

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Limpa o texto extraído.

        Args:
            text: O texto a ser limpo.

        Returns:
            O texto limpo.
        """
        text = ''.join(char for char in text if char.isprintable() or char.isspace())

        # Normalizar espaços em branco
        text = ' '.join(text.split())

        # Corrige alguns erros comuns de OCR
        common_errors = {'l': '1', 'O': '0', '|': 'I'}
        for error, correction in common_errors.items():
            text = text.replace(error, correction)

        return text

class PDFBatchProcessor:
    def __init__(self, input_folder: str, output_folder: str, download_summary_csv: str):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.download_summary_csv = download_summary_csv
        self.processing_summary = pd.DataFrame(columns=[
            'PDF File Name', 'Original File Names', 'Output File Name', 'Processing Date',
            'Extraction Method', 'Page Count', 'Word Count', 'Character Count',
            'Processing Time', 'Error Message', 'OCR Confidence', 'Metadata Fields',
            'Anchor Texts', 'Links'
        ])

    def run(self) -> None:
        """Executa o processamento completo dos arquivos PDF."""
        logger.info("Iniciando o processamento dos arquivos PDF.")
        self._create_output_folder()
        df = self._read_csv()
        self._process_all_files(df)
        self._save_processing_summary()
        logger.info("Processamento concluído.")

    def _create_output_folder(self) -> None:
        """Cria a pasta de saída se ela não existir."""
        os.makedirs(self.output_folder, exist_ok=True)
        logger.info(f"Pasta de saída verificada: {self.output_folder}")

    def _read_csv(self) -> pd.DataFrame:
        """
        Lê o arquivo CSV e retorna um DataFrame.

        Returns:
            Um DataFrame contendo as informações do CSV.
        """
        try:
            df = pd.read_csv(self.download_summary_csv)
            logger.info(f"CSV lido com sucesso: {self.download_summary_csv}")
            return df
        except Exception as e:
            logger.error(f"Erro ao ler o arquivo CSV: {e}")
            raise

    def _process_all_files(self, df: pd.DataFrame) -> None:
        """
        Processa todos os arquivos PDF listados no DataFrame.

        Args:
            df: DataFrame contendo informações sobre os arquivos a serem processados.
        """
        unique_files = df[df['Status'] == 'Baixado']['File Name'].unique()

        # Clean a list by removing NaNs and duplicates
        def clean_list(data):
            return list(dict.fromkeys(filter(lambda x: x == x, data)))

        for file in tqdm(unique_files, desc="Processando arquivos"):
            logger.info(f"Processando arquivo: {file}")
            file_entries = df[df['File Name'] == file]

            original_file_names = clean_list(file_entries['Original File Name'].tolist())
            anchor_texts = clean_list(file_entries['Anchor Text'].tolist())
            links = clean_list(file_entries['Link'].tolist())

            # Create the PDFInfo object with cleaned lists
            pdf_info = PDFInfo(
                file_name=file,
                original_file_names=original_file_names,
                anchor_texts=anchor_texts,
                links=links
            )

            processor = PDFProcessor(os.path.join(self.input_folder, file), pdf_info)
            result = processor.process()
            if result:
                output_file_name = self._save_json(result, os.path.splitext(file)[0])
                self._update_processing_summary(result, output_file_name)

    def _save_json(self, data: Dict, file_name: str) -> str:
        """
        Salva os dados processados em um arquivo JSON.

        Args:
            data: Dicionário contendo os dados a serem salvos.
            file_name: Nome do arquivo (sem extensão) para salvar o JSON.

        Returns:
            O nome do arquivo JSON salvo.
        """
        output_file_name = f"{file_name}.json"
        output_path = os.path.join(self.output_folder, output_file_name)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Arquivo JSON salvo: {output_path}")
        return output_file_name

    def _update_processing_summary(self, result: Dict, output_file_name: str) -> None:
        """
        Atualiza o resumo do processamento com os resultados de um arquivo.

        Args:
            result: Dicionário contendo os resultados do processamento.
            output_file_name: Nome do arquivo JSON de saída.
        """
        metadata = result.get('metadata', {})
        error_message = result.get('error', '')

        new_row = pd.DataFrame({
            'PDF File Name': [result['file_name']],
            'Original File Names': [', '.join(metadata.get('original_file_names', []))],
            'Output File Name': [output_file_name],
            'Processing Date': [metadata.get('processing_date', '')],
            'Extraction Method': [metadata.get('ocr_tool', '')],
            'Page Count': [metadata.get('page_count', 0)],
            'Word Count': [metadata.get('word_count', 0)],
            'Character Count': [metadata.get('char_count', 0)],
            'Processing Time': [round(metadata.get('processing_time', 0), 2)],
            'Error Message': [error_message],
            'OCR Confidence': [round(metadata.get('ocr_confidence', 0), 2)],
            'Metadata Fields': [', '.join(metadata.keys())],
            'Anchor Texts': [', '.join(metadata.get('anchor_texts', []))],
            'Links': [', '.join(metadata.get('links', []))]
        })

        self.processing_summary = pd.concat([self.processing_summary, new_row], ignore_index=True)

    def _save_processing_summary(self) -> None:
        """Salva o resumo do processamento em um arquivo CSV."""
        self.processing_summary.to_csv(PRE_PROCESSING_SUMMARY_CSV_FILE, index=False, encoding='utf-8')
        logger.info(f"Resumo do processamento salvo em: {PRE_PROCESSING_SUMMARY_CSV_FILE}")

def main():
    """
    Função principal para executar o processamento em lote de PDFs.
    """
    try:
        processor = PDFBatchProcessor(CRAWLING_FOLDER, PRE_PROCESSING_FOLDER, DOWNLOAD_SUMMARY_CSV_FILE)
        processor.run()
    except Exception as e:
        logger.error(f"Ocorreu um erro durante o processamento de PDF: {str(e)}")
        raise

if __name__ == "__main__":
    main()