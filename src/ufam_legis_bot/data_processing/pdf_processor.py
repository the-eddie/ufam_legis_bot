import os
import json
from datetime import datetime
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
    original_file_name: str
    anchor_text: str

class PDFProcessor:
    def __init__(self, file_path: str, pdf_info: PDFInfo):
        self.file_path = file_path
        self.pdf_info = pdf_info

    def process(self) -> Dict:
        """Processa um único arquivo PDF."""
        try:
            text, num_pages, ocr_tool = self._extract_text()
            clean_text = self._clean_text(text)

            return {
                "file_name": self.pdf_info.file_name,
                "text_content": clean_text,
                "metadata": {
                    "page_count": num_pages,
                    "processing_date": datetime.now().strftime("%Y-%m-%d"),
                    "ocr_tool": ocr_tool,
                    "original_file_name": self.pdf_info.original_file_name,
                    "anchor_text": self.pdf_info.anchor_text
                }
            }
        except Exception as e:
            logger.error(f"Erro ao processar {self.file_path}: {e}")
            return {}

    def _extract_text(self) -> Tuple[str, int, str]:
        """
        Extrai texto de um arquivo PDF.

        Returns:
            Uma tupla contendo o texto extraído, número de páginas e a ferramenta de OCR utilizada.
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
            return text, num_pages, "PyPDF2"
        except Exception as e:
            logger.error(f"Erro ao extrair texto de {self.file_path}: {e}")
            raise

    def _perform_ocr(self) -> Tuple[str, int, str]:
        """
        Realiza OCR em um arquivo PDF.

        Returns:
            Uma tupla contendo o texto extraído por OCR, número de páginas e a ferramenta de OCR utilizada.
        """
        try:
            images = convert_from_path(self.file_path)
            total_text = ""
            for image in images:
                image_np = np.array(image)
                gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                clean_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                text = pytesseract.image_to_string(Image.fromarray(clean_image), lang='por')
                total_text += text + "\n"

            logger.info(f"OCR realizado com sucesso para {self.file_path}")
            return total_text, len(images), "Tesseract OCR"
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
    def __init__(self, input_folder: str, output_folder: str, csv_file: str):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.csv_file = csv_file

    def run(self) -> None:
        """Executa o processamento completo dos arquivos PDF."""
        logger.info("Iniciando o processamento dos arquivos PDF.")
        self._create_output_folder()
        df = self._read_csv()
        self._process_all_files(df)
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
            df = pd.read_csv(self.csv_file)
            logger.info(f"CSV lido com sucesso: {self.csv_file}")
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
        files = df['File Name'].tolist()

        for file in tqdm(files, desc="Processando arquivos"):
            logger.info(f"Processando arquivo: {file}")
            pdf_info = PDFInfo(
                file_name=file,
                original_file_name=df[df['File Name'] == file]['Original File Name'].iloc[0],
                anchor_text=df[df['File Name'] == file]['Anchor Text'].iloc[0]
            )
            processor = PDFProcessor(os.path.join(self.input_folder, file), pdf_info)
            result = processor.process()
            if result:
                self._save_json(result, os.path.splitext(file)[0])

    def _save_json(self, data: Dict, file_name: str) -> None:
        """
        Salva os dados processados em um arquivo JSON.

        Args:
            data: Dicionário contendo os dados a serem salvos.
            file_name: Nome do arquivo (sem extensão) para salvar o JSON.
        """
        output_path = os.path.join(self.output_folder, f"{file_name}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Arquivo JSON salvo: {output_path}")

def main():
    """
    Função principal para executar o processamento em lote de PDFs.
    """
    try:
        processor = PDFBatchProcessor(INPUT_FOLDER, OUTPUT_FOLDER, CSV_FILE)
        processor.run()
    except Exception as e:
        logger.error(f"Ocorreu um erro durante o processamento de PDF: {str(e)}")
        raise

if __name__ == "__main__":
    main()