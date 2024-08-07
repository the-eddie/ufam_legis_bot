import os
import logging
import hashlib
from typing import List, Tuple
from urllib.parse import unquote

import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from requests.exceptions import RequestException, Timeout

from ufam_legis_bot.config import *

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

class WebCrawler:
    """
    Web crawler para buscar e baixar arquivos de páginas web.
    """

    def __init__(self, timeout: int = TIMEOUT, max_retries: int = MAX_RETRIES):
        """
        Inicializa o WebCrawler.

        Args:
            timeout: Tempo limite para requisições em segundos.
            max_retries: Número máximo de tentativas para cada download.
        """
        self.session = requests.Session()
        self.timeout = timeout
        self.max_retries = max_retries

    def get_file_links(self, url: str, query_selector: str, allowed_extensions: List[str]) -> List[Tuple[str, str]]:
        """
        Busca links de arquivos de uma URL que correspondem aos critérios especificados.

        Args:
            url: A URL da página web para crawling.
            query_selector: O seletor para localizar o conteúdo relevante.
            allowed_extensions: Lista de extensões de arquivo permitidas.

        Returns:
            Uma lista de tuplas contendo (texto de âncora, URL do arquivo).
        """
        logging.info(f"Buscando os links de arquivos em {url}")
        try:
            response = self.session.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            selected_content = soup.select(query_selector)
            if not selected_content:
                logging.warning(f"Nenhum conteúdo encontrado para o seletor: {query_selector}")
                return []

            file_links = []
            for element in selected_content:
                links = element.find_all('a', href=True)
                for link in links:
                    href = link['href']
                    if any(href.lower().endswith(ext) for ext in allowed_extensions):
                        full_url = urljoin(url, href)
                        file_links.append((link.text.strip(), full_url))

            logging.info(f"Encontrados {len(file_links)} links de arquivos")
            return file_links
        except requests.RequestException as e:
            logging.error(f"Erro ao buscar URL {url}: {str(e)}")
            return []

    def download_files(self, file_links: List[Tuple[str, str]], download_folder: str) -> None:
        """
        Baixa arquivos dos links fornecidos e os salva na pasta especificada.

        Args:
            file_links: Uma lista de tuplas contendo (texto âncora, URL do arquivo).
            download_folder: O caminho da pasta onde os arquivos serão salvos.
        """
        logging.info(f"Baixando arquivos em {download_folder}")
        os.makedirs(download_folder, exist_ok=True)

        df = pd.DataFrame(columns=['Anchor Text', 'Link', 'File Name', 'Original File Name', 'MD5 Hash'])

        for file_counter, (anchor_text, link) in enumerate(file_links, start=1):
            if link in df['Link'].values:
                df = self._update_duplicate_link(df, link, anchor_text)
                continue

            file_content = self._download_with_retry(link)
            if file_content is None:
                logging.warning(f"Falha ao baixar {link} após {self.max_retries} tentativas. Pulando para o próximo arquivo.")
                continue

            original_file_name = os.path.basename(unquote(link))
            file_name = f"file_{file_counter:04d}.pdf"
            file_path = os.path.join(download_folder, file_name)

            with open(file_path, 'wb') as f:
                f.write(file_content)

            md5_hash = self._calculate_md5(file_path)

            if md5_hash in df['MD5 Hash'].values:
                df = self._handle_duplicate_file(df, file_path, md5_hash, anchor_text, link)
            else:
                df = self._add_new_file_entry(df, anchor_text, link, file_name, original_file_name, md5_hash)
                logging.info(f"Downloaded: {file_name}")

        csv_path = os.path.join(download_folder, 'download_summary.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8')
        logging.info(f"CSV salvo em {csv_path}")

    def _download_with_retry(self, url: str) -> bytes:
        """
        Tenta baixar um arquivo com um número máximo de tentativas.

        Args:
            url: A URL do arquivo para download.

        Returns:
            O conteúdo do arquivo, ou None se o download falhar após todas as tentativas.
        """
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                return response.content
            except (RequestException, Timeout) as e:
                logging.warning(f"Tentativa {attempt + 1} falhou para {url}: {str(e)}")
                if attempt == self.max_retries - 1:
                    logging.error(f"Todas as tentativas falharam para {url}")
                    return None
        return None

    @staticmethod
    def _update_duplicate_link(df: pd.DataFrame, link: str, anchor_text: str) -> pd.DataFrame:
        """
        Atualiza a entrada para um link duplicado.

        Args:
            df: O DataFrame contendo as informações dos arquivos.
            link: O link duplicado.
            anchor_text: O texto âncora do link duplicado.

        Returns:
            O DataFrame atualizado.
        """
        mask = df['Link'] == link
        df.loc[mask, 'Anchor Text'] += f"; {anchor_text}"
        df.loc[mask, 'Link'] += f"; {link}"
        logging.info(f"Link duplicado atualizado: {link}")
        return df

    @staticmethod
    def _handle_duplicate_file(df: pd.DataFrame, file_path: str, md5_hash: str, anchor_text: str, link: str) -> pd.DataFrame:
        """
        Assegura que arquivos duplicados são removidos.

        Args:
            df: O DataFrame contendo as informações dos arquivos.
            file_path: O caminho do arquivo duplicado.
            md5_hash: O hash MD5 do arquivo duplicado.
            anchor_text: O texto âncora do link do arquivo duplicado.
            link: O link do arquivo duplicado.

        Returns:
            O DataFrame atualizado.
        """
        os.remove(file_path)
        mask = df['MD5 Hash'] == md5_hash
        df.loc[mask, 'Anchor Text'] += f"; {anchor_text}"
        df.loc[mask, 'Link'] += f"; {link}"
        logging.info(f"Arquivo duplicado removido: {os.path.basename(file_path)}")
        return df

    @staticmethod
    def _add_new_file_entry(df: pd.DataFrame, anchor_text: str, link: str, file_name: str, original_file_name: str, md5_hash: str) -> pd.DataFrame:
        """
        Adiciona uma nova entrada para um arquivo único.

        Args:
            df: O DataFrame contendo as informações dos arquivos.
            anchor_text: O texto âncora do link do arquivo.
            link: O link do arquivo.
            file_name: O nome do arquivo salvo.
            original_file_name: O nome original do arquivo.
            md5_hash: O hash MD5 do arquivo.

        Returns:
            O DataFrame atualizado com a nova entrada.
        """
        new_row = pd.DataFrame({
            'Anchor Text': [anchor_text],
            'Link': [link],
            'File Name': [file_name],
            'Original File Name': [original_file_name],
            'MD5 Hash': [md5_hash]
        })
        return pd.concat([df, new_row], ignore_index=True)

    @staticmethod
    def _calculate_md5(file_path: str) -> str:
        """
        Calcula o hash MD5 de um arquivo.

        Args:
            file_path: O caminho do arquivo.

        Returns:
            O hash MD5 do arquivo.
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def crawl_and_download(self, source_link: str, query_selector: str, file_extensions: List[str], download_folder: str) -> None:
        """
        Rastreia uma página web, encontra links de arquivos e faz o download.

        Args:
            source_link: A URL da página web para buscar os links.
            query_selector: O seletor CSS para localizar o conteúdo relevante.
            file_extensions: Lista de extensões de arquivo permitidas.
            download_folder: O caminho da pasta onde os arquivos serão salvos.
        """
        logging.info("Iniciando processo de crawling e download")
        file_links = self.get_file_links(source_link, query_selector, file_extensions)
        if file_links:
            self.download_files(file_links, download_folder)
        else:
            logging.warning("Nenhum arquivo encontrado para download")
        logging.info("Processo de crawling e download concluído")

def main() -> None:
    """
    Cria uma instância de WebCrawler e faz download de arquivos de legislação no site da UFAM
    """
    try:
        crawler = WebCrawler(timeout=TIMEOUT, max_retries=MAX_RETRIES)
        crawler.crawl_and_download(
            source_link=SOURCE_LINK,
            query_selector=QUERY_SELECTOR,
            file_extensions=FILE_EXTENSIONS,
            download_folder=INPUT_FOLDER
        )
    except Exception as e:
        logging.error(f"Ocorreu um erro durante o crawling: {str(e)}")
        raise

if __name__ == "__main__":
    main()