"""
title: Document Optical Reconstruction Kit (DORK)
author: Sawyer Borror <sawyerksu@gmail.com>
author_url: https://github.com/thesawdawg/open-webui-docAI.git
version: 2.3
license: MIT
Dependencies: open_webui, pdf2image
"""
from pydantic import BaseModel, Field
import logging
import requests
from open_webui.config import UPLOAD_DIR, DATA_DIR, WEBUI_URL
import os
from urllib.parse import urlparse, parse_qs, urlencode
from pypdf import PdfReader, PdfWriter, PageObject
from pypdf.generic import (
    DictionaryObject,
    NameObject,
    NumberObject,
    ContentStream,
    RectangleObject,
)
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image
from fastapi import Request
from open_webui.utils.chat import generate_chat_completion
from open_webui.utils.models import get_all_base_models
from open_webui.models.users import Users
from open_webui.models.files import Files, FileForm
import base64
import json
import aiohttp
import re
import fpdf
import uuid
from io import BytesIO
from typing import List, Union
import random
from pathlib import Path

logger = logging.getLogger(__name__)
# Configure logger to output to stdout
# logger.addHandler(logging.StreamHandler())
# logger.setLevel(logging.DEBUG)


class Pipe:
    """Pipe class handles the processing of a message or a file uploaded to the chatbot interface."""

    class Valves(BaseModel):
        """Valves class contains the valves for the pipe."""
        PNG_CONVERT_THREAD_COUNT: int = Field(default=4, description="The number of threads to use when converting the image to a PDF.")
        DOWNLOAD_DIR: str = Field(default=UPLOAD_DIR, description="The directory to use when downloading the file.")
        PNG_DIR: str = Field(default=os.path.join(DATA_DIR, "pngs"), description="The directory to use when saving the PNG images. A sub-directory will be created for each downloaded file that will contain the png files.")
        COMBINED_PDF_DIR: str = Field(default=os.path.join(DATA_DIR, "combined_pdfs"), description="The directory to use when saving the combined PDF.")
        MODEL_CONTEXT_SIZE: int = Field(default=8192, description="The context size to use when generating the chat completion.")
        PNG_DPI: int = Field(default=400, description="The DPI to use when converting the image to a PDF.")


    class UserValves(BaseModel):
        """UserValves class contains the valves for the user."""
        MODEL_ID: str = Field(default="llama3.2-vision:latest")
        STREAM: bool = Field(default=False)
        SYSTEM_PROMPT: str = Field(default="You are a file transcriber. You will carefully analyze and transcribe text within an image or file and respond with the verbatim transcription formatted with markdown. If there is no text within the image or file, provide a brief description of what is within the image or file. Do not respond with any additional text or summaries.")
        PROMPT: str = Field(default="Transcribe the text in the image or file and respond with a verbatim transcription.")
        TEMPERATURE: float = Field(default=0.7)
        MAX_TOKENS: int = Field(default=8192)
        COMPILE_TO_PDF: bool = Field(default=True, description="Whether to compile the PDF with the OCR text.")
        

    def __init__(self):
        """Initializes the Pipe class by setting the valves."""
        self.user_valves = self.UserValves()
        self.valves = self.Valves()


    async def pipe(self, body: dict, __user__: dict, __request__: Request, __event_emitter__=None, __event_call__=None, __files__=None):
        """
        Processes the last user message and handles user action. This function must be named "pipe" for open-webui to process it.

        Args:
            body (dict): The message or file uploaded to the chatbot interface.
            __event_emitter__ (Optional): The event emitter to send the status updates.
            __event_call__ (Optional): The event call to wait for the user action.

        Returns:
            None
        """
        # Get the current UserValves state from the __user__ dictionary
        self.user_valves = __user__['valves']

        # Default custom text commands.
        # This will be the response if we dont have any tasks available to process
        # The commands will be options the user may input to trigger specific actions
        GLOBAL_DEFAULT_RESPONSE = f"""

        **Welcome to the Document Optical Reconstruction Kit (DORK)**
        The input you provided was not recognized as a valid command.

        *Available commands*:
        - /chat [message]
            - Send a message to the chatbot. Nothing fancy, just simple chat.
        
        """

        

        # Ensure our directories exist
        os.makedirs(self.valves.PNG_DIR, exist_ok=True)
        os.makedirs(self.valves.DOWNLOAD_DIR, exist_ok=True)
        os.makedirs(self.valves.COMBINED_PDF_DIR, exist_ok=True)
        
        @staticmethod
        async def update_status(message: str, __event_emitter__=None, done: bool = False):
            """
            Updates the status of the pipe.

            Args:
                message (str): The status message.
                __event_emitter__ (Optional): The event emitter to send the status updates.
                done (bool): Whether the pipe is finished or not.

            Returns:
                None
            """
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": message,
                        "done": done
                    }
                })

        @staticmethod
        async def update_response(message: str, __event_emitter__=None, add_line: bool = True):
            """
            Updates the response of the pipe.

            Args:
                message (str): The response message.
                __event_emitter__ (Optional): The event emitter to send the response updates.
                add_line (bool): Whether to add a line break to the response or not.

            Returns:
                None
            """
            if add_line:
                message = message + "\n"
            if __event_emitter__:
                await __event_emitter__({
                    "type": "message",
                    "data": {
                        "content": message,
                    }
                })

        @staticmethod
        async def action(__event_call__=None, title: str = "Enter your message...", message: str = "Enter your message...", action_type: str = "input", placeholder: str = "Enter your message..."):
            """
            Waits for user action.

            Args:
                __event_call__ (Optional): The event call to wait for the user action.
                title (str): The title of the action.
                message (str): The message to be displayed to the user.
                action_type (str): The type of action to be performed.
                placeholder (str): The placeholder for the input field.

            Returns:
                str: The user action input.
            """
            if __event_call__:
                if action_type == "input":
                    return await __event_call__(
                        {
                            "type": action_type,
                            "data": {
                                "title": title,
                                "message": message,
                                "placeholder": placeholder,
                            },
                        }
                    )
                elif action_type == "file":
                    return await __event_call__(
                        {
                            "type": action_type
                        }
                    )

        @staticmethod
        def download_file(url, output_folder=None):
            """
            Downloads a file from the given URL and saves it to the given output folder.

            Args:
                url (str): The URL of the file to be downloaded.
                output_folder (str): The folder to save the downloaded file in.

            Returns:
                str: The path to the downloaded file.
            """
            output_folder = output_folder or self.valves.DOWNLOAD_DIR
            Path(output_folder).mkdir(parents=True, exist_ok=True)
            if "drive.google.com" in url:
                file_id = extract_google_drive_id(url)
                if not file_id:
                    raise ValueError("Invalid Google Drive URL")
                url = f"https://drive.google.com/uc?export=download&id={file_id}"
            elif "1drv.ms" in url or "onedrive.live.com" in url:
                url = convert_onedrive_url(url)
            response = requests.get(url, stream=True)
            if response.status_code != 200:
                raise Exception(f"Failed to download file. HTTP Status Code: {response.status_code}")
            filename = get_filename_from_url_or_headers(url, response.headers)
            if not filename:
                raise Exception("Unable to determine the filename")
            file_path = Path(output_folder) / filename
            with open(file_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"File downloaded successfully: {file_path}")
            return str(file_path)

        @staticmethod
        def extract_google_drive_id(url):
            """
            Extracts the Google Drive file ID from the given URL.

            Args:
                url (str): The URL containing the Google Drive file ID.

            Returns:
                str: The extracted Google Drive file ID.
            """
            parsed_url = urlparse(url)
            if "id=" in url:
                return parse_qs(parsed_url.query).get("id", [None])[0]
            if "file/d/" in url:
                parts = parsed_url.path.split("/")
                return parts[parts.index("d") + 1] if "d" in parts else None
            return None

        @staticmethod
        def check_google_file_already_downloaded(file_id, output_folder=self.valves.DOWNLOAD_DIR):
            """
            Checks if the Google Drive file has already been downloaded.

            Args:
                file_id (str): The Google Drive file ID.
                output_folder (str): The folder to save the downloaded file in.

            Returns:
                str or False: The file path if the file already exists, False otherwise.
            """
            file_path = Path(output_folder) / f"{file_id}.pdf"
            if file_path.exists():
                print(f"File already downloaded: {file_path}")
                return str(file_path)
            return False

        @staticmethod
        def convert_onedrive_url(url):
            """
            Converts a OneDrive URL to a direct download link.

            Args:
                url (str): The OneDrive URL to be converted.

            Returns:
                str: The direct download link.
            """
            if "1drv.ms" in url:
                response = requests.head(url, allow_redirects=True)
                url = response.url
            if "onedrive.live.com" in url:
                parsed_url = urlparse(url)
                query = parse_qs(parsed_url.query)
                query["download"] = ["1"]
                return f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}?{urlencode(query, doseq=True)}"
            return url

        @staticmethod
        def get_filename_from_url_or_headers(url, headers=None):
            """
            Extracts the filename from the given URL or headers.

            Args:
                url (str): The URL containing the filename.
                headers (dict): The headers containing the filename.

            Returns:
                str: The extracted filename.
            """
            try:
                if not isinstance(url, str):
                    logger.error(f"Invalid URL type: {type(url)}")
                    return None
                    
                r = requests.head(url, allow_redirects=True)

                # The filename often appears in the Content-Disposition header
                content_disp = r.headers.get("Content-Disposition", "")
                # e.g.: 'attachment; filename="example.pdf"; filename*=UTF-8''example.pdf'

                # First try to get filename from Content-Disposition header
                m = re.search(r'filename="([^"]+)"', content_disp)
                if m:
                    filename = m.group(1)
                    return filename

                # If that fails, try to get filename from URL path
                path = urlparse(url).path
                if path:
                    filename = Path(path).name
                    if filename:
                        return filename

                return None

            except Exception as e:
                logger.error(f"Error getting filename from URL {url}: {str(e)}")
                return None

        @staticmethod
        def is_pdf(file_path):
            """
            Checks if the given file is a PDF.

            Args:
                file_path (str): The path to the file to be checked.

            Returns:
                bool: True if the file is a PDF, False otherwise.
            """
            try:
                with open(file_path, "rb") as file:
                    reader = PdfReader(file)
                return True
            except Exception:
                return False

        @staticmethod
        async def convert_pdf_to_pngs(pdf_path, __event_emitter__, output_folder, png_dpi=200, chunk_size=10):
            """
            Converts a PDF file to a series of PNG images in chunks to avoid
            large memory usage or timeouts on very large PDFs.
            
            Args:
                pdf_path (str): Path to the PDF file to be converted.
                __event_emitter__: Some event emitter function to update status (async).
                output_folder (str): Folder to save the PNG images.
                png_dpi (int): DPI setting for the PNG conversion.
                chunk_size (int): How many pages to process at once.

            Returns:
                list: A list of paths to the saved PNG images.
            """
            Path(output_folder).mkdir(parents=True, exist_ok=True)

            # 1) Get the total number of pages in the PDF
            info = pdfinfo_from_path(pdf_path)
            total_pages = info["Pages"]

            page_paths = []
            current_page = 1
            
            # 2) Process in batches
            while current_page <= total_pages:
                end_page = min(current_page + chunk_size - 1, total_pages)
                
                # Convert a range of pages [current_page, end_page]
                page_images = convert_from_path(
                    pdf_path,
                    dpi=png_dpi,
                    first_page=current_page,
                    last_page=end_page,
                    thread_count=4 
                )
                
                for idx, image in enumerate(page_images, start=current_page):
                    page_path = Path(output_folder) / f"page_{idx}.png"
                    image.save(page_path, "PNG")
                    await update_status(
                        f"Converted page {idx}/{total_pages}...", __event_emitter__
                    )
                    page_paths.append(str(page_path))

                current_page += chunk_size  # move to the next chunk

            logger.debug(f"PDF converted to PNGs. Pages saved in: {output_folder}")
            return page_paths

        @staticmethod
        async def convert_image_to_png(image_path):
            """
            Converts an image to a PNG file.

            Args:
                image_path (str): The path to the image file to be converted.

            Returns:
                str: The path to the converted PNG file.
            """
            image = Image.open(image_path)
            png_path = Path(self.valves.PNG_DIR) / (Path(image_path).stem + ".png")
            image.save(png_path, "PNG")
            return str(png_path)
        @staticmethod
        def is_image(file_path):
            """
            Checks if the given file is an image. Allowed filetypes are: png, jpg, jpeg, tiff

            Args:
                file_path (str): The path to the file to be checked.

            Returns:
                bool: True if the file is an image, False otherwise.
            """
            return Path(file_path).suffix.lower() in {'.png', '.jpg', '.jpeg', '.tiff'}

        @staticmethod
        def check_existing_file(filename):
            """
            Checks if the given file already exists in the UPLOAD_DIR.

            Args:
                file_url (str): The URL of the file to be checked.

            Returns:
                str or False: The file path if the file already exists, False otherwise.
            """
            try:
                file_path = Path(UPLOAD_DIR) / filename
                if file_path.exists():
                    logger.debug(f"File exists. Returning file path. {UPLOAD_DIR}/{filename}")
                    return str(file_path)
                logger.debug(f"File does not exist. Returning False.")
                return False
            except Exception as e:
                logger.error(f"Error checking file: {file_url}\nError Message:{e}")
                return None

        @staticmethod
        def convert_image_to_base64(image_path):
            """
            Converts an image to a base64-encoded string.
            """
            logging.info(f"Converting image to base64: {image_path}")
            try:
                with open(image_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                return encoded_string
            except FileNotFoundError:
                logging.error(f"Error: File not found - {image_path}")
                return None
            except Exception as e:
                logging.error(f"Error: {e}")
                return None

        @staticmethod
        def parse_range(s):
            """
            Checks if s matches the pattern: {number}-{number}.
            If left_number < right_number, returns list(range(left_number, right_number+1)).
            Otherwise, returns None.
            """
            pattern = r'^(\d+)-(\d+)$'  # Matches "number-number"
            match = re.match(pattern, s)
            if match:
                left_num = int(match.group(1))
                right_num = int(match.group(2))
                if left_num < right_num:
                    return list(range(left_num, right_num + 1))
            return None

            # Example usage:
            #
            # parse_range("5-10")   # [5, 6, 7, 8, 9, 10]
            # parse_range("10-5")   # None (because left_num >= right_num)
            # parse_range("xyz-123")# None (no match)

        @staticmethod
        def gen_combined_pdf(data_list, output_path=self.valves.COMBINED_PDF_DIR):
            """
            Creates a PDF with 2 pages per item:
            - Page 1: Displays the PNG image.
            - Page 2: Displays the transcription text.

            :param items: A list of dictionaries, each with:
                {
                "png_file_path": <str>,  # path to the PNG file
                "page_number": <int>,    # (optional if you want ordering)
                "transcription_text": <str>
                }
            :param output_path: Path where the combined PDF should be saved.
            """

            try:
                # Sort items by page_number if you need a specific order
                # If you do NOT need ordering, skip the sort.
                items_sorted = sorted(data_list, key=lambda x: x["page_number"])
                
                pdf = fpdf.FPDF(unit="pt")  # Using points to make dimensioning simpler
                pdf.set_auto_page_break(auto=False)  # We control layout explicitly
                
                # Add a Unicode font
                # pdf.add_font('DejaVu', '', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', uni=True)
                pdf.set_font('Arial', size=12)

                # Character replacement map for common Unicode characters
                char_replace = {
                    '—': '-',  # em dash
                    '–': '-',  # en dash
                    '"': '"',  # smart quote
                    '"': '"',  # smart quote
                    ''': "'",  # smart quote
                    ''': "'",  # smart quote
                    '…': '...',  # ellipsis
                    '•': '*',  # bullet
                    '©': '(c)',  # copyright
                    '®': '(R)',  # registered trademark
                    '™': '(TM)',  # trademark
                    '°': ' degrees',  # degree symbol
                    '±': '+/-',  # plus-minus
                    '×': 'x',  # multiplication
                    '÷': '/',  # division
                    '≤': '<=',  # less than or equal
                    '≥': '>=',  # greater than or equal
                    '≠': '!=',  # not equal
                    '≈': '~',  # approximately equal
                    '½': '1/2',  # one half
                    '¼': '1/4',  # one quarter
                    '¾': '3/4',  # three quarters
                }

                page_width = pdf.w - 2 * pdf.l_margin

                for item in items_sorted:
                    png_path = item["png_file_path"]
                    transcription_text = item["transcription_text"]
                    
                    # Replace unsupported characters
                    for old, new in char_replace.items():
                        transcription_text = transcription_text.replace(old, new)
                    
                    # Remove any remaining non-ASCII characters
                    transcription_text = ''.join(char if ord(char) < 128 else '' for char in transcription_text)

                    # ------------------
                    # Page 1: PNG image
                    # ------------------
                    pdf.add_page()

                    # Center the image on the page
                    image_width = 400
                    image_x = (pdf.w - image_width) / 2
                    image_y = (pdf.h - image_width) / 2
                    pdf.image(png_path, x=image_x, y=image_y, w=image_width)
                    pdf.multi_cell(page_width, 20, f"Page {item['page_number']} for: {png_path}")

                    # ---------------------------
                    # Page 2: Transcription Text
                    # ---------------------------
                    pdf.add_page()
                    
                    # Start writing the text from x=50, y=50
                    pdf.set_xy(30, 30)
                    pdf.set_margin(30)
                    
                    pdf.multi_cell(page_width, 20, f"Page {item['page_number']} for: {png_path}")

                    # For multi-line text, use multi_cell
                    pdf.multi_cell(
                        page_width - 60,  # Adjust width to account for margins  
                        12,               # Adjust height to your liking
                        transcription_text
                    )

                # Finally, save the PDF
                pdf.output(output_path)
                return True
            except Exception as e:
                logger.error(f"Error creating PDF: {e}")
                return e

        @staticmethod
        def print_element(element, indent=0, slice_length=50):
            """
            Recursively prints elements. If it's a list or dict, drill down.
            Otherwise, slice if it's a long string.
            """
            prefix = " " * indent  # indentation for structured printing

            if isinstance(element, str):
                # Truncate if needed
                text = element if len(element) <= slice_length else element[:slice_length] + "..."
                print(f"{prefix}- (string) {text}")

            elif isinstance(element, list):
                print(f"{prefix}- (list) [")
                for i, sub_item in enumerate(element):
                    print(f"{prefix}  [{i}] ", end="")
                    print_element(sub_item, indent=indent + 4, slice_length=slice_length)
                print(f"{prefix}]")

            elif isinstance(element, dict):
                print(f"{prefix}- (dict) {{")
                for k, v in element.items():
                    print(f"{prefix}  '{k}': ", end="")
                    print_element(v, indent=indent + 4, slice_length=slice_length)
                print(f"{prefix}}}")

            else:
                # Fallback for int, float, bool, None, etc.
                val_str = str(element)
                if len(val_str) > slice_length:
                    val_str = val_str[:slice_length] + "..."
                print(f"{prefix}- (other) {val_str}")

        @staticmethod
        def print_sliced_array_recursive(elements, slice_length=50):
            """
            Iterates over a list of elements (any type) and prints them recursively.
            """
            for idx, item in enumerate(elements):
                print(f"Index {idx}:")
                print_element(item, indent=4, slice_length=slice_length)
                print()  # Blank line after each top-level item

        @staticmethod
        def get_attached_images(message_content: list[dict]):
            """
            Returns a list of attached images.

            Args:
                message_content (list[dict]): A list of message content dictionaries.

            Returns:
                list[str]: A list of image URLs.
            """
            attached_images = []
            for item in message_content:
                if item['type'] == 'image_url':
                    attached_images.append(item['image_url']['url'])
            return attached_images

        @staticmethod
        def convert_unsupported_image(image_path: str):
            """
            Converts an unsupported image format such as .tiff to a supported png.
            """
            supported_formats = ['png', 'jpg', 'jpeg']

            input_path = image_path
            image_name = Path(input_path).name
            output_path = Path(self.valves.PNG_DIR) / (image_name + ".png")

            with Image.open(input_path) as img:
                # Save the image as a PNG
                img.save(output_path, format='PNG')

            return str(output_path)
        
        @staticmethod
        def generate_chat_form(image_path: str):
            """
            Generates a chat form for a given image.
            """
            chat_data = {
                "model": self.user_valves.MODEL_ID,
                "stream": True, # We want the model to stream the response regardless, we can handle the output on our end
                "options": {
                    "temperature": self.user_valves.TEMPERATURE,
                    "max_tokens": self.user_valves.MAX_TOKENS, 
                    "num_ctx": self.valves.MODEL_CONTEXT_SIZE
                }
            }
            image_data = convert_image_to_base64(image_path)

            messages_payload = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": self.user_valves.SYSTEM_PROMPT
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.user_valves.PROMPT
                        },
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data
                        }
                    }]
                }
            ]

            chat_data["messages"] = messages_payload
            return chat_data

        @staticmethod
        async def handle_pdf(
            file_path: str,
            png_dir: str,
            logger,
            __event_emitter__
        ) -> List[str]:
            """
            Checks if a PDF file has already been converted to PNGs. 
            If not, converts it. Returns a list of PNG file paths.
            """
            # Extract only the first part of the base name before any extension
            base_name = Path(file_path).stem
            target_dir = Path(png_dir) / base_name
            target_dir.mkdir(parents=True, exist_ok=True)

            existing_pngs = [str(p) for p in target_dir.glob("*.png")]
            if existing_pngs:
                # Already converted
                logger.debug(f"PDF already converted in {target_dir}")
                return existing_pngs

            # Not converted yet
            logger.debug(f"Converting PDF to PNG: {file_path}")
            await update_status("Converting File...", __event_emitter__)
            new_pngs = await convert_pdf_to_pngs(file_path, __event_emitter__, str(target_dir))
            return new_pngs

        @staticmethod
        def handle_image(
            file_path: str,
            png_dir: str,
            logger
        ) -> str:
            """
            Handle an image file. If it's already a PNG (or supported),
            it might just return the existing file. Otherwise, convert 
            to a supported PNG and return that new path.
            """
            if len(file_path) > 200:
                logger.debug(f"Image file path too long: {file_path[:200]}...")
                raise Exception(f"Image file path too long")
            logger.debug(f"Handling image: {file_path}")
            if is_png(file_path):  
                # If you have a check for PNGs, or it’s already correct format
                return file_path

            # Otherwise, convert to a PNG (if necessary)
            clean_png = convert_unsupported_image(file_path)
            return clean_png

        @staticmethod
        async def process_downloaded_file(
            file_path: str,
            png_dir: str,
            logger,
            __event_emitter__
        ) -> Union[List[str], str]:
            """
            Given a local file path, determine if it’s PDF or image, and handle accordingly.
            Returns either a list of PNGs or a string error.
            """
            if is_pdf(file_path):
                png_files = await handle_pdf(file_path, png_dir, logger, __event_emitter__)
                return png_files
            elif is_image(file_path):
                clean_png = handle_image(file_path, png_dir, logger)
                return [clean_png]
            else:
                # Unsupported file type
                filename = Path(file_path).name
                error_msg = f"What type of file are you trying to convert? I don’t know what to do with {filename}."
                return error_msg

        @staticmethod
        async def process_url_task(
            task: dict,
            png_dir: str,
            logger,
            __event_emitter__
        ) -> Union[List[str], str]:
            """
            Handle a single 'url' task. This includes:
            - Getting the filename from URL/headers
            - Checking if it exists locally
            - Downloading if it does not exist
            - Then passing off to PDF/image handler
            Returns either a list of PNG file paths or an error string.
            """
            url = task["data"]["url"]
            logger.debug(f"Processing URL task: {url}")
            
            # 1. Get filename
            filename = get_filename_from_url_or_headers(url)
            if not filename or isinstance(filename, Exception):
                if filename is None:
                    logger.info("Filename was None from get_filename_from_url_or_headers")
                    return []  # or some suitable message
                else:
                    logger.error(f"Error getting filename: {url}\nError Message: {filename}")
                    return "Error getting filename. Please check the URL and try again."

            # 2. Check if file already exists
            existing_path = check_existing_file(filename)
            if isinstance(existing_path, Exception):
                logger.error(f"Error checking existing file: {filename}\nError Message: {existing_path}")
                return "Error checking existing file. Please check the URL and try again."

            # 3. If it exists locally, process it
            if existing_path:
                logger.debug(f"File already exists: {existing_path}")
                return await process_downloaded_file(existing_path, png_dir, logger, __event_emitter__)

            # 4. Otherwise, download
            await update_status("Downloading File...", __event_emitter__)
            downloaded_path = download_file(url)
            if isinstance(downloaded_path, Exception):
                logger.error(f"Error downloading file: {url}\nError Message: {downloaded_path}")
                return "Error downloading file. Please check the URL and try again."

            # 5. Process the newly downloaded file
            return await process_downloaded_file(downloaded_path, png_dir, logger, __event_emitter__)

        @staticmethod
        async def process_file_task(
            task: dict,
            png_dir: str,
            logger,
            __event_emitter__
        ) -> Union[List[str], str]:
            """
            Handle a single 'file' task (the user already has local file paths).
            Returns either a list of PNG file paths or an error string.
            """
            logger.debug("Processing 'file' task.")
            png_files = []

            for file in task["data"]["files"]:
                file_path = file["path"]
                logger.debug(f"Handling local file: {file_path}")
                
                if is_pdf(file_path):
                    # Convert or retrieve PDF
                    base_name = Path(file_path).stem
                    target_dir = Path(png_dir) / base_name
                    target_dir.mkdir(parents=True, exist_ok=True)

                    existing_pngs = [str(p) for p in target_dir.glob("*.png")]
                    if existing_pngs:
                        png_files.extend(existing_pngs)
                    else:
                        new_pngs = await convert_pdf_to_pngs(file_path, __event_emitter__, str(target_dir))
                        png_files.extend(new_pngs)
                elif is_image(file_path):
                    # Convert or just take the existing image
                    clean_png = convert_unsupported_image(file_path)
                    png_files.append(clean_png)
                else:
                    filename = Path(file_path).name
                    err = f"What type of file are you trying to convert? I don’t know what to do with {filename}."
                    return err

            return png_files

        @staticmethod
        async def process_image_task(
            task: dict,
            png_dir: str,
            logger
        ) -> List[str]:
            """
            Handle a single 'image' task (base64-encoded images).
            Returns a list of PNG file paths saved to disk.
            """
            logger.debug("Processing 'image' (base64) task.")
            png_files = []
            for image in task["data"]["images"]:
                image_filename = f"{random.randint(1, 1000)}image.png"
                output_path = Path(png_dir) / image_filename
                try:
                    # If the string includes a prefix like "data:image/png;base64,", remove it:
                    # (Only do this if you're sure such a prefix exists.)
                    if image.startswith("data:image"):
                        image = image.split(",", 1)[1]

                    # Decode the Base64 string to binary data
                    image_data = base64.b64decode(image)

                    # Write the data to a PNG file
                    with open(output_path, "wb") as f:
                        f.write(image_data)
                    
                    logger.debug(f"Processed image: {output_path}")
                    png_files.append(str(output_path))
                except Exception as e:
                    err = f"Failed to process image: {e}"
                    logger.error(err)
                    return e
            return png_files

        # A function that extracts the page number from the filepath
        @staticmethod
        def extract_page_number(filepath: str) -> int:
            match = re.search(r'page_(\d+)\.png', filepath)
            if match:
                return int(match.group(1))
            # If there's no match (unexpected), return something that won't break sorting
            return 0


        #BEGIN PIPELINE 
        files = __files__ or None

        # Oink oink...ill just collect stuff for global use when i need it. When there are memory issues, look to my sty
        data_hog = []
        # Start with a failure status and hope for the best
        pipe_success = False
        # If we have an attached file, check if it's a PDF, .doc, .docx, .xlsx, .xls, etc
        doc_type = None
        # Store the paths to the attached files
        attached_files: list[dict] = []
        # Once we do pass everything to the LLM, we'll store the responses along with additional information
        llm_responses: list[dict] = []
        # We will need to convert PDFs to PNGs for the LLM to process them. We will store the PNG file paths here
        png_files = []
        # if the user message is a URL, we will store it here to be downloaded and processed
        doc_url = False

        # Create user to auth chat generation
        USER = Users.get_user_by_id(__user__["id"])

        # Get the last user message -- careful logging this bad boy...if there is an image attachment...prepare for base64 hell
        last_user_message = body['messages'][-1]['content']
    
        if not files:
            logger.debug(f"-----No files found-----")
        else:
            for file in files:
                logger.debug(f"File: {file}")
                file_path = Files.get_file_by_id(file['file']['id']).path
                if file_path:
                        # Extract the file type from the file path
                        file_type = Path(file_path).suffix
                        attached_files.append(
                            {
                                "path": file_path,
                                "type": file_type
                            }
                        )
        if len(attached_files) > 0:
                logger.debug(f"Attached files: {attached_files}")
        
        
        # Depending on attachments and message, we will have different tasks to perform. We will store them as a list of dict.
        tasks = []

        try:
            # Lets check if we need any further user input
            # 1. Any attached files
            if len(attached_files) > 0:
                tasks.append({
                    "type": "file",
                    "data": {
                        "files": attached_files
                    }
                })
            # 2. Any attached images
            attached_images = []
            if isinstance(last_user_message, list):
                for item in last_user_message:
                    if item['type'] == 'image_url':
                        attached_images.append(item['image_url']['url'])
            if len(attached_images) > 0:
                tasks.append({
                    "type": "image",
                    "data": {
                        "images": attached_images
                    }
                })
            # 3. User message a URL
            if isinstance(last_user_message, str):
                logger.debug(f"Last user message is a string: {last_user_message}")
                if last_user_message.startswith("http") or last_user_message.startswith("https"):
                    logger.debug(f"Last user message is a URL: {last_user_message}")
                    tasks.append({
                        "type": "url",
                        "data": {
                            "url": last_user_message
                        }
                    })
            # 4. TODO: add custom commands list extended functionality

            new_user_input = None
            # Check if we have any tasks
            if len(tasks) == 0 and last_user_message is not None and isinstance(last_user_message, str):
                # We couldnt figure out any tasks to perform. Lets ask the user to try again through an action input
                # new_user_input = await action(
                #     __event_call__,
                #     "Enter a Web Address",
                #     "I could not figure out what you meant. Please try again. Please paste the URL to your file/image here.",
                #     "input",
                #     "Enter your url..."
                # )

                    if not last_user_message.startswith("http") and not last_user_message.startswith("https"):
                        # User entered a non URL string. Try again
                        return "Invalid URL...Please try again.\n\n" + GLOBAL_DEFAULT_RESPONSE 
                    else:
                        tasks.append({
                            "type": "url",
                            "data": {
                                "url": last_user_message
                            }
                        })

            # We should have at least 1 task now
            if len(tasks) == 0:
                return "Error...still no tasks...we definitely need a task here."

            png_files = []

            for task in tasks:
                task_type = task.get("type")
                logger.debug(f"Processing Task: {task['type']}")

                if task_type == "url":
                    result = await process_url_task(task, self.valves.PNG_DIR, logger, __event_emitter__)
                elif task_type == "file":
                    result = await process_file_task(task, self.valves.PNG_DIR, logger, __event_emitter__)
                elif task_type == "image":
                    result = await process_image_task(task, self.valves.PNG_DIR, logger)
                else:
                    # Unknown task type
                    logger.warning(f"Unknown task type encountered: {task_type}")
                    continue

                # If the result is a string, treat as error; if list, treat as successful PNGs
                if isinstance(result, str):
                    return result  # Return the error message immediately
                else:
                    png_files.extend(result)

            # Final check: we should have at least 1 PNG file
            if len(png_files) == 0:
                return "Error...still no png files...we definitely need at least one PNG."

            user_pages = None
            if len(png_files) == 1:
                user_pages = "1"
            else:
                user_pages = await action(
                    __event_call__, 
                    "Process Images", 
                    f"""Total pages: {len(png_files)}
                    Enter the page(s) you want to process (enter "all" for all pages).
                    Examples: 3 or 3-5 or 3,4,5""", 
                    "input"
                )
            if user_pages:
                # We have user input on the selected png files to process
                selected_pages = parse_range(user_pages)
                if not selected_pages:
                    selected_pages = [int(page.strip()) for page in user_pages.split(",")]
                    if not selected_pages:
                        await update_status("Invalid page numbers. Cmon...", __event_emitter__, True)
                        return "You didn't enter any valid page numbers."
                # Verify all of the selected pages exist
                if not all(page in range(1, len(png_files) + 1) for page in selected_pages):
                    await update_status("Invalid page numbers. Cmon...", __event_emitter__, True)
                    return "You entered an invalid page number or range. Please try again."

                page_count = len(selected_pages)

                # Process the selected pages through the LLM
                current_page = 1
                sorted_png_files = sorted(png_files, key=lambda x: extract_page_number(x))
                logger.info(f"Sorted PNG files: {sorted_png_files}")
                for page in selected_pages:

                    page_path = sorted_png_files[page - 1]
                    if not Path(page_path).exists():
                        await update_status(f"Page: {page} does not exist at {page_path}. Skipping...", __event_emitter__, True)
                        logger.info(f"Page {page} does not exist at {page_path}. Skipping...")
                        continue
                    logger.info(f"Processing page: {page} {current_page} of {page_count}: {page_path}")
                    await update_status(f"Processing page: {page} ({page_path}) {current_page} of {page_count}", __event_emitter__, True)

                    # Generate the chat form (OpenAI API) for the page
                    chat_form = generate_chat_form(page_path)
                    try:
                        response = await generate_chat_completion(
                            __request__,
                            chat_form,
                            USER
                        )
       
                        model_responses = ""
                        display_stream = self.user_valves.STREAM
                        if display_stream == True:
                            await update_response(f"\n\n##Page {page}##\n", __event_emitter__)
                        
                        # Get the response from the LLM
                        async for chunk in response.body_iterator:
                            if chunk:
                                chunk_data = chunk.replace('data: ', '')
                                if chunk_data.strip() == '[DONE]':
                                    continue
                                try:
                                    json_chunk = json.loads(chunk_data)
                                    if content := json_chunk.get('choices', [{}])[0].get('delta', {}).get('content'):
                                        model_responses += content
                                        if self.user_valves.STREAM:
                                            await update_response(content, __event_emitter__, False)

                                except json.JSONDecodeError:
                                    logger.error(f"JSONDecodeError in chunk: {chunk_data}")
                                    continue
                                
                        if model_responses:
                            # Append the response to the list
                            llm_responses.append(
                                {
                                    "page": page,
                                    "png_file_path": page_path,
                                    "transcription": model_responses
                                }
                            )
                            current_page += 1
                        else:
                            await update_response("No response from LLM.", __event_emitter__)
                            logger.error(f"No response from LLM for page {page} at {page_path}.")
                    
                    except Exception as e:
                        logger.error(f"Error processing page {page} - {page_path}: {e}")
                        await update_response(f"Error processing page {page} - {page_path}: {e}", __event_emitter__)
                        continue

                if self.user_valves.COMPILE_TO_PDF == True:
                    # Extract "transcription" and "png_file_path" field from llm_responses
                    data = [{"transcription_text": response["transcription"], "png_file_path": response["png_file_path"], "page_number": response["page"]} for response in llm_responses]
                    # Generate a new PDF combining the pngs and model transcriptions
                    doc_title = Path(png_files[0]).stem
                    output_path = Path(self.valves.PNG_DIR) / (doc_title + "_transcription.pdf")
                    new_pdf_doc = gen_combined_pdf(data, str(output_path))
                    # Save the PDF
                    try:
                        # generate a guid for the file
                        doc_id = uuid.uuid4()
                    
                        if new_pdf_doc == True:
                            form_data = FileForm(
                                id=str(doc_id),
                                filename=doc_title + ".pdf",
                                path=str(output_path),
                                data={},
                                meta={}
                            )
                            new_file = Files.insert_new_file(__user__["id"], form_data)
                            new_file_id = new_file.id
                            await update_response(f"\n\nDownload PDF: {WEBUI_URL}/api/v1/files/{new_file_id}/content", __event_emitter__)
                        else:
                            await update_response(f"\n\n!Error! Unable to save compiled PDF: {str(new_pdf_doc)}", __event_emitter__)

                    except Exception as e:
                        logger.error(f"Error saving PDF: {e}")
                        await update_response(f"\n\n!Error! Unable to save compiled PDF: {str(e)}", __event_emitter__)
                else:
                    pass
            else:
                await update_response("No pages selected.", __event_emitter__)
            
            # Add final status update inside the session context
            await update_status("Processing complete!", __event_emitter__, done=True)

        except Exception as e:
            await update_response(f"\n\n!Error! General Error: {str(e)}", __event_emitter__)
            await update_status(f"Error: {str(e)}", __event_emitter__, done=True)        
