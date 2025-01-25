"""
title: Doc Transcriber
author: Sawyer Borror <sawyerksu@gmail.com>
author_url: https://github.com/thesawdawg/open-webui-docAI.git
version: 1.4
license: MIT
Dependencies: pydantic, open_webui, urllib3, pypdf, pdf2image, pillow
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
from pdf2image import convert_from_path
from PIL import Image
from fastapi import Request
from open_webui.utils.chat import generate_chat_completion
from open_webui.models.users import Users
from open_webui.models.files import Files, FileForm
import base64
import json
import aiohttp
import re
import fpdf
import uuid
from io import BytesIO

logger = logging.getLogger(__name__)
# Configure logger to output to stdout
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


class Pipe:
    """Pipe class handles the processing of a message or a file uploaded to the chatbot interface."""

    class Valves(BaseModel):
        """UserValves class contains the valves for the user."""
        MODEL_ID: str = Field(default="llama3.2-vision:latest")
        STREAM: bool = Field(default=True)
        SYSTEM_PROMPT: str = Field(default="You are a file transcriber. You will carefully analyze and transcribe text within an image or file and respond with the verbatim transcription formatted with markdown. If there is no text within the image or file, provide a brief description of what is within the image or file. Do not respond with any additional text or summaries.")
        PROMPT: str = Field(default="Transcribe the text in the image or file and respond with a verbatim transcription.")
        TEMPERATURE: float = Field(default=0.7)
        MAX_TOKENS: int = Field(default=8192)
        COMPILE_TO_PDF: bool = Field(default=True, description="WIP!!!---Whether to compile the PDF with the OCR text.")
        COMBINED_PDF_DIR: str = Field(default=os.path.join(DATA_DIR, "combined_pdfs"), description="The directory to use when saving the combined PDF.")
        MODEL_CONTEXT_SIZE: int = Field(default=8192, description="The context size to use when generating the chat completion.")
        PNG_DPI: int = Field(default=400, description="The DPI to use when converting the image to a PDF.")
        PNG_CONVERT_THREAD_COUNT: int = Field(default=4, description="The number of threads to use when converting the image to a PDF.")
        DOWNLOAD_DIR: str = Field(default=UPLOAD_DIR, description="The directory to use when downloading the file.")
        PNG_DIR: str = Field(default=DATA_DIR, description="The directory to use when saving the PNG images. A sub-directory will be created for each downloaded file that will contain the png files.")
    


    def __init__(self):
        """Initializes the Pipe class by setting the valves."""
        self.valves = self.Valves()

    async def on_startup(self):
        """This function is called when the server is started."""
        print("on_startup: doc_transcriber")

    async def on_shutdown(self):
        """This function is called when the server is stopped."""
        print("on_shutdown: doc_transcriber")

    async def on_valves_updated(self):
        """This function is called when the valves are updated."""
        print("on_valves_updated: doc_transcriber")

    async def pipe(self, body: dict, __user__: dict, __request__: Request, __event_emitter__=None, __event_call__=None):
        """
        Processes the last user message and handles user action. This function must be named "pipe" for open-webui to process it.

        Args:
            body (dict): The message or file uploaded to the chatbot interface.
            __event_emitter__ (Optional): The event emitter to send the status updates.
            __event_call__ (Optional): The event call to wait for the user action.

        Returns:
            None
        """
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
        def download_file(url, output_folder=self.valves.DOWNLOAD_DIR):
            """
            Downloads a file from the given URL and saves it to the given output folder.

            Args:
                url (str): The URL of the file to be downloaded.
                output_folder (str): The folder to save the downloaded file in.

            Returns:
                str: The path to the downloaded file.
            """
            os.makedirs(output_folder, exist_ok=True)
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
            file_path = os.path.join(output_folder, filename)
            with open(file_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"File downloaded successfully: {file_path}")
            return file_path

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
                str: The path to the downloaded file.
            """
            file_path = os.path.join(output_folder, f"{file_id}.pdf")
            if os.path.isfile(file_path):
                print(f"File already downloaded: {file_path}")
                return file_path
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
        def get_filename_from_url_or_headers(url, headers):
            """
            Extracts the filename from the given URL or headers.

            Args:
                url (str): The URL containing the filename.
                headers (dict): The headers containing the filename.

            Returns:
                str: The extracted filename.
            """
            if "Content-Disposition" in headers:
                content_disposition = headers["Content-Disposition"]
                if "filename=" in content_disposition:
                    filename = content_disposition.split("filename=")[1].strip('"; ')
                    return filename
            return os.path.basename(urlparse(url).path)

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
        async def convert_pdf_to_pngs(pdf_path,__event_emitter__, output_folder=self.valves.PNG_DIR):
            """
            Converts a PDF file to a series of PNG images.These images
            will be sent to a vision model for OCR. We want to ensure we retain the
            highest quality images possible.

            Args:
                pdf_path (str): The path to the PDF file to be converted.
                output_folder (str): The folder to save the PNG images in.

            Returns:
                list: A list of paths to the saved PNG images.
            """
            # Extract a title from the PDF file
            title_folder = os.path.splitext(os.path.basename(pdf_path))[0]

            # Create the output folder if it doesn't exist
            os.makedirs(os.path.join(output_folder, title_folder), exist_ok=True)

            # Convert the PDF to PNG images, setting configs to retain the highest quality
            page_images = convert_from_path(pdf_path, dpi=self.valves.PNG_DPI, thread_count=self.valves.PNG_CONVERT_THREAD_COUNT)
            page_paths = []
            for idx, image in enumerate(page_images):
                page_path = os.path.join(output_folder,title_folder, f"page_{idx + 1}.png")
                image.save(page_path, "PNG")
                await update_status(f"Converted page {idx + 1}/{len(page_images)}...", __event_emitter__)
                page_paths.append(page_path)
                # Append the path and base64 encoded image to the list
                # page_paths.append((page_path, base64.b64encode(image.tobytes()).decode('utf-8')))
            print(f"PDF converted to PNGs. Pages saved in: {output_folder}")
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
            png_path = os.path.splitext(os.path.join(self.valves.PNG_DIR, os.path.basename(image_path))) + os.path.splitext(image_path)[0] + ".png"
            image.save(png_path, "PNG")
            return png_path
        @staticmethod
        def is_image(file_path):
            """
            Checks if the given file is an image. Allowed filetypes are: png, jpg, jpeg.

            Args:
                file_path (str): The path to the file to be checked.

            Returns:
                bool: True if the file is an image, False otherwise.
            """
            try:
                with open(file_path, "rb") as file:
                    image = Image.open(file)
                    image.verify()
                    # Check if filetype is supported
                    if image.format.lower() in ['png', 'jpg', 'jpeg']:
                        return True
                return False
            except Exception as e:
                logger.error(f"Error checking image: {file_path}\nError Message:{e}")
                return False

        @staticmethod
        def check_existing_file(file_url):
            """
            Checks if the given file already exists in the UPLOAD_DIR.

            Args:
                file_url (str): The URL of the file to be checked.

            Returns:
                str or False: The file path if the file already exists, False otherwise.
            """
            try:
                filename = get_filename_from_url_or_headers(file_url, requests.head(file_url).headers)
                logger.debug(f"Checking if file exists for file URL: {file_url}. Filename: {filename}")
                if os.path.exists(os.path.join(UPLOAD_DIR, filename)):
                    logger.debug(f"File exists. Returning file path. {UPLOAD_DIR}/{filename}")
                    return os.path.join(UPLOAD_DIR, filename)
                logger.debug(f"File does not exist. Returning False.")
                return False
            except Exception as e:
                logger.error(f"Error checking file: {file_url}\nError Message:{e}")
                return False

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
        def gen_pdf(model_responses, original_doc_title):
            """
            Generates a PDF from a list of model responses.

            Args:
                model_responses (list): A list of model responses.
                original_doc_title (str): The title of the original document/image.

            Returns:
                str: The path to the generated PDF file.
            """
            pdf = fpdf.FPDF(format="letter", unit="pt")
            for response in model_responses:
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(0, 10, response)
            try:
                pdf.output(os.path.join(self.valves.PNG_DIR, original_doc_title,"transcription.pdf"), "F")
                return os.path.join(self.valves.PNG_DIR, original_doc_title,"transcription.pdf")
            except Exception as e:
                logger.error(f"Error generating PDF: {e}")
                return None

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

                for item in items_sorted:
                    png_path = item["png_file_path"]
                    transcription_text = item["transcription_text"]

                    # ------------------
                    # Page 1: PNG image
                    # ------------------
                    pdf.add_page()
                    # Optionally set a page size or orientation here if desired
                    # pdf.add_page(orientation="P", format="Letter")

                    # Insert the image at some position, e.g., x=50, y=50
                    # and give it a width. The height will scale automatically
                    # if you don't specify it.
                    pdf.image(png_path, x=50, y=50, w=400)
                    
                    # If you want to label this page or show the page_number, you could:
                    # pdf.set_xy(50, 500)
                    # pdf.set_font("Arial", size=12)
                    # pdf.cell(0, 20, f"Page for: {png_path}", ln=True)

                    # ---------------------------
                    # Page 2: Transcription Text
                    # ---------------------------
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    # Start writing the text from x=50, y=50
                    pdf.set_xy(50, 50)

                    # For multi-line text, use multi_cell.
                    # The first two arguments are cell width and height.
                    pdf.multi_cell(
                        w=500,              # Adjust width to your liking
                        h=18,               # Line height in points
                        txt=transcription_text
                    )

                # Finally, save the PDF
                pdf.output(output_path)
                return True
            except Exception as e:
                logger.error(f"Error creating PDF: {e}")
                return e
            

        #BEGIN PIPELINE 

        logger.debug(f"Pipe Valves: {self.valves.__dict__}")

        # Start with a failure status and hope for the best
        pipe_success = False
        doc_type = None
        try:
            llm_responses = []
            async with aiohttp.ClientSession() as session:
                # __request__.state.client_session = session
                
                try:
                    png_files = []
                    # Get the last user message
                    last_user_message = body['messages'][-1]['content']
                    
                    # If the prompt is a URL, do not wait for user action
                    if last_user_message.startswith("http") or last_user_message.startswith("https"):
                        user_action_input = last_user_message
                    else:
                        user_action_input = await action(__event_call__, "Enter your message...", "Enter your message...", "input")

                    doc_url = False
                    if isinstance(user_action_input, str):
                        logger.debug(f"User Action Input: {user_action_input}")
                        if user_action_input.startswith("http") or user_action_input.startswith("https"):
                            doc_url = True

                    if doc_url:
                        await update_status("Resolving document URL...", __event_emitter__)
                        # Check if we have already downloaded the file
                        existing_file = check_existing_file(user_action_input)
                        google_link = user_action_input.startswith("https://drive.google.com")
                        if google_link:
                            await update_status("Downloading Google Drive file...", __event_emitter__)
                            file_id = extract_google_drive_id(user_action_input)
                            existing_file = check_google_file_already_downloaded(file_id)

                        if not existing_file == False:
                            await update_status(f"File already exists...{existing_file}", __event_emitter__)
                            downloaded_file_path = existing_file
                        else:
                            await update_status("Downloading file...", __event_emitter__)
                            downloaded_file_path = download_file(user_action_input)

                        if is_pdf(downloaded_file_path):
                            # Check if we have already converted the file
                            doc_type = "PDF"
                            doc_title = os.path.splitext(os.path.basename(downloaded_file_path))[0]
                            if os.path.exists(os.path.join(self.valves.PNG_DIR, doc_title)):
                                await update_status(f"Pages already exist...{os.path.join(self.valves.PNG_DIR, doc_title)}", __event_emitter__)
                                page_pngs = os.listdir(os.path.join(self.valves.PNG_DIR, doc_title))
                                png_files.extend(page_pngs)
                            else:
                                doc_title = os.path.splitext(os.path.basename(downloaded_file_path))[0]
                                await update_status("Converting to PNGs...", __event_emitter__)
                                page_pngs = await convert_pdf_to_pngs(downloaded_file_path,__event_emitter__)
                                png_files.extend(page_pngs)

                        elif is_image(downloaded_file_path):
                            doc_type = "Image"
                            await update_status(f"Image: {downloaded_file_path}", __event_emitter__)
                            png_files.append(downloaded_file_path)
                            doc_title = os.path.splitext(os.path.basename(downloaded_file_path))[0]
                        else:
                            await update_status(f"Unsupported file type: {downloaded_file_path}", __event_emitter__)
                            await update_response(f"Unsupported file type: {downloaded_file_path}", __event_emitter__)

                        # Process PNG files
                        if len(png_files) > 1:
                            await update_status("Processing PNG files...", __event_emitter__)
                            user_pages = await action(
                                __event_call__, 
                                "Process Images", 
                                f"""Total pages: {len(png_files)}
                                Enter the page(s) you want to process.
                                Examples: 3 or 3-5 or 3,4,5""", 
                                "input"
                            )
                        else:
                            await update_status("Processing PNG file...", __event_emitter__)
                            user_pages = "1"

                        if user_pages:
                            # Extract the page numbers from the user input
                            # If user_pages equals regex example: 3-5, then extract 3,4,5
                            page_numbers = parse_range(user_pages)
                            if not page_numbers:
                                page_numbers = [int(page.strip()) for page in user_pages.split(",")]
                                if not page_numbers:
                                    await update_status("Invalid page numbers. Cmon...", __event_emitter__)
                                    return

                            # Create user to auth chat generation
                            user = Users.get_user_by_id(__user__["id"])
                            # Make the payload
                            chat_data = {
                                "model": self.valves.MODEL_ID,
                                "stream": self.valves.STREAM,
                                "options": {
                                    "temperature": self.valves.TEMPERATURE,
                                    "max_tokens": self.valves.MAX_TOKENS, 
                                    "num_ctx": self.valves.MODEL_CONTEXT_SIZE
                                }
                            }
                            current_page = 1
                            # Process the selected PNG files
                            for page_number in page_numbers:
                                await update_status(f"Processing page {page_number} ({current_page}/{len(page_numbers)})...", __event_emitter__)
                                try:
                                    # Get the image data from the png_file as base64
                                    if doc_type == "PDF":
                                        image_path = os.path.join(self.valves.PNG_DIR, doc_title, f"page_{page_number}.png")
                                    else:
                                        image_path = png_files[0]
                                    # image_path = os.path.join(self.valves.PNG_DIR, doc_title, f"page_{page_number}.png")
                                    image_data = convert_image_to_base64(image_path)

                                    messages_payload = [
                                        {
                                            "role": "system",
                                            "content": [
                                                {
                                                    "type": "text",
                                                    "text": self.valves.SYSTEM_PROMPT
                                                }
                                            ]
                                        },
                                        {
                                            "role": "user",
                                            "content": [
                                                {
                                                    "type": "text",
                                                    "text": self.valves.PROMPT
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

                                    # Enable streaming in the chat data
                                    chat_data["stream"] = True

                                    # Send it to the LLM magic box
                                    try:
                                        response = await generate_chat_completion(
                                            request=__request__, 
                                            form_data=chat_data, 
                                            user=user, 
                                            bypass_filter=True
                                        )
                                        
                                        collected_response = ""
                                        await update_response(f"---Page {page_number}---\n", __event_emitter__)
                                        async for chunk in response.body_iterator:
                                            if chunk:
                                                chunk_data = chunk.replace('data: ', '')
                                                if chunk_data.strip() == '[DONE]':
                                                    continue
                                                try:
                                                    json_chunk = json.loads(chunk_data)
                                                    if content := json_chunk.get('choices', [{}])[0].get('delta', {}).get('content'):
                                                        collected_response += content
                                                        if self.valves.STREAM == True:
                                                            await update_response(content, __event_emitter__, False)
                                                        else:
                                                            pass
    
                                                except json.JSONDecodeError:
                                                    logger.error(f"JSONDecodeError in chunk: {chunk_data}")
                                                    continue
                                        await update_response(f"\n---End Page {page_number}---\n", __event_emitter__)

                                        if collected_response:
                                            # Append the response to the list
                                            llm_responses.append({"page": page_number,"pdf_file_path": downloaded_file_path, "png_file_path": image_path,"transcription": collected_response})
                                            current_page += 1

                                    except Exception as e:
                                        await update_response(f"Error with LLM: {str(e)}", __event_emitter__)
                                        logger.error(f"Error with LLM: {str(e)}")

                                except Exception as e:
                                    logger.error(f"Error processing page: {e}")
                                    await update_response(str(e), __event_emitter__)

                            if self.valves.COMPILE_TO_PDF == True:
                                await update_response("\n\nGenerating PDF...", __event_emitter__)
                                # Join all of the responses into a new PDF and save it
                                # Extract "transcription" field from llm_responses
                                data = [{"transcription_text": response["transcription"], "png_file_path": response["png_file_path"], "page_number": response["page"]} for response in llm_responses]
                                # Generate a new PDF combining the png and model transcription
                                output_path = os.path.join(self.valves.PNG_DIR, doc_title + "_transcription.pdf")
                                new_pdf_doc = gen_combined_pdf(data,output_path)
                                # Save the PDF
                                try:
                                    # generate a guid for the file
                                    doc_id = uuid.uuid4()
                                
                                    if new_pdf_doc == True:
                                        form_data = FileForm(
                                            id=str(doc_id),
                                            filename=doc_title + ".pdf",
                                            path=output_path,
                                            data={},
                                            meta={}
                                        )
                                        new_file = Files.insert_new_file(__user__["id"], form_data)
                                        new_file_id = new_file.id
                                        await update_response(f"Download PDF: {WEBUI_URL}/api/v1/files/{new_file_id}/content", __event_emitter__)
                                    else:
                                        await update_response(f"Unable to save compiled PDF: {str(new_pdf_doc)}", __event_emitter__)

                                except Exception as e:
                                    logger.error(f"Error saving PDF: {e}")
                                    await update_response(f"Unable to save compiled PDF: {str(e)}", __event_emitter__)
                            else:
                                pass
                        else:
                            await update_response("No pages selected.", __event_emitter__)
                    else:
                        await update_response("No document found.", __event_emitter__)

                    # Add final status update inside the session context
                    await update_status("Processing complete!", __event_emitter__, done=True)

                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await update_response(str(e), __event_emitter__)
                    await update_status(f"Error: {str(e)}", __event_emitter__, done=True)

        except Exception as e:
            logger.error(f"Error in main: {e}")
            await update_status(f"Error in main: {e}", __event_emitter__, done=True)
