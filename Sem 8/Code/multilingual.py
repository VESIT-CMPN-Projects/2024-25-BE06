from langdetect import detect
from googletrans import Translator
import pandas as pd

employee_df = pd.read_csv('employee_data.csv')

def detect_language(text):
    """Detect the language of the input text."""
    try:
        return detect(text)
    except:
        return "en"  # Default to English if language detection fails

def translate_text(text, dest_lang):
    """Translate text to the desired language using Google Translate."""
    translator = Translator()
    try:
        return translator.translate(text=text, dest=dest_lang).text
    except Exception as e:
        return f"Translation error: {e}"

def process_chat(user_input):
    """Process user input and provide a response in the user's input language."""
    detected_lang = detect_language(user_input)
    response = ""

    if "policy" in user_input.lower():
        response = "Company policies are as follows: [Sample policy data]"
    else:
        response = "I'm here to assist you with employee data and policies."

    # Translate response if the user's input language is not English
    if detected_lang != "en":
        response = translate_text(response, detected_lang)
        print(f"Type of translated response: {type(response)}")  # Debugging

    return response