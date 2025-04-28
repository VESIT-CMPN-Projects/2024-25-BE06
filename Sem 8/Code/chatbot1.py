import os
import re
import uuid
import pandas as pd
import pinecone
import google.generativeai as genai
import langchain
import markdown
from multilingual import translate_text, detect_language

import PyPDF2
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from fpdf import FPDF
from io import BytesIO

from flask import Flask, render_template, request, jsonify, send_file
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import LLMMathChain
from langchain.agents import Tool
from langchain_experimental.tools import PythonAstREPLTool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from email_sender import get_hr_email, send_email_to_hr

langchain.debug = True  # Enable LangChain debugging

app = Flask(__name__)

# ------------------------------------------------------------------
# 1. Load CSV and Global Variables
# ------------------------------------------------------------------
df = pd.read_csv('employee_data.csv')
user_name = None

# ------------------------------------------------------------------
# 2. Helper Functions
# ------------------------------------------------------------------

# Global dictionary to hold credentials from the chatbot session.
session_credentials = {}

def get_employee_info(target_employee_name):
    """
    Retrieve specific employee info after authenticating the requester.
    In the chatbot, the user's credentials (employee ID and password)
    should be provided in a prior message (e.g., in the format "EmployeeID:Password")
    and stored in the session_credentials global variable.
    
    Only employees from the HR Department are authorized to view employee details.
    
    Parameters:
      target_employee_name (str): The name of the employee whose details are being requested.
    
    Returns:
      str: Employee details if authentication and authorization succeed; 
           otherwise, a message prompting for credentials or indicating an error.
    """
    global session_credentials

    # Check if the credentials have been provided via the chatbot.
    if not session_credentials.get("employee_id") or not session_credentials.get("password"):
        return ("Please provide your credentials in the format 'EmployeeID:Password'. "
                "Once provided, ask for the employee details again.")
    
    try:
        # Convert the stored employee ID to an integer.
        try:
            auth_employee_id_int = int(session_credentials["employee_id"])
        except ValueError:
            # Clear invalid credentials.
            session_credentials.clear()
            return "Invalid employee ID format in your credentials."

        auth_password = session_credentials["password"]

        # Authenticate the requesting employee by checking employee_id and password.
        auth_row = df[(df["employee_id"] == auth_employee_id_int) & (df["password"] == auth_password)]
        if auth_row.empty:
            # Clear credentials if authentication fails.
            session_credentials.clear()
            return "Authentication failed: Invalid employee ID or password."

        # Check if the authenticated employee belongs to the HR Department.
        dept = auth_row.iloc[0]["organizational_unit"].strip().lower()
        if dept != "hr department":
            return "You do not have authorization to access employee details."

        # Retrieve and return the target employee's information.
        result = df[df["name"].str.lower() == target_employee_name.lower()]
        if result.empty:
            return f"No information found for employee: {target_employee_name}"
        return result.to_string(index=False)
    except Exception as e:
        return f"Error retrieving employee information: {e}"



# ------------------------------------------------------------------
# 3. Google Generative AI + Pinecone Setup
# ------------------------------------------------------------------
    
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

genai.configure(api_key=GOOGLE_API_KEY)
model = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-latest", google_api_key=GOOGLE_API_KEY)

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
index_other="other-hr-policies"
index1 = pc.Index(index_other)
embed = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
vectorstore = PineconeVectorStore(index=index, embedding=embed, text_key="text")
other_vectorstore=PineconeVectorStore(index=index1, embedding=embed, text_key="text")

model1 = genai.GenerativeModel("models/gemini-1.5-pro-latest")

# ------------------------------------------------------------------
# 4. Tools
# ------------------------------------------------------------------
def fetch_and_format_other_hr_policy(query):
    """Retrieve Other HR policy text and format it nicely."""
    print(f"fetch_and_format_other_hr_policy called with query: {query}")
    try:
        docs_with_scores = other_vectorstore.similarity_search_with_score(query, k=1)
        if not docs_with_scores:
            return "No relevant HR policy sections found."

        policy_content = "\n\n".join([doc.page_content for doc, score in docs_with_scores])

        gemini_prompt = f"""
Reformat the following HR policy text into a short, well-structured HTML snippet 
with headings (<h2>, <h3>), bullet points (<ul>, <li>), and bold (<b>) where needed. 
Do NOT include <html>, <head>, or <body> tags, and do NOT use triple backticks or code fences. 
Only return the final HTML snippet.

Policy text:
{policy_content}
"""
        response = model1.generate_content(gemini_prompt)
        if not response.text:
            return "Error: Gemini returned an empty response."

        formatted_response = response.text.strip()
        formatted_response = re.sub(r'(html)?', '', formatted_response, flags=re.IGNORECASE)
        formatted_response = re.sub(r'</?(html|body).*?>', '', formatted_response, flags=re.IGNORECASE)

        return formatted_response

    except Exception as e:
        return f"Error retrieving Other HR policies: {e}"


def fetch_and_format_policy(query):
    """Retrieve policy text from Pinecone and ask Gemini to format it with headings/bullets."""
    print(f"fetch_and_format_policy called with query: {query}")
    try:
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=1)
        if not docs_with_scores:
            return "No relevant policy sections found."

        policy_content = "\n\n".join([doc.page_content for doc, score in docs_with_scores])

        gemini_prompt = f"""
Reformat the following policy text into a short, well-structured HTML snippet 
with headings (<h2>, <h3>), bullet points (<ul>, <li>), and bold (<b>) where needed. 
Do NOT include <html>, <head>, or <body> tags, and do NOT use triple backticks or code fences. 
Only return the final HTML snippet.

Policy text:
{policy_content}
"""
        response = model1.generate_content(gemini_prompt)
        if not response.text:
            return "Error: Gemini returned an empty response."

        formatted_response = response.text.strip()
        formatted_response = re.sub(r'(html)?', '', formatted_response, flags=re.IGNORECASE)
        formatted_response = re.sub(r'</?(html|body).*?>', '', formatted_response, flags=re.IGNORECASE)

        return formatted_response

    except Exception as e:
        return f"Error retrieving policies: {e}"
    
def fetch_and_analyze_employee_data(query):
    """
    Convert a natural language request into a Pandas operation via Gemini,
    execute it on the employee DataFrame, and return the result.
    
    This version first verifies that the requester has provided valid credentials
    (stored in session_credentials) and is from the HR Department.
    
    Parameters:
      query (str): The natural language query describing the desired data analysis.
      
    Returns:
      str: The result of the query or an error/authorization message.
    """
    global session_credentials

    # Check for credentials in the session
    if not session_credentials.get("employee_id") or not session_credentials.get("password"):
        return ("Please provide your credentials in the format 'EmployeeID:Password'. "
                "Once provided, ask for employee data analysis again.")
    
    try:
        # Convert employee_id to int for proper comparison
        try:
            auth_employee_id_int = int(session_credentials["employee_id"])
        except ValueError:
            session_credentials.clear()
            return "Invalid employee ID format in your credentials."

        auth_password = session_credentials["password"]

        # Authenticate the requesting employee
        auth_row = df[(df["employee_id"] == auth_employee_id_int) & (df["password"] == auth_password)]
        if auth_row.empty:
            session_credentials.clear()
            return "Authentication failed: Invalid employee ID or password."

        # Check if the authenticated employee belongs to the HR Department
        dept = auth_row.iloc[0]["organizational_unit"].strip().lower()
        if dept != "hr department":
            return "You do not have authorization to analyze employee data."

        # Proceed with converting the natural language query to a Pandas operation.
        print(f"fetch_and_analyze_employee_data called with query: {query}")
        gemini_prompt = f"""
Convert the following natural language request into a valid Pandas operation 
for a DataFrame with these columns: {', '.join(df.columns)}.

Request: "{query}"

Return only the code (e.g., df.query(...) or df[df['...']==...] etc.) inside triple backticks.
No extra text or explanation.
"""
        response = model1.generate_content(gemini_prompt)
        if not response.text:
            return "Error: Gemini returned an empty response."

        structured_query = response.text.strip().strip("``")
        print("DEBUG: Gemini Generated Query ->", structured_query)

        try:
            result = eval(structured_query, {"df": df, "pd": pd})
        except Exception as e:
            return f"Error executing employee data query: {e}"

        if isinstance(result, int):
            return f"There are {result} employees in the requested department."
        elif isinstance(result, pd.DataFrame):
            if result.empty:
                return "No matching employee data found."
            return result.to_string(index=False)
        else:
            return str(result)

    except Exception as e:
        return f"Error retrieving employee data: {e}"

def handle_meeting_request(input_text):
    """Send meeting email to HR by extracting employee name and looking up email from CSV."""
    try:
        print(f"HR Email: {get_hr_email()}")
        # Extract name and optionally employee ID
        name_match = re.search(r'([\w\s]+)', input_text, re.IGNORECASE)
        employee_name = name_match.group(1).strip() if name_match else None

        if not employee_name:
            employee_name="employee"

        # Search employee info from CSV
        # employee_record = df[df['name'].str.lower() == employee_name.lower()]
        # if employee_record.empty:
        #     return f"No record found for {employee_name}. Please check the spelling."

        # Find HR email (could be supervisor email or default HR)
        # hr_email = get_hr_email()

        # Send email
        success = send_email_to_hr(employee_name)

        if success:
            return f"Meeting request sent to HR on behalf of {employee_name}."
        else:
            return "Failed to send meeting request. Please try again later."

    except Exception as e:
        return f"Error handling meeting request: {e}"

llm_math = LLMMathChain.from_llm(llm=model)

# tools = [
#     Tool(
#         name="timekeeping_policy_retrieval",
#         func=fetch_and_format_policy,
#         description="Useful for timekeeping policy questions."
#     ),
#     Tool(
#         name="employee_data_analysis",
#         func=fetch_and_analyze_employee_data,
#         description="Analyzes employee data with Pandas."
#     ),
#     Tool(
#         name="calculator",
#         func=llm_math.run,
#         description="Performs math calculations with LLM."
#     ),
#     Tool(
#         name="employee_info_retrieval",
#         func=get_employee_info,
#         description="Retrieves specific info about an employee by name."
#     ),
#     Tool(
#         name="send_meeting_email",
#         func=handle_meeting_request,
#         description="Use this tool to send a meeting invite email to HR when an employee requests a meeting."
#     ),
# ]

tools = [
    Tool(
    name="other_hr_policy_retrieval",
    func=fetch_and_format_other_hr_policy,
    description="Use this tool to answer queries related to Code of Conduct, Grievances, Anti-Harassment, Disciplinary Policy, Remote Work, Cybersecurity, Travel, Promotions, Exit rules, and similar HR topics."
),
    Tool(
        name="timekeeping_policy_retrieval",
        func=fetch_and_format_policy,
        description="Useful for timekeeping policy questions."
    ),
    Tool(
        name="send_meeting_email",
        func=handle_meeting_request,
        description="Use this tool to send a meeting invite email to HR when an employee requests a meeting."
    ),
    Tool(
        name="employee_data_analysis",
        func=fetch_and_analyze_employee_data,
        description="Analyzes employee data with Pandas."
    ),
    Tool(
        name="calculator",
        func=llm_math.run,
        description="Performs math calculations with LLM."
    ),
    Tool(
        name="employee_info_retrieval",
        func=get_employee_info,
        description="Retrieves specific info about an employee by name."
    ),
]

# system_prompt = """
# You are an HR assistant. Use these tools:
# - send_meeting_email
# - timekeeping_policy_retrieval
# - employee_data_analysis
# - calculator
# - employee_info_retrieval

# Answer user questions about policies, employee data, or calculations.  When a user wants to schedule a meeting with HR, use the 'send_meeting_email' tool.
# When you use a tool, present the tool's output directly to the user in a clear, helpful manner.
# """

system_prompt = """
You are a multilingual chatbot. Respond in the user's input language.
You are an HR assistant. Use these tools:

- send_meeting_email: Use this tool to send a meeting invite email to HR when an employee requests a meeting. If the user does not specify a particular HR person, assume it is for the general HR department.
- timekeeping_policy_retrieval: Useful for timekeeping policy questions like attendance, leaves, clock-ins, holidays, and related topics.
- other_hr_policy_retrieval: Useful for questions about Code of Conduct, Disciplinary Actions, Grievances, Anti-Sexual Harassment (POSH), Remote Work Policy, Cybersecurity Rules, Childcare Benefits, Health & Safety, Background Verification, Travel Reimbursement, Exit Process, and Promotions policy.
- employee_data_analysis: Analyzes employee data with Pandas.
- calculator: Performs math calculations with LLM.
- employee_info_retrieval: Retrieves specific info about an employee by name.

Answer user questions about HR policies, employee data, or calculations. 
When a user wants to schedule a meeting with HR, use the 'send_meeting_email' tool immediately. 
Do not ask the user to specify an HR contact unless they explicitly mention a name.
When you use a tool, present the tool's output directly to the user in a clear, helpful manner.
"""

checkpointer = MemorySaver()
agent = create_react_agent(model, tools, checkpointer=checkpointer, prompt=system_prompt)
user_threads = {}

# ------------------------------------------------------------------
# 5. Summarizer Functions
# ------------------------------------------------------------------
def extract_text_from_pdf(pdf_file):
    """Extract text from the provided PDF file."""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def summarize_text(text, sentence_count=5):
    """Summarize the extracted text to the specified number of sentences."""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join([str(sentence) for sentence in summary])

def generate_pdf_from_summary(summary, output_filename="summary.pdf"):
    """Generate a PDF file from the summary and return it as a BytesIO stream."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, summary)
    pdf_bytes = pdf.output(dest="S").encode("latin1")
    pdf_stream = BytesIO(pdf_bytes)
    pdf_stream.seek(0)
    return pdf_stream

# ------------------------------------------------------------------
# 6. Flask Routes
# ------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

import re

# Global dictionary to store the credentials
session_credentials = {}

@app.route('/chat', methods=['POST'])
def chat():
    global user_name
    global session_credentials
    
    user_input = request.json['message']
    user_id = request.json.get('user_id', 'user1')

    # 1. Check if user input matches the pattern "EmployeeID:Password"
    match = re.match(r'^(\d+):(.*)$', user_input)
    if match:
        # 2. Store credentials in the session_credentials dictionary
        session_credentials["employee_id"] = match.group(1)
        session_credentials["password"] = match.group(2)
        
        # 3. Return a friendly message to the user
        return jsonify({
            'response': 'Credentials received. You can now ask for employee details, e.g., "Get info about Mark Delos Santos".'
        })

    # 4. If it doesn't match credentials, proceed with the usual chatbot logic
    try:
        # preferred_language = detect_language(user_input)

        # # Translate the user input to English for processing
        # translated_input = translate_text(user_input, "en") if preferred_language != "en" else user_input

        if re.search(r'\b(meet|meeting|talk|discuss|schedule)\b.*\b(hr|human resources)\b', user_input, re.IGNORECASE):
            # Use the extracted name if available, else default to "Employee"
            employee_name = user_name if user_name else "Employee"
            return handle_meeting_request(employee_name)
        
        # Maintain a thread for "user1" (or any unique user ID)
        if user_id not in user_threads:
            user_threads[user_id] = str(uuid.uuid4())
        thread_id = user_threads[user_id]
        checkpoint_id = str(uuid.uuid4())

        # Pass the user_input to your agent
        response = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config={"configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}}
        )

        # Extract the AI’s response
        if "messages" in response and response["messages"]:
            last_message = response["messages"][-1]
            if isinstance(last_message, AIMessage):
                text_response = re.sub(
                    r'json\s*{.*?}\s*',
                    '',
                    last_message.content,
                    flags=re.DOTALL
                ).strip()
                # final_response = translate_text(text_response, preferred_language) if preferred_language != "en" else text_response
                # html_response = markdown.markdown(final_response)
                html_response = markdown.markdown(text_response)
                return jsonify({'response': html_response if html_response else "No valid response received."})

        return jsonify({'response': "No valid response received."})
    except Exception as e:
        return jsonify({'response': f'An error occurred: {e}'})


@app.route('/summarize_pdf', methods=['POST'])
def summarize_pdf_route():
    if 'pdf_file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    pdf_file = request.files['pdf_file']
    if pdf_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        extracted_text = extract_text_from_pdf(pdf_file)
        summary = summarize_text(extracted_text, sentence_count=10)
        # Check if the request wants a PDF download (e.g., /summarize_pdf?download=true)
        if request.args.get('download') == 'true':
            pdf_stream = generate_pdf_from_summary(summary)
            return send_file(pdf_stream, as_attachment=True, download_name="summary.pdf", mimetype="application/pdf")
        else:
            return jsonify({'summary': summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
