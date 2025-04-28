import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
import os
from dotenv import load_dotenv


load_dotenv()  # Load environment variables from .env file

# Now you can access them:
sender_email = os.getenv('SENDER_EMAIL')
def get_hr_email(csv_file='employee_data.csv'):
    """Fetch HR email from the employee database CSV"""
    try:
        df = pd.read_csv(csv_file)
        hr_row = df[df['position'].str.lower() == 'hr']
        if not hr_row.empty:
            return hr_row.iloc[0]['email']
    except Exception as e:
        print(f"Error reading CSV: {e}")
    return None

def send_email_to_hr(employee_name):
    """Send meeting request email to HR"""
    sender_email = os.getenv('SENDER_EMAIL')
    sender_password = os.getenv('SENDER_PASSWORD')
    hr_email = get_hr_email()  # <-- Fetch HR email here
    print(f"HR Email: {get_hr_email()}")
    if not sender_email or not sender_password or not hr_email:
        print("Sender or HR email credentials not found.")
        return False

    subject = "Meeting Request from Employee"
    body = f"Dear HR,\n\nEmployee {employee_name} has requested to schedule a meeting with you. Please respond to them at your convenience.\n\nBest Regards,\nYour Company Bot"

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = hr_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, hr_email, msg.as_string())
        server.quit()
        print("Email sent to HR successfully!")
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False