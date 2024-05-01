import streamlit as st
import os
import io
import pandas as pd
import boto3
import pytz
import base64
import fitz
from datetime import datetime
from PyPDF2 import PdfReader
from docx import Document
from botocore.exceptions import NoCredentialsError, ClientError


# Set Streamlit page configuration
st.set_page_config(page_title="Learning Repository")

# S3 Bucket name
BUCKET_NAME = os.environ["S3_BUCKET_NAME"]

# AWS User Credentials
AWS_ACCESS_KEY_ID=os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY=os.environ["AWS_SECRET_ACCESS_KEY"]
AWS_REGION=os.environ["AWS_REGION"]

# Kendra & S3 Config
s3_data_source_id=os.environ["S3_DATA_SOURCE_ID"]
kendra_index_id=os.environ["KENDRA_INDEX_ID"]

# Custom CSS to inject into the Streamlit app
css = """
<style>
table {
    text-align: left;
}
thead th {
    text-align: left;
}
</style>
"""

st.markdown(css, unsafe_allow_html=True)

# Get Singapore timezone
sgt_zone = pytz.timezone('Asia/Singapore')

# Check user type
if 'user_type' not in st.session_state:
    st.session_state['user_type'] = ""

ICON = "images/dataset.png"
def write_top_bar():
    col1, col2, col3 = st.columns([3,8,2])
    with col1:
        st.image(ICON, use_column_width=False,width=150)
    with col2:
        header = f"Learning Repository"
        powered_by = f"Powered by Amazon Kendra and S3"
        st.title(header)
        st.caption(powered_by)

# Function to upload files to S3
def upload_to_s3(local_file_path, s3_file_path):
    s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name=AWS_REGION)
    try:
        s3.upload_file(local_file_path, BUCKET_NAME, s3_file_path)
        return True
    except FileNotFoundError:
        st.error("File not found.")
        return False
    except NoCredentialsError:
        st.error("AWS credentials not available.")
        return False
    except ClientError as e:
        st.error(f"Error uploading file: {e}")
        return False

# Function to list files in S3 bucket
def list_s3_files():
    s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name=AWS_REGION)

    try:
        response = s3.list_objects(Bucket=BUCKET_NAME)
        files = [{'Document Name': os.path.basename(obj['Key']),
                  'Modified Date': obj['LastModified'].astimezone(sgt_zone),'File Size (KB)': obj['Size'] / 1024}
                   for obj in response.get('Contents', [])]
        sorted_files = sorted(files, key=lambda x: x['Modified Date'], reverse=True)
        return sorted_files
    except ClientError as e:
        st.error(f"Error accessing S3 bucket: {e}")
        return []

# Function to preview uploaded file
def file_preview(uploaded_file):
    st.subheader("Uploaded File Preview:")
    try:
        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        elif uploaded_file.type == "text/plain":
            df = pd.DataFrame({'Content': [uploaded_file.getvalue().decode('utf-8')]})
        elif uploaded_file.type == "application/pdf":
            # Save the PDF file to the root location temporarily
            pdf_file_path = f"{uploaded_file.name}"
            with open(pdf_file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
           
            # Display the first 3 sentences of a PDF file
            pdf_text = get_text_from_pdf(pdf_file_path)
            df = pd.DataFrame({'Content': [pdf_text]})
            
            # Preview the first page of the PDF
            preview_pdf(pdf_file_path)
            
            #Remove the PDF file
            os.remove(pdf_file_path)
            
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            #df = pd.DataFrame({'Info': ['Word document uploaded']})
            docx_text = get_text_from_docx(uploaded_file)
            df = pd.DataFrame({'Content': [docx_text]})
        else:
            st.error("Unsupported file format")
            return
        
        if uploaded_file:
            st.write(df)
    except Exception as e:
        st.error(f"Error reading file: {e}")


# Function to process and upload file to S3
def process_and_upload(uploaded_file):
    file_path = uploaded_file.name
    try:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        if upload_to_s3(file_path, uploaded_file.name):
            st.success(f"File successfully uploaded to S3 bucket: {BUCKET_NAME}")
            refresh_uploaded_files()
    except IOError as e:
        st.error(f"Error saving file: {e}")
        
    # Remove the local file after uploading to S3
    os.remove(file_path)

# Function to refresh and display updated files in S3
def refresh_uploaded_files():
    st.subheader("Updated Files in S3:")
    files_info = list_s3_files()
    
    if files_info:
        st.table(pd.DataFrame(files_info)) 
    else:
        st.info("No files in the S3 bucket.")
        
# Function to get text from PDF
def get_text_from_pdf(pdf_file):
    text = ""
    with open(pdf_file, "rb") as f:
        pdf_reader = PdfReader(f)
        for page_num in range(min(3, len(pdf_reader.pages))):
            text += pdf_reader.pages[page_num].extract_text()
    return text

# Function to Preview PDF      
def preview_pdf(pdf_file_path):
    doc = fitz.open(pdf_file_path)
    page = doc[0]
    pix = page.get_pixmap()
    pix1 = fitz.Pixmap(pix,0) if pix.alpha else pix
    img = pix1.tobytes("ppm")
           
    st.image(img,use_column_width=False,width=400)

# Function to get text from Docx
def get_text_from_docx(docx_file):
    doc = Document(docx_file)
    text = ""
    for paragraph in doc.paragraphs[:3]:
        text += paragraph.text + "\n"
    return text    
   

# Initialize the Kendra client
def init_kendra_client():
    try:
        return boto3.client('kendra', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name=AWS_REGION)
    except NoCredentialsError:
        st.error("AWS credentials not found. Please configure your credentials.")
        return None



# Function to start the data source sync job
def start_sync(kendra_client):
    try:
        response = kendra_client.start_data_source_sync_job(
            Id=s3_data_source_id,
            IndexId=kendra_index_id
        )
        return response
    except Exception as e:
        st.error(f"Error starting sync job: {e}")
        return None
    
# Function to check status of the data source sync job   
def check_sync(kendra_client):
    try:
        response = kendra_client.list_data_source_sync_jobs(
            Id=s3_data_source_id,
            IndexId=kendra_index_id
        )
        return response
    except Exception as e:
        st.error(f"Error checking sync job: {e}")
        return None

def display_dict_as_table(data_dict):
    # Flatten the nested dictionary
    flat_dict = {}
    for key, value in data_dict.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat_dict[f"{key}_{sub_key}"] = sub_value
        else:
            flat_dict[key] = value
    
    # Convert to DataFrame
    df = pd.DataFrame([flat_dict])

    # Convert string representations of datetime to actual datetime objects
    for col in df.columns:
        if 'Time' in col:
            df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)
            df[col] = df[col].apply(lambda x: x.astimezone(sgt_zone) if isinstance(x, datetime) else x)
    # Display the DataFrame in Streamlit
    st.table(df)


def get_file_from_s3(s3_file_path):
    s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name=AWS_REGION)
    try:
        file_obj = io.BytesIO()
        s3.download_fileobj(BUCKET_NAME, s3_file_path, file_obj)
        file_obj.seek(0)  # Go to the start of the file
        return file_obj
    except NoCredentialsError:
        st.error("AWS credentials not available.")
        return None
    except ClientError as e:
        st.error(f"Error downloading file: {e}")
        return None
            

# Main function for Streamlit app
def main():
    # Page and sidebar headers
    write_top_bar()

    # Display uploaded files in S3
    st.markdown("### Uploaded Files in S3:")
    files_info = list_s3_files()
    
    if files_info:
        # Convert the list of dictionaries to a DataFrame for easier manipulation
        df = pd.DataFrame(files_info)

        # Format the 'Modified Date' and 'File Size (KB)'
        df['Modified Date'] = pd.to_datetime(df['Modified Date']).dt.strftime("%Y-%m-%d %H:%M:%S")
        df['File Size (KB)'] = df['File Size (KB)'].map(lambda x: f"{x:.2f} KB")

        # Initialize a column for the download buttons
        df['Download'] = ""

        for index, file_info in df.iterrows():
            s3_file_path = file_info['Document Name']

            # Get the file object from S3
            file_obj = get_file_from_s3(s3_file_path)
            if file_obj:
                # Convert file object to a download-able format
                b64 = base64.b64encode(file_obj.read()).decode()
                href = f'<a href="data:file/octet-stream;base64,{b64}" download="{s3_file_path}">Download</a>'
                df.at[index, 'Download'] = href
            else:
                df.at[index, 'Download'] = "Unavailable"

        # Use Streamlit's built-in functionality to display the DataFrame
        st.write(df.to_html(escape=False), unsafe_allow_html=True)
        
    if st.session_state["user_type"] == "ADMIN":
        st.markdown("")              
        # File uploader
        uploaded_file = st.file_uploader("Choose a file", type=["csv","xlsx","txt","pdf","docx"])
        if uploaded_file:
            st.success("File successfully uploaded.")
            file_preview(uploaded_file)

            # Upload to S3 button
            if st.button("Upload to S3"):
                process_and_upload(uploaded_file)

        st.markdown("---")
        st.markdown("### Amazon Kendra")
        kendra_client = init_kendra_client()
        if kendra_client:
            st.success("Connected to Amazon Kendra service")
        else:
            st.error("Failed connection to Amazon Kendra service")

        col1, col2 = st.columns([3,3])
        sync_start_response = {}
        sync_status_response = {}
        with col1:
            if st.button("Start Sync"):
                sync_start_response = start_sync(kendra_client)
        if sync_start_response:
            st.success("Sync job started successfully!")
            #st.json(sync_start_response)

        with col2:  
            if st.button("Check Sync"):
                sync_status_response = check_sync(kendra_client)["History"][0]
        if sync_status_response:
            st.success("Sync job status retrieved successfully!")
            display_dict_as_table(sync_status_response)

            
if __name__ == "__main__":
    if st.session_state.user_type == "":
        st.error("Please proceed to login...")
    else:
        main()