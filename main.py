from google.cloud import storage
from google.cloud import aiplatform
from google.cloud import secretmanager
import os

# Environment variables (set in Cloud Run)
INPUT_BUCKET = os.environ.get("INPUT_BUCKET")
OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET")
PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = os.environ.get("LOCATION", "us-central1") # Vertex AI location
SECRET_ID = os.environ.get("SECRET_ID", "gcs-access-key")

def access_secret_version(secret_id, project_id="your-project-id"):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

def transcribe_pdf(pdf_uri, prompt="Transcribe this PDF into plain English text and separate it into paragraphs. Return only the text, no markup"):
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    
    model = "gemini-2.5-pro-001" # Replace with the correct model name if needed
    
    response = aiplatform.generative_model.GenerativeModel(model_name=model).generate_content(
        contents=prompt + f" PDF:{pdf_uri}", # Pass the PDF URI to the model
        generation_config={
            "max_output_tokens": 2048
        },
        stream=False
    )
    return response.text

def process_pdfs(request):
    # Access the secret
    credentials_json = access_secret_version(SECRET_ID, PROJECT_ID)
    
    # Authenticate with the service account (using the secret)
    storage_client = storage.Client.from_service_account_info(json.loads(credentials_json))

    input_bucket = storage_client.bucket(INPUT_BUCKET)
    output_bucket = storage_client.bucket(OUTPUT_BUCKET)

    for blob in input_bucket.list_blobs():
        if blob.name.endswith(".pdf"):
            print(f"Processing {blob.name}")
            pdf_uri = f"gs://{INPUT_BUCKET}/{blob.name}"
            try:
                transcription = transcribe_pdf(pdf_uri)
                # Save the transcription to the output bucket
                output_blob = output_bucket.blob(blob.name.replace(".pdf", ".txt"))
                output_blob.upload_from_string(transcription, content_type="text/plain")
                print(f"Transcription saved to gs://{OUTPUT_BUCKET}/{blob.name.replace('.pdf', '.txt')}")
            except Exception as e:
                print(f"Error processing {blob.name}: {e}")

    return "PDF processing complete"

if __name__ == "__main__":
    # This is for local testing.  Set environment variables before running.
    os.environ["INPUT_BUCKET"] = "your-input-bucket-name"
    os.environ["OUTPUT_BUCKET"] = "your-output-bucket-name"
    os.environ["PROJECT_ID"] = "your-project-id"
    os.environ["SECRET_ID"] = "gcs-access-key" # Or the name you gave it
    import json
    process_pdfs(None) # Doesn't use the request object in this example
