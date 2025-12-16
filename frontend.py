import streamlit as st
import requests
import uuid

# --- CONFIGURATION ---
BASE_URL = "http://192.168.0.124:8000"
CHAT_URL = f"{BASE_URL}/chat/"
UPLOAD_URL = f"{BASE_URL}/upload-pdf/"

st.set_page_config(page_title="Project ISI-TIH Chat", page_icon="🤖", layout="wide")
st.title("🤖 ISI-TIH Chat: Multimodal RAG Chat")

# --- SESSION STATE INITIALIZATION ---
# 1. Generate a unique Session ID for this specific user/tab
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# 2. Initialize chat history list if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- DISPLAY CHAT HISTORY ---
# Iterate through the history and display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- SIDEBAR: DOCUMENT UPLOAD ---
with st.sidebar:
    st.header("📂 Manage Documents")
    st.write("Upload a PDF to add it to the knowledge base.")
    
    # File Uploader Widget
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    # Upload Button
    if uploaded_file is not None:
        if st.button("Upload and Process"):
            with st.spinner("Uploading to backend..."):
                try:
                    # Prepare the file for the API request
                    # 'file' matches the name of the argument in FastAPI: file: UploadFile = File(...)
                    files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                    
                    response = requests.post(UPLOAD_URL, files=files)
                    
                    if response.status_code == 202:
                        st.success(f"✅ '{uploaded_file.name}' uploaded successfully!")
                        st.info("The server is now processing chunks and embeddings in the background.")
                    else:
                        st.error(f"❌ Upload failed. Status: {response.status_code}")
                        st.error(response.text)
                        
                except Exception as e:
                    st.error(f"Connection Error: {e}")

    st.markdown("---")
    st.markdown("**Session ID:**")
    st.caption(st.session_state.session_id)
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- MAIN CHAT INTERFACE ---

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # 1. Display User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 2. Add to local history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 3. Call the Backend API
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")

        try:
            payload = {
                "question": prompt,
                "session_id": st.session_state.session_id
            }
            
            response = requests.post(CHAT_URL, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                answer = data["answer"]
                sources = data["sources"]
                
                # Format the output with sources
                source_text = ", ".join([f"`{s}`" for s in sources])
                full_response = f"{answer}\n\n---\n**Sources:** {source_text}"
                message_placeholder.markdown(full_response)
                
                # 4. Add Assistant response to local history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            else:
                error_msg = f"Error: {response.status_code} - {response.text}"
                message_placeholder.error(error_msg)

        except Exception as e:
            message_placeholder.error(f"Connection Error: {e}")