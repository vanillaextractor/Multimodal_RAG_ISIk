import streamlit as st
import requests
import uuid

# --- CONFIGURATION ---
BASE_URL = "http://localhost:8000"
CHAT_URL = f"{BASE_URL}/chat/"
UPLOAD_URL = f"{BASE_URL}/upload-pdfs/"

st.set_page_config(page_title="Project ISI-TIH Chat", page_icon="🤖", layout="wide")
st.title("🤖 ISI-TIH Chat: Multimodal RAG Chat")

# --- SESSION STATE INITIALIZATION ---
# 1. Generate a unique Session ID for this specific user/tab
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_document" not in st.session_state:
    st.session_state.current_document = None

# --- DISPLAY CHAT HISTORY ---
# Iterate through the history and display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- SIDEBAR: DOCUMENT UPLOAD ---
with st.sidebar:
    st.header("📂 Manage Documents")
    st.write("Upload PDF files to add them to the knowledge base.")
    
    # File Uploader Widget
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    
    # Upload Button
    if uploaded_files:
        if st.button("Upload and Process"):
            with st.spinner("Uploading to backend..."):
                try:
                    # Prepare the files for the API request
                    # 'files' matches the name of the argument in FastAPI: files: List[UploadFile] = File(...)
                    files_payload = [("files", (f.name, f.getvalue(), "application/pdf")) for f in uploaded_files]
                    
                    response = requests.post(UPLOAD_URL, files=files_payload)
                    
                    if response.status_code == 200:
                        st.session_state.current_document = None
                        st.success(f"✅ {len(uploaded_files)} files fully processed and ready for your queries!")
                    else:
                        st.error(f"❌ Upload failed. Status: {response.status_code}")
                        st.error(response.text)
                        
                except Exception as e:
                    st.error(f"Connection Error: {e}")

    st.markdown("---")
    st.markdown("**Session ID:**")
    st.caption(st.session_state.session_id)
    if st.button("Clear Chat History"):
        try:
            requests.post(f"{BASE_URL}/clear-db/")
            st.success("Database cleared successfully.")
        except Exception as e:
            st.error(f"Failed to clear database: {e}")
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
            if st.session_state.current_document:
                payload["document_name"] = st.session_state.current_document
            
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