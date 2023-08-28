import streamlit as st
import shutil
import os

from chroma_handler import process_pdf
from chatbot_as_function import conv_agent
from config import upload_dir


def sidebar():
    sidebar = st.sidebar
    sidebar.title("File Uploader")
    with sidebar.form(key='file-handler', clear_on_submit=True):
        uploaded_files = st.file_uploader(label="PDF Files",
                                          type="pdf",
                                          accept_multiple_files=True,
                                          key="pdf_upload")
        submitted = st.form_submit_button("Upload")
        if uploaded_files and submitted:
            # if os.path.exists(upload_dir):
            #     shutil.rmtree(upload_dir)
            # No need to remove path, if path doesn't exists, just create one
            with st.spinner("Please wait for the process to finish..."):
                if not os.path.exists(upload_dir):
                    os.makedirs(upload_dir)
                for fl in uploaded_files:
                    with open(os.sep.join([upload_dir, fl.name]), "wb") as f:
                        f.write(fl.getbuffer())
                for f in uploaded_files:
                    process_pdf(os.sep.join([upload_dir, f.name]))
            st.write("Upload successful!")


def main():
    sidebar()
    st.title("AJ4X - You friendly AI bot")

    # handle chats till now
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["message"])

    if query := st.chat_input("Ask a question!"):
        with st.chat_message("user"):
            st.markdown(query)
            st.session_state.messages.append({"role": "user",
                                              "message": query})
        with st.chat_message("bot"):
            with st.spinner("Thinking..."):
                bot_response = conv_agent(query)
            st.markdown(bot_response)
            st.session_state.messages.append({"role": "assistant",
                                              "message": bot_response})


if __name__ == '__main__':
    main()
