import streamlit as st
import shutil
import os

from chroma_handler import process_pdf
from chatbot2 import chatbot_response
from config import upload_dir


if "disabled" not in st.session_state:
    st.session_state["disabled"] = False


def sidebar():

    def disabled():
        st.session_state["disabled"] = True
    
    def enabled():
        st.session_state["disabled"] = False

    sidebar = st.sidebar
    sidebar.title("File Upload Section")
    with sidebar.form(key='file-handler', clear_on_submit=True):
        uploaded_files = st.file_uploader(label="PDF Files",
                                          type="pdf",
                                          accept_multiple_files=True,
                                          key="pdf_upload", disabled=st.session_state.disabled)
        submitted = st.form_submit_button("Upload", disabled=st.session_state["disabled"])
        if submitted:
            disabled()
            with st.spinner('Wait for the process to finish...'):
                if os.path.exists(upload_dir):
                    shutil.rmtree(upload_dir)
                os.makedirs(upload_dir)
                for fl in uploaded_files:
                    with open(os.sep.join([upload_dir, fl.name]), "wb") as f:
                        f.write(fl.getbuffer())
                for f in uploaded_files:
                    process_pdf(os.sep.join([upload_dir, f.name]))
                st.write("Upload successful!")
                enabled()




def main():
    sidebar()
    st.title("Chat bot")

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
            # bot_response = f"Echo: {query}"
            bot_response = chatbot_response(query)
            st.markdown(bot_response)
            st.session_state.messages.append({"role": "bot",
                                              "message": bot_response})


if __name__ == '__main__':
    main()
