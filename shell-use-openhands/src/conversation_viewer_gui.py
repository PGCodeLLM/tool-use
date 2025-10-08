# conversation_viewer.py
import streamlit as st

# Example conversation trace (replace with your own data)
conversation_trace = [
    {"role": "system", "content": "You are ChatGPT, a helpful assistant."},
    {"role": "user", "content": "Hello! How are you?"},
    {"role": "assistant", "content": "I'm doing well, thank you! How can I help you today?"},
    {"role": "user", "content": "Explain what reinforcement learning is in simple terms."},
    {"role": "assistant", "content": "Reinforcement learning is like training a pet with rewards and penalties. "
                                     "The system learns by trial and error, getting better over time."}
]

st.set_page_config(page_title="LLM Conversation Viewer", layout="wide")
st.title("🤖 LLM Conversation Trace Viewer")

# Sidebar for metadata
st.sidebar.header("Conversation Metadata")
st.sidebar.write(f"Total turns: {len(conversation_trace)}")

# Main visualization
for i, message in enumerate(conversation_trace):
    role = message["role"].capitalize()
    content = message["content"]

    if message["role"] == "user":
        st.chat_message("user").write(content)
    elif message["role"] == "assistant":
        st.chat_message("assistant").write(content)
    else:
        with st.expander(f"{role} Message"):
            st.write(content)
