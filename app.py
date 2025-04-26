# app.py
import streamlit as st
from time import sleep
from dotenv import load_dotenv
import os
from inference import chat_pipeline # Assuming your inference logic is here
import uuid
import gc # Import garbage collector
from qdrant_client import QdrantClient
# Use LangChain's FastEmbed wrapper for better integration
from langchain_community.embeddings import FastEmbedEmbeddings
from langsmith import Client as LangSmithClient
from langchain_groq import ChatGroq

# --- Configuration for Memory Optimization ---
# Option 1: Limit history length passed to the backend pipeline
# This reduces the size of input data sent to the LLM and embedding model.
MAX_HISTORY_FOR_PIPELINE = 10 # Keep last N messages (adjust as needed)

# Option 2: Truncate history stored in session state (UX CHANGE!)
# Set to True to enable truncation of the *full* chat history stored in browser memory.
# This saves memory in the browser session state but means older messages are lost
# for display and potential future processing.
TRUNCATE_STORED_HISTORY = False
MAX_STORED_MESSAGES = 50 # Max messages per chat IF TRUNCATE_STORED_HISTORY is True


# Load environment variables
load_dotenv()

# --- Page Configuration (Professional Look) ---
st.set_page_config(
    page_title="Bhagavad Gita Assistant",
    page_icon="🕉️",
    layout="wide", # Use wide layout for chat interfaces
    initial_sidebar_state="expanded", # Keep sidebar open initially
    menu_items={
        'About': "Ask questions about the Bhagavad Gita using AI."
    }
)

# --- Initialization ---
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}
    st.session_state.chat_id_counter = 0
    st.session_state.current_chat_id = None
    print("Initialized chat_histories session state.")


@st.cache_resource
def initialize():
    try:
        # Initialize embeddings. The size of the model affects memory.
        # embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-large-en-v1.5", cache_dir="./cache")
        embed_model = FastEmbedEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1", cache_dir="./cache")

        # Initialize LangSmith Client (lightweight)
        LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
        langsmith_client = None # Start with None
        if LANGSMITH_API_KEY:
            langsmith_client = LangSmithClient(api_key=LANGSMITH_API_KEY)
        else:
            print("[WARN] LANGSMITH_API_KEY not set. LangSmith tracing disabled.")

        # Initialize LLM (another primary memory consumer). Model size matters.
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")
        # Using a common Groq model. Changing model_name would impact memory and functionality.
        llm = ChatGroq(model_name="deepseek-r1-distill-llama-70b", api_key=groq_api_key)

        # Initialize Qdrant Client (lightweight connection manager)
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        if not qdrant_url or not qdrant_api_key:
            raise ValueError("QDRANT_URL or QDRANT_API_KEY environment variable not set.")
        qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            prefer_grpc=True
        )
        return embed_model, langsmith_client, llm, qdrant_client
    except Exception as e:
        print(f"Initialization failed: {e}")
        # In a service, you might log and return an error rather than exiting.
        # For a standalone script, exit(1) is appropriate.
        exit(1)

def generate_chat_id():
    return str(uuid.uuid4())

def create_new_chat(switch_to_new=True):
    st.session_state.chat_id_counter += 1
    new_chat_id = generate_chat_id()
    new_chat_name = f"Conversation {st.session_state.chat_id_counter}"
    st.session_state.chat_histories[new_chat_id] = {"name": new_chat_name, "messages": []}
    print(f"Created new chat: ID={new_chat_id}, Name={new_chat_name}")
    if switch_to_new:
        st.session_state.current_chat_id = new_chat_id
        print(f"Switched to new chat: ID={new_chat_id}")
        # Use st.rerun to update the UI immediately
        st.rerun()
    # No need for rerun if not switching, the chat appears in the list on next interaction

#load embeddings, llm, and qdrant client
embed_model, langsmith_client, llm, qdrant_client = initialize()


# Ensure at least one chat exists on first load or if state is broken
if not st.session_state.chat_histories:
    print("No chat histories found, creating initial chat.")
    st.session_state.chat_id_counter = 1
    initial_chat_id = generate_chat_id()
    st.session_state.chat_histories[initial_chat_id] = {"name": "Conversation 1", "messages": []}
    st.session_state.current_chat_id = initial_chat_id
elif st.session_state.current_chat_id not in st.session_state.chat_histories:
    # This handles cases where a chat ID might be in the state but no longer in histories
    print(f"Current chat ID '{st.session_state.current_chat_id}' invalid or not found. Switching to first available chat.")
    # Safely get the first key if histories is not empty
    st.session_state.current_chat_id = next(iter(st.session_state.chat_histories), None)
    if st.session_state.current_chat_id is None:
         # This case should ideally not happen if the block above creates an initial chat,
         # but as a safeguard:
         print("No chats available after checking state, creating new one.")
         create_new_chat(switch_to_new=True) # Create and switch
    else:
        st.rerun() # Rerun to load the valid chat


# --- Helper Function to Truncate Stored History (if enabled) ---
def truncate_history_if_needed(messages_list):
    """Truncates the messages list in-place if TRUNCATE_STORED_HISTORY is True."""
    if TRUNCATE_STORED_HISTORY and len(messages_list) > MAX_STORED_MESSAGES:
        # Keep only the last MAX_STORED_MESSAGES
        # Modify list in-place using slicing assignment
        print(f"Truncating stored history from {len(messages_list)} to {MAX_STORED_MESSAGES} messages.")
        messages_list[:] = messages_list[-MAX_STORED_MESSAGES:]
        # Optional: Explicitly call GC after potential large list modification
        # gc.collect() # Can be uncommented if you suspect major memory retention here


# --- Sidebar Styling and Functionality ---
with st.sidebar:
    # st.image(...) # Add your image here if needed
    st.sidebar.title("🕉️ Gita Chats")
    st.sidebar.caption("Manage your explorations")

    # Add a "Start New Conversation" button
    if st.button("➕ Start New Conversation", use_container_width=True, type="primary"):
        create_new_chat() # This function now handles rerunning

    st.divider()

    # Chat Selection Radio Buttons
    chat_options = list(st.session_state.chat_histories.keys())
    # Ensure chat_options is not empty before proceeding
    if not chat_options:
        st.error("Error: No conversations found.")
        # This state should ideally be caught by the initialization block,
        # but as a safeguard, stop execution if we somehow reach here with no chats.
        st.stop()

    chat_display_names = {
        chat_id: data["name"]
        for chat_id, data in st.session_state.chat_histories.items()
    }

    def format_chat_option(chat_id):
        """Formats the chat ID for display in the radio button."""
        return chat_display_names.get(chat_id, f"Chat {chat_id[:4]}...")

    try:
        # Find the index of the current chat ID in the options list
        current_chat_index = chat_options.index(st.session_state.current_chat_id)
    except ValueError:
        # If the current chat ID is somehow invalid, default to the first chat
        print(f"Current chat ID '{st.session_state.current_chat_id}' not found in options. Defaulting to first chat.")
        current_chat_index = 0
        st.session_state.current_chat_id = chat_options[0] # Update state
        st.rerun() # Rerun to sync UI and state

    # Use the radio button to select a chat
    selected_chat_id = st.radio(
        "Your Conversations:",
        options=chat_options,
        format_func=format_chat_option,
        index=current_chat_index,
        key=f"chat_selector_{st.session_state.current_chat_id}" # Unique key for reruns
    )

    # If a different chat was selected via the radio button, update state and rerun
    if selected_chat_id != st.session_state.current_chat_id:
        st.session_state.current_chat_id = selected_chat_id
        print(f"Switched to chat: ID={selected_chat_id}")
        st.rerun() # Rerun to load the selected chat's history

    st.divider()

    # Delete Chat Button and Confirmation
    # Only show delete option if there's more than one chat
    if len(st.session_state.chat_histories) > 1:
        # Use columns just for placing the initial delete button on the right
        col1_delete, col2_delete = st.columns([0.8, 0.2])
        with col1_delete:
            # Use a placeholder in the first column to display confirmation later
            delete_confirmation_placeholder = st.empty()
        with col2_delete:
             # The actual delete button in the second column
             # Set a unique key
             if st.button("🗑️", help="Delete current conversation", key=f"delete_btn_{st.session_state.current_chat_id}"):
                 # Set a session state flag to indicate confirmation is needed for this chat
                 st.session_state._show_delete_confirm = st.session_state.current_chat_id
                 st.rerun() # Rerun to show the confirmation UI

        # This block runs on the rerun triggered by clicking the delete button
        # Check if the confirmation flag is set for the currently displayed chat
        if "_show_delete_confirm" in st.session_state and st.session_state._show_delete_confirm == st.session_state.current_chat_id:
            # Display the confirmation UI *inside the placeholder* in the first column
            # IMPORTANT: Do NOT use .columns() inside this placeholder in the sidebar
            with delete_confirmation_placeholder.container(): # Use container to group elements
                st.warning(f"Delete '{chat_display_names.get(st.session_state.current_chat_id, 'this chat')}'? ")
                # Place buttons directly, they will stack vertically
                confirm_delete = st.button("Confirm Delete", key=f"confirm_delete_{st.session_state.current_chat_id}", type="primary")
                cancel_delete = st.button("Cancel", key=f"cancel_delete_{st.session_state.current_chat_id}")

                # Handle confirmation actions
                if confirm_delete:
                    chat_to_delete = st.session_state.current_chat_id
                    del st.session_state.chat_histories[chat_to_delete] # Delete the history from state
                    print(f"Deleted chat: ID={chat_to_delete}")
                    # Trigger garbage collection explicitly after deleting a potentially large object
                    gc.collect()
                    print("Triggered garbage collection.")
                    # Switch to the first available chat
                    st.session_state.current_chat_id = next(iter(st.session_state.chat_histories))
                    # Clear the confirmation flag
                    del st.session_state._show_delete_confirm
                    # Clear the placeholder content
                    delete_confirmation_placeholder.empty()
                    st.rerun() # Rerun to update the UI (show new chat, remove confirmation)

                if cancel_delete:
                    # Clear the confirmation flag
                    del st.session_state._show_delete_confirm
                    # Clear the placeholder content
                    delete_confirmation_placeholder.empty()
                    st.rerun() # Rerun to remove confirmation and show delete button again
    elif st.session_state.chat_histories:
          # Message shown if there's only one chat and it cannot be deleted
          st.caption("Cannot delete the last conversation.")

    st.divider()
    st.info("Seek wisdom, ask questions about the divine knowledge within the Gita.", icon="💡")
    # Add note about memory setting
    if TRUNCATE_STORED_HISTORY:
        st.warning(f"Memory Saving Active: Stored chat history is limited to the last {MAX_STORED_MESSAGES} messages per chat.", icon="⚠️")
    # Add note about history passed to backend
    st.info(f"Note: The AI pipeline processes only the last {MAX_HISTORY_FOR_PIPELINE} messages for context.", icon="ℹ️")


# --- Main Chat Area Styling ---
st.title("🕉️ Bhagavad Gita AI Assistant")
st.caption("Your companion for exploring the timeless wisdom of the Gita.")

# Ensure we have a valid chat selected before proceeding
if not st.session_state.current_chat_id or st.session_state.current_chat_id not in st.session_state.chat_histories:
    st.error("Please select or start a new conversation from the sidebar.")
    st.stop() # Stop execution if no valid chat is found

# Get the messages for the currently selected chat
current_chat_data = st.session_state.chat_histories[st.session_state.current_chat_id]
current_messages = current_chat_data["messages"]

# Display the current chat name
st.header(f"📜 {current_chat_data['name']}")
st.divider()

# Use a container for the chat messages display area
chat_container = st.container()
with chat_container:
    # Display existing messages in the chat history
    for msg in current_messages:
        role = msg.get("role")
        content = msg.get("content", "")
        avatar_icon = "👤" if role == "user" else "🕉️"
        with st.chat_message(role, avatar=avatar_icon):
            if role == "assistant":
                try:
                    # Parse the assistant message content to separate thinking and answer
                    parts = content.split("<split>", 1)
                    thinking = parts[0].strip() if len(parts) > 1 else ""
                    answer = parts[1].strip() if len(parts) > 1 else content.strip() # Use full content if no split

                    if thinking:
                        # Display thinking process in an expander
                        with st.expander("🧠 Thinking Process...", expanded=False):
                            st.info(thinking)
                    # Display the main answer using markdown
                    st.markdown(answer)
                except Exception as e:
                    # Fallback display if parsing fails
                    st.error(f"Error displaying message: {e}")
                    st.markdown(content) # Display raw content if parsing fails
            else:
                # Display user message using markdown
                st.markdown(content)

# --- Chat Input ---
# Get user input from the chat input box
prompt = st.chat_input(f"Ask about the Gita in '{current_chat_data['name']}'...")

if prompt:
    # Add the user's message to the current conversation's history
    user_message = {"role": "user", "content": prompt}
    current_messages.append(user_message)

    # Apply truncation to the stored history AFTER adding the new message (if enabled)
    # This ensures the latest message is always included before potential truncation.
    truncate_history_if_needed(current_messages)

    # Rerun the app to immediately display the user's message
    # This also clears the chat input box
    st.rerun()

# After the rerun from prompt submission, the user message is displayed.
# Now, if the last message was from the user (meaning we just added it),
# generate the assistant's response.
if current_messages and current_messages[-1]["role"] == "user":
    # Display a placeholder for the assistant's response while it's being generated
    with chat_container: # Ensure the assistant message appears in the chat area
        with st.chat_message("assistant", avatar="🕉️"):
            message_placeholder = st.empty() # Placeholder for the answer text
            thinking_placeholder = st.empty() # Placeholder for the thinking expander
            full_response_content = "" # Variable to store the complete response from pipeline

            with st.spinner("🕉️ Seeking divine wisdom..."):
                try:
                    # --- Call the RAG Pipeline ---
                    # Option 1: Pass only the TAIL of the history to the backend pipeline
                    # This is a memory optimization for the *backend inference process*.
                    # The full history is still stored in Streamlit's session state
                    # unless TRUNCATE_STORED_HISTORY is True.
                    # Use slicing to get the last N messages
                    history_for_pipeline = current_messages[-MAX_HISTORY_FOR_PIPELINE:]
                    print(f"Passing last {len(history_for_pipeline)} messages to chat_pipeline.")

                    # Invoke the chat pipeline function from inference.py
                    # The pipeline returns the formatted response string ("thinking<split>answer")
                    response = chat_pipeline(history_for_pipeline, embed_model, langsmith_client, llm, qdrant_client)
                    full_response_content = response # Store the full response string

                    # --- Parse and Display Assistant Response ---
                    # Split the response into thinking and answer parts
                    parts = response.split("<split>", 1)
                    thinking = parts[0].strip() if len(parts) > 1 else ""
                    answer = parts[1].strip() if len(parts) > 1 else response.strip() # Use full response if no split

                    # Display the thinking process if available
                    if thinking:
                        with thinking_placeholder.expander("🧠 Thinking Process...", expanded=False):
                             st.info(thinking)

                    # Simulate typing animation for the answer
                    rendered_answer = ""
                    words = answer.split()
                    for i, word in enumerate(words):
                        rendered_answer += word + " "
                        # Update the placeholder with partial answer + cursor
                        message_placeholder.markdown(rendered_answer + ("▌" if i < len(words) - 1 else ""))
                        sleep(0.03) # Adjust typing speed here

                    # Display the final complete answer without the cursor
                    message_placeholder.markdown(rendered_answer.strip())

                except Exception as e:
                    # Handle any errors during the pipeline call or processing
                    st.error(f"An error occurred while seeking wisdom: {e}")
                    full_response_content = f"Error: Could not get response. Details: {e}"
                    message_placeholder.markdown(full_response_content) # Display error in place

            # --- Store Assistant Response ---
            # Add the assistant's full response (including <think> if present) to the history
            # This keeps the original response format for display on subsequent page loads.
            assistant_message = {"role": "assistant", "content": full_response_content}
            current_messages.append(assistant_message)

            # Apply truncation to the stored history AFTER adding the assistant's message (if enabled)
            truncate_history_if_needed(current_messages)

            # Optional: Rerun after response is added. Usually not needed as the page
            # is already updated by the typing animation and message append.
            # st.rerun()