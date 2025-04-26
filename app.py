import streamlit as st
from time import sleep
from dotenv import load_dotenv
import os
from inference import chat_pipeline # Assuming your inference logic is here
import uuid
import gc # Import garbage collector

# --- Configuration for Memory Optimization ---
# Option 1: Limit history length passed to the backend pipeline
MAX_HISTORY_FOR_PIPELINE = 10 # Keep last 5 user/assistant pairs (adjust as needed)

# Option 2: Truncate history stored in session state (UX CHANGE!)
# Set to True to enable truncation, False to keep full history in memory
TRUNCATE_STORED_HISTORY = False
MAX_STORED_MESSAGES = 50 # Max messages per chat IF TRUNCATE_STORED_HISTORY is True


# Load environment variables
load_dotenv()

# --- Page Configuration (Professional Look) ---
# (Keep existing st.set_page_config)
st.set_page_config(
    page_title="Bhagavad Gita Assistant",
    page_icon="🕉️",
    layout="wide",  # Use wide layout for chat interfaces
    initial_sidebar_state="expanded", # Keep sidebar open initially
    menu_items={
        'About': "Ask questions about the Bhagavad Gita using AI."
    }
)

# --- Initialization (Keep existing logic) ---
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}
    st.session_state.chat_id_counter = 0
    st.session_state.current_chat_id = None

def generate_chat_id():
    return str(uuid.uuid4())

def create_new_chat(switch_to_new=True):
    st.session_state.chat_id_counter += 1
    new_chat_id = generate_chat_id()
    new_chat_name = f"Conversation {st.session_state.chat_id_counter}"
    st.session_state.chat_histories[new_chat_id] = {"name": new_chat_name, "messages": []}
    if switch_to_new:
        st.session_state.current_chat_id = new_chat_id
        print(f"Created and switched to new chat: ID={new_chat_id}, Name={new_chat_name}")
        st.rerun()
    else:
         print(f"Created new chat (no switch): ID={new_chat_id}, Name={new_chat_name}")

# Ensure at least one chat exists on first load or if state is broken
if not st.session_state.chat_histories:
    print("No chat histories found, creating initial chat.")
    st.session_state.chat_id_counter = 1
    initial_chat_id = generate_chat_id()
    st.session_state.chat_histories[initial_chat_id] = {"name": "Conversation 1", "messages": []}
    st.session_state.current_chat_id = initial_chat_id
elif st.session_state.current_chat_id not in st.session_state.chat_histories:
    print(f"Current chat ID '{st.session_state.current_chat_id}' invalid, switching.")
    st.session_state.current_chat_id = next(iter(st.session_state.chat_histories))


# --- Helper Function to Truncate Stored History (if enabled) ---
def truncate_history_if_needed(messages_list):
    if TRUNCATE_STORED_HISTORY and len(messages_list) > MAX_STORED_MESSAGES:
        # Keep only the last MAX_STORED_MESSAGES
        # Modify list in-place to update session state correctly
        messages_list[:] = messages_list[-MAX_STORED_MESSAGES:]
        print(f"Truncated stored history to {MAX_STORED_MESSAGES} messages.")


# --- Sidebar Styling and Functionality ---
with st.sidebar:
    # (Keep existing sidebar image/title/caption)
    # st.image(...)
    st.sidebar.title("🕉️ Gita Chats")
    st.sidebar.caption("Manage your explorations")

    if st.button("➕ Start New Conversation", use_container_width=True, type="primary"):
        create_new_chat()

    st.divider()

    # (Keep existing chat selection logic - radio buttons)
    chat_options = list(st.session_state.chat_histories.keys())
    chat_display_names = {
        chat_id: data["name"]
        for chat_id, data in st.session_state.chat_histories.items()
    }
    def format_chat_option(chat_id):
        return chat_display_names.get(chat_id, f"Chat {chat_id[:4]}...")
    try:
        current_chat_index = chat_options.index(st.session_state.current_chat_id)
    except ValueError:
        current_chat_index = 0
        if chat_options: st.session_state.current_chat_id = chat_options[0]
        else: st.error("Error: No chats available."); st.stop()

    selected_chat_id = st.radio(
        "Your Conversations:", options=chat_options, format_func=format_chat_option,
        index=current_chat_index, key=f"chat_selector_{st.session_state.current_chat_id}"
    )
    if selected_chat_id != st.session_state.current_chat_id:
        st.session_state.current_chat_id = selected_chat_id
        print(f"Switched to chat: ID={selected_chat_id}")
        st.rerun()

    st.divider()

    # (Keep existing delete button logic)
    if len(st.session_state.chat_histories) > 1:
        col1, col2 = st.columns([0.8, 0.2])
        with col1: delete_placeholder = st.empty()
        with col2:
             if st.button("🗑️", help="Delete current conversation", key=f"delete_btn_{st.session_state.current_chat_id}"):
                 delete_placeholder.warning(f"Delete '{chat_display_names[st.session_state.current_chat_id]}'? ")
                 confirm_col, cancel_col = delete_placeholder.columns(2)
                 if confirm_col.button("Confirm Delete", key=f"confirm_delete_{st.session_state.current_chat_id}", type="primary"):
                      chat_to_delete = st.session_state.current_chat_id
                      del st.session_state.chat_histories[chat_to_delete] # Delete the history
                      print(f"Deleted chat: ID={chat_to_delete}")
                      # Option 3: Trigger garbage collection after deletion
                      gc.collect()
                      print("Triggered garbage collection.")
                      st.session_state.current_chat_id = next(iter(st.session_state.chat_histories))
                      delete_placeholder.empty()
                      st.rerun()
                 if cancel_col.button("Cancel", key=f"cancel_delete_{st.session_state.current_chat_id}"):
                      delete_placeholder.empty()
    elif st.session_state.chat_histories:
         st.caption("Cannot delete the last conversation.")

    st.divider()
    st.info("Seek wisdom, ask questions about the divine knowledge within the Gita.", icon="💡")
    # Add note about memory setting
    if TRUNCATE_STORED_HISTORY:
        st.warning(f"Memory Saving Active: Chat history is limited to the last {MAX_STORED_MESSAGES} messages.", icon="⚠️")


# --- Main Chat Area Styling ---
# (Keep existing Main title/caption)
st.title("🕉️ Bhagavad Gita AI Assistant")
st.caption("Your companion for exploring the timeless wisdom of the Gita.")

if not st.session_state.current_chat_id or st.session_state.current_chat_id not in st.session_state.chat_histories:
    st.error("Please select or start a new conversation from the sidebar.")
    st.stop()

current_chat_data = st.session_state.chat_histories[st.session_state.current_chat_id]
current_messages = current_chat_data["messages"]

# (Keep existing current chat header/divider)
st.header(f"📜 {current_chat_data['name']}")
st.divider()

chat_container = st.container()
with chat_container:
    # (Keep existing message display loop)
    for msg in current_messages:
        role = msg.get("role")
        content = msg.get("content", "")
        avatar_icon = "👤" if role == "user" else "🕉️"
        with st.chat_message(role, avatar=avatar_icon):
            if role == "assistant":
                try:
                    parts = content.split("<split>", 1)
                    thinking = parts[0].strip() if len(parts) > 1 else ""
                    answer = parts[1].strip() if len(parts) > 1 else content.strip()
                    if thinking:
                        with st.expander("🧠 Thinking Process...", expanded=False):
                            st.info(thinking)
                    st.markdown(answer)
                except Exception as e: st.error(f"Error displaying: {e}"); st.markdown(content)
            else: st.markdown(content)

# --- Chat Input ---
prompt = st.chat_input(f"Ask about the Gita in '{current_chat_data['name']}'...")

if prompt:
    # Add user message
    user_message = {"role": "user", "content": prompt}
    current_messages.append(user_message)
    # Apply truncation AFTER adding message (if enabled)
    truncate_history_if_needed(current_messages)

    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Process and display assistant response
    with st.chat_message("assistant", avatar="🕉️"):
        message_placeholder = st.empty()
        thinking_placeholder = st.empty()
        full_response_content = ""

        with st.spinner("🕉️ Seeking divine wisdom..."):
            try:
                # Option 1: Pass only the TAIL of the history to the pipeline
                history_for_pipeline = current_messages[-MAX_HISTORY_FOR_PIPELINE:]
                print(f"Passing last {len(history_for_pipeline)} messages to pipeline.") # Debug print

                response = chat_pipeline(history_for_pipeline) # Use the slice here
                full_response_content = response

                # (Keep existing response parsing/display logic)
                parts = response.split("<split>", 1)
                thinking = parts[0].strip() if len(parts) > 1 else ""
                answer = parts[1].strip() if len(parts) > 1 else response.strip()
                if thinking:
                     with thinking_placeholder.expander("🧠 Thinking Process...", expanded=False): st.info(thinking)
                rendered_answer = ""
                words = answer.split()
                for i, word in enumerate(words):
                    rendered_answer += word + " "
                    message_placeholder.markdown(rendered_answer + ("▌" if i < len(words) - 1 else ""))
                    sleep(0.03)
                message_placeholder.markdown(rendered_answer.strip())

            except Exception as e:
                st.error(f"An error occurred: {e}")
                full_response_content = f"Error: Could not get response. Details: {e}"
                message_placeholder.markdown(full_response_content)

        # Add assistant response
        assistant_message = {"role": "assistant", "content": full_response_content}
        current_messages.append(assistant_message)
        # Apply truncation AFTER adding message (if enabled)
        truncate_history_if_needed(current_messages)

        # (Optional rerun)
        # st.rerun()