import streamlit as st
from time import sleep
from dotenv import load_dotenv
import os
from inference import chat_pipeline # Assuming your inference logic is here
import uuid

# Load environment variables
load_dotenv()

# --- Page Configuration (Professional Look) ---
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
    # Try to name based on first user message later if desired
    new_chat_name = f"Conversation {st.session_state.chat_id_counter}"
    st.session_state.chat_histories[new_chat_id] = {"name": new_chat_name, "messages": []}
    if switch_to_new:
        st.session_state.current_chat_id = new_chat_id
        print(f"Created and switched to new chat: ID={new_chat_id}, Name={new_chat_name}")
        st.rerun() # Rerun needed immediately after switching
    else:
         print(f"Created new chat (no switch): ID={new_chat_id}, Name={new_chat_name}")
         # No rerun needed if not switching immediately

# Ensure at least one chat exists on first load or if state is broken
if not st.session_state.chat_histories:
    print("No chat histories found, creating initial chat.")
    st.session_state.chat_id_counter = 1
    initial_chat_id = generate_chat_id()
    st.session_state.chat_histories[initial_chat_id] = {"name": "Conversation 1", "messages": []}
    st.session_state.current_chat_id = initial_chat_id
elif st.session_state.current_chat_id not in st.session_state.chat_histories:
    print(f"Current chat ID '{st.session_state.current_chat_id}' invalid, switching.")
    st.session_state.current_chat_id = next(iter(st.session_state.chat_histories)) # Switch to first available


# --- Sidebar Styling and Functionality ---
with st.sidebar:
    # st.image("https://www.bhagavad-gita.us/wp-content/uploads/2019/05/lord-krishna-high-resolution-images.jpg", use_container_width=True) # Example image - replace with your own logo/image if desired
    st.sidebar.title("🕉️ Gita Chats")
    st.sidebar.caption("Manage your explorations")

    if st.button("➕ Start New Conversation", use_container_width=True, type="primary"):
        create_new_chat()

    st.divider() # Visual separator

    # Chat selection using Radio - more visually constrained but clear selection state
    chat_options = list(st.session_state.chat_histories.keys())
    chat_display_names = {
        chat_id: data["name"]
        for chat_id, data in st.session_state.chat_histories.items()
    }

    def format_chat_option(chat_id):
        return chat_display_names.get(chat_id, f"Chat {chat_id[:4]}...") # Fallback display

    try:
        current_chat_index = chat_options.index(st.session_state.current_chat_id)
    except ValueError:
        current_chat_index = 0
        if chat_options:
             st.session_state.current_chat_id = chat_options[0]
        else:
             st.error("Error: No chats available.")
             st.stop()

    # Use label_visibility="collapsed" if the "Select Chat" label is redundant
    selected_chat_id = st.radio(
        "Your Conversations:",
        options=chat_options,
        format_func=format_chat_option,
        index=current_chat_index,
        key=f"chat_selector_{st.session_state.current_chat_id}" # Helps reset widget state
    )

    if selected_chat_id != st.session_state.current_chat_id:
        st.session_state.current_chat_id = selected_chat_id
        print(f"Switched to chat: ID={selected_chat_id}")
        st.rerun()

    st.divider() # Visual separator

    # Delete button with confirmation
    if len(st.session_state.chat_histories) > 1:
        # Use columns for better layout of delete button + confirmation text
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
             delete_placeholder = st.empty() # To show confirmation prompt
        with col2:
             if st.button("🗑️", help="Delete current conversation", key=f"delete_btn_{st.session_state.current_chat_id}"):
                 delete_placeholder.warning(f"Delete '{chat_display_names[st.session_state.current_chat_id]}'? ")
                 # Add confirm/cancel buttons within the placeholder area
                 confirm_col, cancel_col = delete_placeholder.columns(2)
                 if confirm_col.button("Confirm Delete", key=f"confirm_delete_{st.session_state.current_chat_id}", type="primary"):
                      chat_to_delete = st.session_state.current_chat_id
                      del st.session_state.chat_histories[chat_to_delete]
                      print(f"Deleted chat: ID={chat_to_delete}")
                      st.session_state.current_chat_id = next(iter(st.session_state.chat_histories)) # Switch to first remaining
                      delete_placeholder.empty() # Clear confirmation on success
                      st.rerun()
                 if cancel_col.button("Cancel", key=f"cancel_delete_{st.session_state.current_chat_id}"):
                      delete_placeholder.empty() # Clear confirmation on cancel

    elif st.session_state.chat_histories:
         st.caption("Cannot delete the last conversation.")

    st.divider()
    st.info("Seek wisdom, ask questions about the divine knowledge within the Gita.", icon="💡")


# --- Main Chat Area Styling ---

# Main Application Title (Top of the page)
st.title("🕉️ Bhagavad Gita AI Assistant")
st.caption("Your companion for exploring the timeless wisdom of the Gita.")
# Maybe add a subtle divider st.divider() if needed visually

# Ensure a valid chat is selected before proceeding
if not st.session_state.current_chat_id or st.session_state.current_chat_id not in st.session_state.chat_histories:
    st.error("Please select or start a new conversation from the sidebar.")
    st.stop() # Stop execution if no valid chat is selected

# Get current chat data
current_chat_data = st.session_state.chat_histories[st.session_state.current_chat_id]
current_messages = current_chat_data["messages"]

# Display Current Conversation Name (less prominent than main title)
st.header(f"📜 {current_chat_data['name']}")
st.divider()

# Container for chat messages for better visual grouping (optional, can add borders/background with CSS)
chat_container = st.container() # You can set height using container(height=...) if needed

with chat_container:
    # Display previous messages with avatars
    for msg in current_messages:
        role = msg.get("role")
        content = msg.get("content", "")
        avatar_icon = "👤" if role == "user" else "🕉️" # Or use custom avatar URLs

        with st.chat_message(role, avatar=avatar_icon):
            if role == "assistant":
                try:
                    parts = content.split("<split>", 1)
                    thinking = parts[0].strip() if len(parts) > 1 else ""
                    answer = parts[1].strip() if len(parts) > 1 else content.strip()

                    if thinking:
                        with st.expander("🧠 Thinking Process...", expanded=False): # Keep collapsed initially
                            st.info(thinking) # Use st.info or st.code for thinking block
                    st.markdown(answer) # Display the main answer
                except Exception as e:
                    st.error(f"Error displaying assistant message: {e}")
                    st.markdown(content) # Fallback
            else: # User message
                st.markdown(content)

# --- Chat Input ---
# The chat input widget is usually placed at the bottom automatically
prompt = st.chat_input(f"Ask about the Gita in '{current_chat_data['name']}'...")

if prompt:
    # Add and display user message
    current_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Process and display assistant response
    with st.chat_message("assistant", avatar="🕉️"):
        message_placeholder = st.empty() # For streaming answer
        thinking_placeholder = st.empty() # To show thinking expander *during* generation
        full_response_content = ""

        with st.spinner("🕉️ Seeking divine wisdom..."):
            try:
                response = chat_pipeline(current_messages)
                full_response_content = response

                parts = response.split("<split>", 1)
                thinking = parts[0].strip() if len(parts) > 1 else ""
                answer = parts[1].strip() if len(parts) > 1 else response.strip()

                # Display thinking process *first* if available
                if thinking:
                     with thinking_placeholder.expander("🧠 Thinking Process...", expanded=False):
                         st.info(thinking) # Or st.code(thinking)

                # Stream the answer part
                rendered_answer = ""
                words = answer.split()
                for i, word in enumerate(words):
                    rendered_answer += word + " "
                    message_placeholder.markdown(rendered_answer + ("▌" if i < len(words) - 1 else "")) # Cursor effect
                    sleep(0.03) # Adjust speed for desired effect
                message_placeholder.markdown(rendered_answer.strip()) # Final answer

            except Exception as e:
                st.error(f"An error occurred: {e}")
                full_response_content = f"Error: Could not get response. Details: {e}"
                message_placeholder.markdown(full_response_content)

        # Append the full assistant response to the history
        current_messages.append({"role": "assistant", "content": full_response_content})

        # Rerun isn't strictly needed here as state update happens,
        # but sometimes helps ensure UI consistency *immediately* after complex updates.
        # Test without it first. st.rerun()