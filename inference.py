# Inference.py - Memory Optimized Version (with acknowledgements of limitations)

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
# Use LangChain's FastEmbed wrapper for better integration
from langchain_community.embeddings import FastEmbedEmbeddings
from langsmith import Client as LangSmithClient

from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
# Import necessary prompt template components
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from langchain_core.runnables import RunnableSequence
from langchain_core.tracers.context import tracing_v2_enabled

# Load env
load_dotenv()

# --- Initialization ---
# Models are the primary memory consumers. They are loaded once at script start.
# Significant memory reduction often requires using smaller models, quantization,
# or offloading, which are outside the scope of simple code edits keeping functionality intact.
# --- Helper Functions ---

# Fetch prompt object from LangSmith
# Memory: Stores the prompt template object, which is typically small.
def fetch_langsmith_prompt(langsmith_client, prompt_name: str) -> ChatPromptTemplate | None:
    """Fetches a prompt object (likely ChatPromptTemplate) from LangSmith."""
    if not langsmith_client:
        print("[LangSmith] Client not available. Cannot fetch prompt.")
        return None
    try:
        # pull_prompt often returns a Prompt object which might be a ChatPromptTemplate
        prompt_object = langsmith_client.pull_prompt(prompt_name)

        if isinstance(prompt_object, ChatPromptTemplate):
            return prompt_object
        elif hasattr(prompt_object, 'messages'):
            # Attempt to convert if it has a messages structure but isn't ChatPromptTemplate
            print(f"[LangSmith] Fetched object for '{prompt_name}' is not a ChatPromptTemplate, but has messages. Attempting conversion.")
            return ChatPromptTemplate.from_messages(prompt_object.messages)
        else:
            print(f"[LangSmith] Fetched object for '{prompt_name}' is not a recognizable prompt type. Type: {type(prompt_object)}")
            return None
    except Exception as e:
        # Catching potential exceptions during pull or conversion
        print(f"[LangSmith] Prompt fetch failed for {prompt_name}: {e}")
        return None

# Convert message history to LangChain chat history
# Memory: Creates a new list of LangChain message objects from the input list.
# Size depends on conversation length. Limiting history length would save memory,
# but changes functionality.
def format_chat_history(messages):
    """Converts app message history to LangChain message objects."""
    lc_messages = []
    # Process messages efficiently, avoiding intermediate structures where possible.
    # This loop is already quite efficient.
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            # Extract the final answer part, discarding the thinking part for the history representation
            try:
                # Split only once to handle potential multiple <split> markers if they existed
                # (though typically it's one)
                _, answer = content.split("<split>", 1)
            except ValueError:
                 # If no <split>, assume the entire content is the answer
                 answer = content
            # Append the assistant's *answer* to history, not the raw output with <think>
            # This saves a tiny amount of memory for the history itself by not storing <think> tags,
            # and aligns with how chat history is usually represented to an LLM.
            lc_messages.append(AIMessage(content=answer.strip()))
    return lc_messages

# Qdrant search
# Memory: Stores the query embedding and the search results temporarily.
# The joined 'context' string can consume memory depending on k and doc size.
# Size of context is dictated by k and the content of stored chunks.
def search_qdrant(query: str, client: QdrantClient, embed_model, k: int = 5, collection_name: str = "bhagavad-gita-bge-small"):
    """Performs vector search in Qdrant and returns context strings."""
    try:
        # embed_query creates an embedding vector, temporary memory usage.
        query_embedding = embed_model.embed_query(query)
        # search_result contains search hits; temporary memory usage.
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=k
        )
        # Extract context strings. List comprehension is efficient.
        # Joining into a single string consumes memory proportional to total context size.
        contexts = [
            hit.payload.get("context", "")
            for hit in search_result
            if hit.payload is not None
        ]
        # Filter empty strings before joining to avoid unnecessary characters in the context string
        return "\n".join(filter(None, contexts))
    except Exception as e:
        print(f"Qdrant search failed: {e}")
        return "" # Return empty context on error

# Build the final chain with prompt template and LLM
# Memory: Creates the runnable sequence object. Does not duplicate model memory,
# but holds references. Prompt template object is small.
def build_rag_chain(base_prompt: ChatPromptTemplate, llm) -> RunnableSequence:
    """Constructs the final RAG chain."""
    # Ensure the necessary input variables are present in the prompt template
    required_vars = {"chat_history", "context", "query"}
    # Check input_variables directly on the prompt
    if not required_vars.issubset(base_prompt.input_variables):
         # Attempt intelligent injection if history is missing but other vars are okay
         if {"context", "query"}.issubset(base_prompt.input_variables) and "chat_history" not in base_prompt.input_variables:
             print("[Prompt Fix] 'chat_history' variable not found. Attempting to add MessagesPlaceholder.")
             messages = []
             placeholder_added = False
             # Iterate through existing messages to find a place for the placeholder
             for msg in base_prompt.messages:
                 messages.append(msg)
                 # Add after the first System message found, or at the beginning if none
                 if isinstance(msg, (SystemMessage, SystemMessagePromptTemplate)) and not placeholder_added:
                      messages.append(MessagesPlaceholder(variable_name="chat_history"))
                      placeholder_added = True

             # If no suitable place was found (e.g., no system message), add at the beginning
             if not placeholder_added:
                 messages.insert(0, MessagesPlaceholder(variable_name="chat_history"))

             final_prompt = ChatPromptTemplate.from_messages(messages)
             # Final check if now all required vars are present
             if not required_vars.issubset(final_prompt.input_variables):
                 raise ValueError(
                     f"Failed to inject 'chat_history' placeholder. Prompt template must include input variables: {required_vars}. "
                     f"Found after attempted fix: {final_prompt.input_variables}"
                 )
             base_prompt = final_prompt # Use the modified prompt
             print("[Prompt Fix] Successfully added 'chat_history' placeholder.")
         else:
            # If even context/query are missing, it's a fundamental prompt issue
            raise ValueError(
                f"Prompt template must include input variables: {required_vars}. "
                f"Found: {base_prompt.input_variables}"
            )

    # The chain links prompt and LLM. This object itself is not typically a memory bottleneck
    # compared to the underlying models.
    return base_prompt | llm

# --- Main Pipeline ---
def chat_pipeline(messages: list[dict], embed_model, langsmith_client, llm, qdrant_client):
    """
    Main RAG pipeline processing user input and chat history.
    Memory: Manages the flow of data (history, context, LLM output).
    Intermediate variables like chat_history, context, input_data, result,
    llm_output, thinking, final_answer consume memory temporarily per call.
    """
    if not messages:
        print("Error: No messages provided.")
        return "Error: No messages provided."

    # Get the latest user query. Efficiently access the last element.
    last_message = messages[-1]
    user_query = last_message.get("content", "").strip()
    if not user_query:
        print("Error: Latest message has no content.")
        return "Error: Latest message has no content."

    print(f"Processing query: '{user_query[:50]}...'") # Print truncated query

    # Format history (all messages except the last user query).
    # chat_history list memory depends on conversation length.
    chat_history = format_chat_history(messages[:-1])

    # 1. Vector Search
    # context string memory depends on k and document size.
    print("Searching context...")
    context = search_qdrant(user_query, qdrant_client, embed_model, k=5)
    if not context:
        print("[WARN] No context found from vector search.")

    # 2. Fetch Prompt Template from LangSmith
    # Prompt template object memory is small.
    print("Fetching prompt template...")
    prompt_name = "geeta-gpt" # Make this configurable if needed
    rag_prompt_template = fetch_langsmith_prompt(langsmith_client, prompt_name)

    # Use a fallback default prompt if fetch fails
    # Default prompt object memory is small.
    if rag_prompt_template is None:
        print(f"[WARN] Failed to fetch prompt '{prompt_name}'. Using default fallback.")
        rag_prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are GitaGPT, an AI assistant knowledgeable about the Bhagavad Gita. Answer the user's question based on the provided context and conversation history. If the context is insufficient, say you don't have enough information from the provided text. Format your response clearly. Include a brief thought process enclosed in <think>...</think> tags before the final answer."),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template(
                "Context from the Bhagavad Gita:\n---\n{context}\n---\n\nUser Question: {query}"
            )
        ])
        # Check fallback prompt variables to be safe
        if not {"chat_history", "context", "query"}.issubset(rag_prompt_template.input_variables):
             # This should ideally not happen with a hardcoded prompt, but good for robustness
             print("[CRITICAL] Fallback prompt template is invalid!")
             return "Error: Internal error with fallback prompt template."


    # 3. Build the RAG Chain
    # Chain object memory is small relative to models.
    print("Building RAG chain...")
    try:
        chain = build_rag_chain(rag_prompt_template, llm)
    except ValueError as e:
        print(f"Error building chain: {e}")
        return f"Error: Could not build the processing chain. Details: {e}"

    # 4. Invoke Chain (with LangSmith tracing if enabled)
    # input_data dictionary memory depends on history, context, query sizes.
    # result object memory holds the LLM's response string.
    print("Invoking LLM...")
    input_data = {
        "chat_history": chat_history, # Potentially large list of objects
        "context": context,         # Potentially large string
        "query": user_query         # String
    }

    result = None
    try:
        if langsmith_client:
             # tracing_v2_enabled context manager adds overhead for tracking,
             # but is necessary for the LangSmith functionality.
             with tracing_v2_enabled(project_name="BhagavadGita Chat", client=langsmith_client):
                 result = chain.invoke(input_data)
        else:
             result = chain.invoke(input_data)

    except Exception as e:
        print(f"LLM invocation failed: {e}")
        return f"Error during LLM call: {e}" # Return error to the caller

    if not result or not hasattr(result, 'content'):
          print("LLM invocation returned an unexpected result.")
          return "Error: Failed to get a valid response from the language model."

    # 5. Post-process response (separate thinking and answer)
    # llm_output string memory holds the full response.
    # thinking and final_answer strings hold parts of the response.
    # These consume memory proportional to the LLM output size.
    llm_output = result.content
    thinking = ""
    final_answer = llm_output # Default to full output

    # Try to extract thinking part robustly
    start_tag = "<think>"
    end_tag = "</think>"
    try:
        # Find start and end indices efficiently
        start_index = llm_output.find(start_tag)
        end_index = llm_output.find(end_tag)

        if 0 <= start_index < end_index:
            # Extract thinking part. Creates a new string.
            thinking = llm_output[start_index + len(start_tag):end_index].strip()

            # Construct final answer by combining parts before and after the tags.
            # Creates new strings.
            part_before = llm_output[:start_index].strip()
            part_after = llm_output[end_index + len(end_tag):].strip()
            final_answer = (part_before + (" " if part_before and part_after else "") + part_after).strip()

            if not final_answer and not thinking: # Handle edge case where output might be just whitespace
                final_answer = llm_output.strip() # Fallback to trimmed full output
            elif not final_answer and thinking: # Case where content was only inside think tags
                 final_answer = "..." # Or some other indicator

    except Exception as e:
        # If parsing fails, log and use the full output as the answer.
        print(f"Error parsing <think> tags: {e}. Using full output as answer.")
        final_answer = llm_output # Revert to full output on parsing error
        thinking = "" # Clear thinking if parsing failed

    print("Processing complete.")
    # Return combined format. Creates the final return string.
    # This string's size depends on the extracted thinking and final_answer parts.
    return f"{thinking}<split>{final_answer}"

# --- Example Usage (Optional: for testing) ---
if __name__ == "__main__":
    print("Running example chat pipeline...")
    print("Note: Actual memory usage will depend on model sizes and conversation length.")

    # Example message history (like what Streamlit might maintain)
    # Memory usage for this list grows with the number of messages.
    example_messages = [
        {"role": "user", "content": "Who is Arjuna?"},
        {"role": "assistant", "content": "<think>Arjuna is a central figure in the Mahabharata.</think><split>Arjuna is one of the Pandava princes, a great warrior, and the main recipient of Krishna's teachings in the Bhagavad Gita."},
        {"role": "user", "content": "What does Krishna teach him about duty?"}
    ]

    # Run the pipeline. Temporary memory is used within the function.
    response = chat_pipeline(example_messages)

    print("\n--- Response ---")
    # Parse the response for display. Creates temporary strings for thinking and answer.
    try:
        thinking, answer = response.split("<split>", 1)
        print(f"Thinking:\n{thinking}\n")
        print(f"Answer:\n{answer}")
    except ValueError:
        print(f"Raw Response (split failed):\n{response}") # Print raw if split fails
    except Exception as e:
        print(f"Error displaying response: {e}")
        print(f"Raw Response (display error):\n{response}")

    print("\nExample finished.")