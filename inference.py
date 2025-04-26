# Inference.py

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
    SystemMessagePromptTemplate # Might be needed depending on fetched prompt type
)
from langchain_core.runnables import RunnableSequence
from langchain_core.tracers.context import tracing_v2_enabled

# Load env
load_dotenv()

# --- Initialization ---
try:
    # Initialize embeddings using LangChain's wrapper
    embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-large-en-v1.5", cache_dir="./cache")

    langsmith_client = LangSmithClient(api_key=os.getenv("LANGSMITH_API_KEY"))
    if not os.getenv("LANGSMITH_API_KEY"):
        print("[WARN] LANGSMITH_API_KEY not set. LangSmith tracing disabled.")
        langsmith_client = None # Disable client if key is missing

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable not set.")
    llm = ChatGroq(model_name="deepseek-r1-distill-llama-70b", api_key=groq_api_key) # Using a common Groq model, adjust if needed

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if not qdrant_url or not qdrant_api_key:
        raise ValueError("QDRANT_URL or QDRANT_API_KEY environment variable not set.")
    qdrant_client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        prefer_grpc=True
    )
except Exception as e:
    print(f"Initialization failed: {e}")
    exit(1) # Exit if core components fail to initialize

# --- Helper Functions ---

# Fetch prompt object from LangSmith
def fetch_langsmith_prompt(prompt_name: str) -> ChatPromptTemplate | None:
    """Fetches a prompt object (likely ChatPromptTemplate) from LangSmith."""
    if not langsmith_client:
        print("[LangSmith] Client not available. Cannot fetch prompt.")
        return None
    try:
        # pull_prompt often returns a Prompt object which might be a ChatPromptTemplate
        # or convertible to one. Adjust if the return type is different.
        prompt_object = langsmith_client.pull_prompt(prompt_name)
        # You might need validation here to ensure it's the type you expect
        if isinstance(prompt_object, ChatPromptTemplate):
             return prompt_object
        # If it's a different Prompt type, you might need to convert it
        # Example: Convert BasePromptValue to messages if needed
        # elif hasattr(prompt_object, 'messages'):
        #     return ChatPromptTemplate.from_messages(prompt_object.messages)
        else:
             print(f"[LangSmith] Fetched object for '{prompt_name}' is not a ChatPromptTemplate. Type: {type(prompt_object)}")
             # Attempt to create from messages if available
             if hasattr(prompt_object, 'messages'):
                return ChatPromptTemplate.from_messages(prompt_object.messages)
             return None # Or handle other types as needed
    except Exception as e:
        print(f"[LangSmith] Prompt fetch failed for {prompt_name}: {e}")
        return None

# Convert Streamlit message history to LangChain chat history
def format_chat_history(messages):
    """Converts app message history to LangChain message objects."""
    lc_messages = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            try:
                # Handle potential <split> format in stored assistant messages
                thinking, answer = content.split("<split>", 1) # Split only once
            except ValueError:
                answer = content # Use full content if no split marker
            lc_messages.append(AIMessage(content=answer.strip()))
    return lc_messages

# Qdrant search
def search_qdrant(query: str, client: QdrantClient, embed_model, k: int = 5, collection_name: str = "bhagavad-gita"):
    """Performs vector search in Qdrant and returns context strings."""
    try:
        query_embedding = embed_model.embed_query(query) # Use embed_query for single queries
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=k
            # You might want to add `query_filter` or `search_params` here if needed
        )
        # Extract context from payload, ensuring payload and context key exist
        contexts = [
            hit.payload.get("context", "") # Use .get for safety
            for hit in search_result
            if hit.payload is not None # Ensure payload exists
        ]
        return "\n".join(filter(None, contexts)) # Join non-empty contexts
    except Exception as e:
        print(f"Qdrant search failed: {e}")
        return "" # Return empty context on error

# Build the final chain with prompt template and LLM
def build_rag_chain(base_prompt: ChatPromptTemplate, llm) -> RunnableSequence:
    """Constructs the final RAG chain."""
    # Ensure the necessary input variables are present in the prompt template
    required_vars = {"chat_history", "context", "query"}
    if not required_vars.issubset(base_prompt.input_variables):
        raise ValueError(
            f"Prompt template must include input variables: {required_vars}. "
            f"Found: {base_prompt.input_variables}"
        )

    # Make sure MessagesPlaceholder is in the prompt
    has_history_placeholder = any(
        isinstance(msg, MessagesPlaceholder) and msg.variable_name == "chat_history"
        for msg in base_prompt.messages
    )

    if not has_history_placeholder:
        # Inject placeholder intelligently (e.g., after system message if present)
        messages = []
        placeholder_added = False
        for msg in base_prompt.messages:
            messages.append(msg)
            if isinstance(msg, (SystemMessage, SystemMessagePromptTemplate)) and not placeholder_added:
                messages.append(MessagesPlaceholder(variable_name="chat_history"))
                placeholder_added = True

        # If no system message, add placeholder near the beginning (adjust logic as needed)
        if not placeholder_added:
            messages.insert(0, MessagesPlaceholder(variable_name="chat_history"))
            placeholder_added = True

        final_prompt = ChatPromptTemplate.from_messages(messages)
        # Re-check variables after modification
        if not required_vars.issubset(final_prompt.input_variables):
             raise ValueError(
                f"Modified prompt template must include input variables: {required_vars}. "
                f"Found: {final_prompt.input_variables}"
            )
        print("[Prompt Fix] Added 'chat_history' placeholder to the fetched prompt.")
        base_prompt = final_prompt # Use the modified prompt

    return base_prompt | llm

# --- Main Pipeline ---
def chat_pipeline(messages: list[dict]):
    """Main RAG pipeline processing user input and chat history."""
    if not messages:
        return "Error: No messages provided."

    user_query = messages[-1].get("content", "").strip()
    if not user_query:
        return "Error: Latest message has no content."

    print(f"Processing query: {user_query}")

    # Format history (all messages except the last one)
    chat_history = format_chat_history(messages[:-1])

    # 1. Vector Search
    print("Searching context...")
    context = search_qdrant(user_query, qdrant_client, embed_model, k=5)
    if not context:
        print("[WARN] No context found from vector search.")

    # 2. Fetch Prompt Template from LangSmith
    print("Fetching prompt template...")
    prompt_name = "geeta-gpt" # Make this configurable if needed
    rag_prompt_template = fetch_langsmith_prompt(prompt_name)
    # print(rag_prompt_template)
    # Use a fallback default prompt if fetch fails
    if rag_prompt_template is None:
        print(f"[WARN] Failed to fetch prompt '{prompt_name}'. Using default fallback.")
        rag_prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are GitaGPT, an AI assistant knowledgeable about the Bhagavad Gita. Answer the user's question based on the provided context and conversation history. If the context is insufficient, say you don't have enough information from the provided text. Format your response clearly. Include a brief thought process enclosed in <think>...</think> tags before the final answer."),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template(
                "Context from the Bhagavad Gita:\n---\n{context}\n---\n\nUser Question: {query}"
            )
        ])

    # 3. Build the RAG Chain
    print("Building RAG chain...")
    try:
        chain = build_rag_chain(rag_prompt_template, llm)
    except ValueError as e:
        print(f"Error building chain: {e}")
        # Provide a more informative error message back to the caller/UI
        return f"Error: Could not build the processing chain. Problem with prompt variables. Details: {e}"


    # 4. Invoke Chain (with LangSmith tracing if enabled)
    print("Invoking LLM...")
    input_data = {
        "chat_history": chat_history,
        "context": context,
        "query": user_query
    }

    result = None
    if langsmith_client:
        try:
            with tracing_v2_enabled(project_name="BhagavadGita Chat", client=langsmith_client):
                result = chain.invoke(input_data)
        except Exception as e:
            print(f"LLM invocation with tracing failed: {e}")
            # Fallback to invoking without tracing? Or just return error?
            return f"Error during LLM call: {e}"
    else:
        try:
            result = chain.invoke(input_data)
        except Exception as e:
            print(f"LLM invocation failed: {e}")
            return f"Error during LLM call: {e}"

    if not result or not hasattr(result, 'content'):
         print("LLM invocation returned an unexpected result.")
         return "Error: Failed to get a valid response from the language model."

    # 5. Post-process response (separate thinking and answer)
    llm_output = result.content
    thinking = ""
    final_answer = llm_output # Default to full output

    # Try to extract thinking part robustly
    if "<think>" in llm_output and "</think>" in llm_output:
        try:
            start_tag = "<think>"
            end_tag = "</think>"
            start_index = llm_output.find(start_tag)
            end_index = llm_output.find(end_tag)

            if 0 <= start_index < end_index:
                thinking = llm_output[start_index + len(start_tag):end_index].strip()
                # Combine parts before <think> and after </think> as the final answer
                part_before = llm_output[:start_index].strip()
                part_after = llm_output[end_index + len(end_tag):].strip()
                final_answer = (part_before + " " + part_after).strip()
                if not final_answer: # Handle case where entire output was inside <think> tags
                    final_answer = "..." # Or some placeholder
        except Exception as e:
            print(f"Error parsing <think> tags: {e}. Using full output as answer.")
            final_answer = llm_output # Revert to full output on parsing error

    print("Processing complete.")
    # Return combined format for the application layer
    return f"{thinking}<split>{final_answer}"

# --- Example Usage (Optional: for testing) ---
if __name__ == "__main__":
    print("Running example chat pipeline...")

    # Example message history (like what Streamlit might maintain)
    example_messages = [
        {"role": "user", "content": "Who is Arjuna?"},
        {"role": "assistant", "content": "<think>Arjuna is a central figure in the Mahabharata.</think><split>Arjuna is one of the Pandava princes, a great warrior, and the main recipient of Krishna's teachings in the Bhagavad Gita."},
        {"role": "user", "content": "What does Krishna teach him about duty?"}
    ]

    response = chat_pipeline(example_messages)

    print("\n--- Response ---")
    try:
        thinking, answer = response.split("<split>", 1)
        print(f"Thinking:\n{thinking}\n")
        print(f"Answer:\n{answer}")
    except ValueError:
        print(f"Raw Response:\n{response}") # Print raw if split fails
    except Exception as e:
        print(f"Error displaying response: {e}")
        print(f"Raw Response:\n{response}")