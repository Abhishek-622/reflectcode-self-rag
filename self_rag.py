import json
import re
import os
from typing import Dict
from colorama import Fore, Style
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough , RunnableLambda , RunnableParallel
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv


# ==================== Setup ====================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",  
    temperature=0,
    groq_api_key=GROQ_API_KEY
)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


# ==================== Prompts ====================

generate_prompt = PromptTemplate.from_template(
    """Answer the query using ONLY the context. Be precise and code‑focused.
    Query: {query}
    Context: {context}
    Answer:"""
)

critique_prompt_dev = PromptTemplate.from_template(
    """Critique this explanation: did it use context accurately, hallucinate, or miss details?
    Rate relevance (0‑10) and suggest improvements (e.g. 'retrieve more on bugs').
    Query: {query}
    Context: {context}
    Generation: {generation}
    Output JSON: {{"relevance": int, "issues": list, "action": "refine"|"retrieve"|"good"}}
    Output your final answer ONLY as valid JSON. Do not include any extra text, prose, or code fences outside the JSON."""
)

critique_prompt_recruiter = PromptTemplate.from_template(
    """Critique for interview fit (role: {role}): strengths (e.g. clean code), weaknesses, score (1‑10).
    Query: {query}
    Context: {context}
    Generation: {generation}
    Output JSON: {{"strengths": list, "weaknesses": list, "score": int, "action": "refine"|"good"}}
    Output your final answer ONLY as valid JSON. Do not include any extra text, prose, or code fences outside the JSON."""
)

refine_prompt = PromptTemplate.from_template(
    """Refine the previous answer based on this critique.

Critique Summary: {critique}

Task:
1. Keep only the **essential corrected explanation or code**.
2. Remove any repetitive paragraphs or earlier drafts.
3. Present a final, polished response that fully answers the query.
4. Do not include JSON or critique notes in the final answer.

Original Query: {query}
Context (if relevant): {context}
Previous Answer: {generation}

### Refined Final Answer ###
"""
)

# ==================== Chains ====================

def get_generate_chain():
    base = RunnableParallel(
        query=RunnablePassthrough(),
        context=RunnableLambda(lambda x: "\n".join([doc.page_content for doc in retriever.invoke(x["query"])]))
    )
    return base | generate_prompt | llm | StrOutputParser()

def safe_json_load(raw):
    """Safely parse model output that *should* be JSON."""
    if not raw:
        return {"relevance": 5, "issues": ["empty_output"], "action": "good"}
    s = re.sub(r"^```(?:json)?|```$", "", str(raw).strip(), flags=re.MULTILINE)
    try:
        return json.loads(s)
    except Exception:
      
        return {"relevance": 5, "issues": ["bad_json"], "action": "good"}

def get_critique_chain(role="dev"):
    prompt = critique_prompt_dev if role == "dev" else critique_prompt_recruiter
    base = RunnableParallel(
    query=RunnablePassthrough(),
    context=RunnableLambda(
        lambda x: "\n".join(
            [doc.page_content for doc in retriever.invoke(x["query"])]
        )
    ),
    generation=RunnablePassthrough()
)

    # Add a role key dynamically
    add_role = RunnableLambda(lambda x: {**x, "role": role})

    return (
        base
        | add_role
        | prompt
        | llm
        | RunnableLambda(lambda x: safe_json_load(x.content if hasattr(x, "content") else x))
    )

def get_refine_chain():
    return refine_prompt | llm | StrOutputParser()


generate_chain = get_generate_chain()
critique_chain_dev = get_critique_chain("dev")
critique_chain_recruiter = get_critique_chain("recruiter")
refine_chain = get_refine_chain()


# ==================== Pipeline ====================

def self_rag_pipeline(query: str, mode: str = "dev", role: str = "") -> Dict:
    steps = []
    print(Fore.CYAN + "🔍  Retrieving relevant docs…" + Style.RESET_ALL)
    context_docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in context_docs])

    print(Fore.GREEN + "🤖  Generating initial answer…" + Style.RESET_ALL)
    generation = generate_chain.invoke({"query": query, "context": context})
    steps.append({"step": "Retrieval", "content": context[:500], "docs": len(context_docs)})
    steps.append({"step": "Initial Answer", "content": generation})

    critique_chain = critique_chain_dev if mode == "dev" else critique_chain_recruiter

    for i in range(2):
        print(Fore.MAGENTA + f"🧠  Critique iteration {i+1}…" + Style.RESET_ALL)
        critique = critique_chain.invoke({"query": query, "generation": generation})
        steps.append({
            "step": f"Critique {i+1}",
            "content": f"Relevance/Score: {critique.get('relevance', critique.get('score', 'N/A'))} "
                       f"| Issues: {', '.join(critique.get('issues', []))} | Action: {critique['action']}"
        })

        if critique["action"] == "good":
            break

        if critique["action"] == "retrieve":
            refined_query = query + " " + " ".join(critique.get("issues", [])[:2])
            context_docs = retriever.invoke(refined_query)
            context = "\n".join([doc.page_content for doc in context_docs])

        print(Fore.YELLOW + "🔁  Refining answer…" + Style.RESET_ALL)
        generation = refine_chain.invoke({
            "query": query,
            "context": context,
            "generation": generation,
            "critique": json.dumps(critique)
        })
        steps.append({"step": f"Refinement {i+1}", "content": generation})

    if mode == "recruiter":
        generation += (
            f"\n\n**Review Summary (for {role}):**\n"
            f"Score: {critique.get('score', 0)}/10\n"
            f"Strengths: {', '.join(critique.get('strengths', []))}\n"
            f"Weaknesses: {', '.join(critique.get('weaknesses', []))}"
        )
    print("✅  Reflection loop completed.")
    print("📘  Final Answer Preview:\n", generation[:500], "\n")

    return {"final_answer": generation, "steps": steps,
            "score": critique.get("score") if mode == "recruiter" else None}


# ==================== Test Run ====================

if __name__ == "__main__":
    mode = input("Choose mode (Dev/Recruiter): ").strip().lower() or "dev"
    role = ""
    if mode == "recruiter":
        role = input("Enter target role: ")
    query = input("Enter your query: ") or "Debug this overfitting code."
    result = self_rag_pipeline(query, mode=mode, role=role)