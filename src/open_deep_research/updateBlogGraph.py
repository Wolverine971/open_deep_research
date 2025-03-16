from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from langgraph.graph import StateGraph, START, END
from dataclasses import dataclass, field

# NEW IMPORTS FOR INTERRUPT/COMMAND
from langgraph.types import interrupt, Command
from typing import Literal

@dataclass
class BlogUpdateState:
    blog_url: str
    current_content: Optional[str] = None
    
    # We'll store a short summary and improvement suggestions
    summary: Optional[str] = None
    suggestions: Optional[str] = None
    
    # The rest of the fields used along the workflow
    search_queries: Optional[List[str]] = field(default_factory=list)
    search_results: Optional[List[Dict]] = field(default_factory=list)
    proposed_outline: Optional[str] = None
    approved_outline: Optional[str] = None
    new_content: Optional[str] = None
    updated_blog: Optional[str] = None

def format_search_results(search_results: List[Dict]) -> str:
    """
    Cleanly format the search results to pass into the LLM for outline creation.
    """
    if not search_results:
        return "No search results found."
    formatted = []
    for result in search_results:
        # Truncate or handle missing fields safely.
        title = result.get('title', 'No title')
        url = result.get('url', 'No URL')
        content_preview = result.get('content', '')[:500]
        formatted.append(
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Content (excerpt): {content_preview}...\n"
        )
    return "\n".join(formatted)

def fetch_blog_content(state: BlogUpdateState, config) -> Dict:
    """
    Fetches the blog content from the provided URL.
    If no content is found or an error occurs, handle gracefully.
    """
    loader = WebBaseLoader(state.blog_url)
    documents = loader.load()
    if not documents or not documents[0].page_content:
        return {"current_content": "No content could be loaded from the blog URL."}
    
    current_content = documents[0].page_content
    return {"current_content": current_content}

# --------------------------------------------------------------
# Structured output for analyzing blog content
# --------------------------------------------------------------
class AnalyzeContentOutput(BaseModel):
    summary: str = Field(
        description="A concise summary of the existing blog content."
    )
    suggestions: str = Field(
        description="Suggestions for new or updated, research-based information to add to the blog."
    )

def analyze_content(state: BlogUpdateState, config) -> Dict:
    """
    Analyzes the existing blog content, returning both a concise summary
    and suggestions for new/updated information (research-based, not cosmetic).
    """
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    analyze_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an expert blog editor. You have the following blog content:\n\n"
         "<blog_content>\n{current_content}\n</blog_content>\n\n"
         "Please produce two outputs:\n"
         "1) A concise summary of what the blog discusses.\n"
         "2) Suggestions for new or updated *research-based* information to make it more comprehensive or up-to-date.\n"
         "   Avoid purely aesthetic or design suggestions (e.g., 'add images' or 'improve layout')."
        ),
        ("human", "Please provide the summary and suggestions now.")
    ])
    
    # Use structured output so we can reliably retrieve fields:
    structured_llm = llm.with_structured_output(AnalyzeContentOutput)
    chain = analyze_prompt | structured_llm
    
    # In case the content is empty or minimal, the LLM can handle it.
    analysis = chain.invoke({"current_content": state.current_content})
    
    return {
        "summary": analysis.summary,
        "suggestions": analysis.suggestions
    }

# --------------------------------------------------------------
# Generate search queries from summary + suggestions
# --------------------------------------------------------------
class SearchQueries(BaseModel):
    queries: List[str] = Field(description="List of search queries")

def generate_search_queries(state: BlogUpdateState, config) -> Dict:
    """
    Uses the summary and suggestions to craft research-oriented queries.
    """
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    query_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Your task is to generate search queries to add depth to a blog:\n\n"
         "Blog Summary:\n{summary}\n\n"
         "Suggestions for New/Updated Content:\n{suggestions}\n\n"
         "Generate 3 concise, research-oriented search queries that will help find new facts, "
         "data, or references to enhance the blog's content. Avoid cosmetic improvements or fluff."
        ),
        ("human", "Generate the queries in a JSON-compatible format.")
    ])

    structured_llm = llm.with_structured_output(SearchQueries)
    chain = query_prompt | structured_llm
    
    queries = chain.invoke({
        "summary": state.summary,
        "suggestions": state.suggestions
    })
    return {"search_queries": queries.queries}

def search_web(state: BlogUpdateState, config) -> Dict:
    """
    Perform a web search using TavilySearchResults for each query.
    Gather results into a list. 
    """
    tavily_search = TavilySearchResults(max_results=5)
    queries = state.search_queries or []
    search_results = []
    
    for query in queries:
        results = tavily_search.invoke({"query": query})
        if results:
            search_results.extend(results)
    
    return {"search_results": search_results}

def create_outline(state: BlogUpdateState, config) -> Dict:
    """
    Create a proposed outline for the blog updates, using the formatted search results.
    """
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    outline_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Your task is to create an improved outline for a blog given additional search results.\n\n"
         "Blog Summary:\n'{{state.summary}}'\n\n"
         "Search Results:\n{search_results}\n\n"
         "Focus on incorporating relevant, factual information. Do not include purely aesthetic advice."
        ),
        ("human", "Please provide the outline now.")
    ])
    
    chain = outline_prompt | llm
    search_results_str = format_search_results(state.search_results)
    outline_response = chain.invoke({"search_results": search_results_str})
    
    return {"proposed_outline": outline_response.content}

def human_feedback(state: BlogUpdateState, config) -> Command[Literal["create_outline","generate_new_content"]]:
    """
    Prompt the user to approve the proposed outline or provide revisions.
    If 'approve', proceed to generate_new_content.
    Otherwise, store the revised outline and go back to create_outline.
    """
    proposed_outline = state.proposed_outline or "No outline was generated."

    feedback = interrupt(
        "Here is the proposed outline for additional content:\n\n"
        f"{proposed_outline}\n\n"
        "If you approve this outline, type 'approve'. Otherwise, type any revisions you want to make:\n"
    )

    if isinstance(feedback, str) and feedback.strip().lower() == "approve":
        # The user approves the outline, proceed to next node
        return Command(
            goto="generate_new_content",
            update={"approved_outline": proposed_outline}
        )
    else:
        # The user has provided new edits or feedback; go back to create_outline
        revised_outline = feedback if isinstance(feedback, str) else ""
        return Command(
            goto="create_outline",
            update={"proposed_outline": revised_outline}
        )

def generate_new_content(state: BlogUpdateState, config) -> Dict:
    """
    Generate additional blog content in markdown format based on the approved outline.
    """
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    content_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Your task is to create a blog based off of this outline:\n\n"
         "{approved_outline}\n\n"
         "Please write in markdown format, and focus on the factual details discovered."
        ),
        ("human", "Generate the updated blog content now.")
    ])
    
    chain = content_prompt | llm
    new_content_response = chain.invoke({"approved_outline": state.approved_outline})
    
    return {"new_content": new_content_response.content}

def update_blog(state: BlogUpdateState, config) -> Dict:
    """
    Append the newly generated content to the existing blog content.
    """
    current_content = state.current_content or "No original blog content found."
    new_content = state.new_content or "No new content was generated."
    
    updated_blog = (
        f"{current_content}\n\n"
        "## Updates\n\n"
        f"{new_content}"
    )
    return {"updated_blog": updated_blog}


# ----------------------------------------------------------------
# BUILD THE GRAPH
# ----------------------------------------------------------------
builder = StateGraph(BlogUpdateState)

# Add nodes
builder.add_node("fetch_blog_content", fetch_blog_content)
builder.add_node("analyze_content", analyze_content)
builder.add_node("generate_search_queries", generate_search_queries)
builder.add_node("search_web", search_web)
builder.add_node("create_outline", create_outline)
builder.add_node("human_feedback", human_feedback)
builder.add_node("generate_new_content", generate_new_content)
builder.add_node("update_blog", update_blog)

# Define transitions
builder.add_edge(START, "fetch_blog_content")
builder.add_edge("fetch_blog_content", "analyze_content")
builder.add_edge("analyze_content", "generate_search_queries")
builder.add_edge("generate_search_queries", "search_web")
builder.add_edge("search_web", "create_outline")

# The feedback node can direct either backward or forward
builder.add_edge("create_outline", "human_feedback")
# The `human_feedback` node uses Command to decide next step
builder.add_edge("generate_new_content", "update_blog")
builder.add_edge("update_blog", END)

# Compile the graph
graph = builder.compile()

if __name__ == "__main__":
    # Example usage
    initial_state = BlogUpdateState(blog_url="https://example.com/blog")
    result = graph.invoke(initial_state)
    
    print("\nUpdated Blog Content:\n")
    print(result["updated_blog"])
