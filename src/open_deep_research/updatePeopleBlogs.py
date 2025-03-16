from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from langgraph.graph import StateGraph, START, END
from dataclasses import dataclass, field

# For interrupt/command support
from langgraph.types import interrupt, Command

@dataclass
class BlogUpdateState:
    blog_url: str
    current_content: Optional[str] = None
    
    # Analysis fields
    outline: Optional[str] = None
    person: Optional[str] = None
    enneagram_type: Optional[str] = None
    suggestions: Optional[str] = None
    
    # Workflow fields
    search_queries: List[str] = field(default_factory=list)
    search_results: List[Dict] = field(default_factory=list)
    proposed_outline: Optional[str] = None

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
        content_preview = result.get('content', '')[:500] if result.get('content') else ''
        formatted.append(
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Content (excerpt): {content_preview}...\n"
        )
    return "\n".join(formatted)

def fetch_blog_content(state: BlogUpdateState) -> BlogUpdateState:
    """
    Fetches the blog content from the provided URL.
    Returns an updated state with the blog content.
    """
    try:
        loader = WebBaseLoader(state.blog_url)
        documents = loader.load()
        if not documents or not documents[0].page_content:
            state.current_content = "No content could be loaded from the blog URL."
        else:
            state.current_content = documents[0].page_content
    except Exception as e:
        state.current_content = f"Error loading blog: {str(e)}"
    
    return state

# --------------------------------------------------------------
# Structured output for analyzing blog content
# --------------------------------------------------------------
class AnalyzeContentOutput(BaseModel):
    person: str = Field(
        description="The person the blog is about"
    )
    enneagram_type: str = Field(
        description="The enneagram type of the person"
    )
    outline: str = Field(
        description="An outline of the existing blog structure"
    )
    suggestions: str = Field(
        description="Suggestions for new or updated, research-based information to add to the blog."
    )

def analyze_content(state: BlogUpdateState) -> BlogUpdateState:
    """
    Analyzes the existing blog content, returning both a concise summary
    and suggestions for new/updated information.
    """
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    analyze_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "I have written several blogs analyzing different celebrities enneagram type. You are an expert blog editor an enneagram expert. Here is the blog content:\n\n"
         "<blog_content>\n{current_content}\n</blog_content>\n\n"
         "Please produce four outputs:\n"
         "1) The person the blog is about.\n"
         "2) What the Enneagram type of the person is.\n"
         "3) A outline that maps to each section and subsection of the blog\n"
         "4) Suggestions for new or updated *research-based* information to make it more comprehensive or up-to-date.\n"
         "   Avoid purely aesthetic or design suggestions (e.g., 'add images' or 'improve layout')."
        ),
        ("human", "Please provide the information requested now.")
    ])
    
    # Use structured output for reliable field retrieval
    structured_llm = llm.with_structured_output(AnalyzeContentOutput)
    chain = analyze_prompt | structured_llm
    
    # In case the content is empty or minimal, the LLM can handle it
    analysis = chain.invoke({"current_content": state.current_content})
    
    # Update state with analysis results
    state.person = analysis.person
    state.enneagram_type = analysis.enneagram_type
    state.outline = analysis.outline
    state.suggestions = analysis.suggestions
    
    return state

# --------------------------------------------------------------
# Generate search queries from summary + suggestions
# --------------------------------------------------------------
class SearchQueries(BaseModel):
    queries: List[str] = Field(description="List of search queries")

def generate_search_queries(state: BlogUpdateState) -> BlogUpdateState:
    """
    Uses the summary and suggestions to craft research-oriented queries.
    """
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    query_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         f"Your task is to generate search queries to add depth to a blog about '{state.person}'s Enneagram personality:\n\n"
         f"Blog Outline:\n{state.outline}\n\n"
         f"Suggestions for New/Updated Content:\n{state.suggestions}\n\n"
         "Generate 3 concise search queries that will help find new information, little stories, tidbits, "
         "recent news, or references to enhance the blog's content. Avoid cosmetic improvements or fluff."
        ),
        ("human", "Generate the queries in a JSON-compatible format.")
    ])

    structured_llm = llm.with_structured_output(SearchQueries)
    chain = query_prompt | structured_llm
    
    queries = chain.invoke({})  # No input needed as we're using f-strings above
    state.search_queries = queries.queries
    
    return state

def search_web(state: BlogUpdateState) -> BlogUpdateState:
    """
    Perform a web search using TavilySearchResults for each query.
    Gather results into a list. 
    """
    tavily_search = TavilySearchResults(max_results=5)
    search_results = []
    
    for query in state.search_queries:
        results = tavily_search.invoke({"query": query})
        if results:
            search_results.extend(results)
    
    state.search_results = search_results
    return state

def create_outline(state: BlogUpdateState) -> BlogUpdateState:
    """
    Create a proposed outline for the blog updates, using the formatted search results.
    """
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    outline_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         f"Your task is to create an improved outline for a blog on '{state.person}'s Enneagram type personality, given additional search results.\n\n"
         f"Blog Outline:\n{state.outline}\n\n"
         "Search Results:\n{search_results}\n\n"
         "Focus on incorporating relevant, factual information. Do not include purely aesthetic advice."
        ),
        ("human", "Please provide the outline now.")
    ])
    
    chain = outline_prompt | llm
    search_results_str = format_search_results(state.search_results)
    outline_response = chain.invoke({"search_results": search_results_str})
    
    state.proposed_outline = outline_response.content
    return state

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

# Define transitions
builder.add_edge(START, "fetch_blog_content")
builder.add_edge("fetch_blog_content", "analyze_content")
builder.add_edge("analyze_content", "generate_search_queries")
builder.add_edge("generate_search_queries", "search_web")
builder.add_edge("search_web", "create_outline")
builder.add_edge("create_outline", END)

# Compile the graph
graph = builder.compile()

if __name__ == "__main__":
    # Example usage
    initial_state = BlogUpdateState(blog_url="https://example.com/blog")
    result = graph.invoke(initial_state)
    
    print("\nBlog Analysis:\n")
    print(f"Person: {result.person}")
    print(f"Enneagram Type: {result.enneagram_type}")
    
    print("\nProposed Outline:\n")
    print(result.proposed_outline)

    # {"blog_url": "https://9takes.com/personality-analysis/Khloe-Kardashian"}
