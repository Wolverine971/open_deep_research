from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic  # Add Anthropic import
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from langgraph.graph import StateGraph, START, END
from dataclasses import dataclass, field

# For interrupt/command support
from langgraph.types import interrupt, Command
from enum import Enum

class WorkflowState(str, Enum):
    """Represents the current state of the blog creation workflow."""
    RESEARCHING = "researching"
    OUTLINE_CREATED = "outline_created"
    OUTLINE_APPROVED = "outline_approved"
    BLOG_GENERATED = "blog_generated"

@dataclass
class PersonalityBlogState:
    # Input parameters
    person: str  # The person to analyze
    enneagram_type: str  # Their enneagram type
    
    # Workflow state
    status: WorkflowState = WorkflowState.RESEARCHING
    
    # Research fields
    search_queries: List[str] = field(default_factory=list)
    search_results: List[Dict] = field(default_factory=list)
    person_research: Optional[str] = None
    enneagram_research: Optional[str] = None
    
    # Content fields
    content_outline: Optional[str] = None
    final_blog: Optional[str] = None
    
    # User approval flags
    outline_approved: bool = False

def format_search_results(search_results: List[Dict]) -> str:
    """
    Cleanly format the search results to pass into the LLM.
    """
    if not search_results:
        return "No search results found."
    formatted = []
    for i, result in enumerate(search_results):
        # Truncate or handle missing fields safely.
        title = result.get('title', 'No title')
        url = result.get('url', 'No URL')
        content_preview = result.get('content', '')[:500] if result.get('content') else ''
        formatted.append(
            f"RESULT {i+1}:\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Content (excerpt): {content_preview}...\n"
        )
    return "\n\n".join(formatted)

# --------------------------------------------------------------
# Generate search queries for person and enneagram type
# --------------------------------------------------------------
class SearchQueries(BaseModel):
    person_queries: List[str] = Field(
        description="List of search queries to research the person"
    )
    enneagram_queries: List[str] = Field(
        description="List of search queries to research the enneagram type in relation to the person"
    )

def generate_search_queries(state: PersonalityBlogState) -> PersonalityBlogState:
    """
    Generates search queries to research both the person and their enneagram type.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    query_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are creating a blog about {person}'s personality, focusing on their Enneagram Type {enneagram_type}.\n\n"
         "Your first task is to generate effective search queries to gather comprehensive information about:\n"
         "1. {person}'s life, career, notable achievements, and personality traits\n"
         "2. How Enneagram Type {enneagram_type} might manifest in {person}'s behavior, choices, and life trajectory\n\n"
         "For the person, focus on queries that will uncover:\n"
         "- Formative experiences and background\n"
         "- Career trajectory and pivotal moments\n"
         "- Notable quotes that reveal their personality\n"
         "- How others describe their personality and working style\n"
         "- Recent projects or developments\n\n"
         "For the Enneagram connection, focus on queries that will help connect their type to their life:\n"
         "- General characteristics of Type {enneagram_type} that might apply to {person}\n"
         "- Specific behaviors or patterns that show {person} exemplifying Type {enneagram_type}\n"
         "- How their Type {enneagram_type} might influence their career choices and relationships\n"
        ),
        ("human", "Please generate 3-5 search queries for each category (person research and enneagram connection).")
    ])

    structured_llm = llm.with_structured_output(SearchQueries)
    chain = query_prompt | structured_llm
    
    queries = chain.invoke({
        "person": state.person,
        "enneagram_type": state.enneagram_type
    })
    
    # Store queries in state
    state.search_queries = queries.person_queries + queries.enneagram_queries
    
    return state

# --------------------------------------------------------------
# Perform web search and gather information
# --------------------------------------------------------------
def search_web(state: PersonalityBlogState) -> PersonalityBlogState:
    """
    Perform web searches using the generated queries and collect results.
    """
    tavily_search = TavilySearchResults(max_results=3)
    search_results = []
    
    for query in state.search_queries:
        results = tavily_search.invoke({"query": query})
        if results:
            search_results.extend(results)
    
    state.search_results = search_results
    return state

# --------------------------------------------------------------
# Process search results into organized research
# --------------------------------------------------------------
class ResearchOutput(BaseModel):
    person_research: str = Field(
        description="Organized research about the person's life, career, and personality"
    )
    enneagram_research: str = Field(
        description="Research on how the enneagram type manifests in the person's life"
    )

def organize_research(state: PersonalityBlogState) -> PersonalityBlogState:
    """
    Process search results into organized research about the person and their enneagram type.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    
    research_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are researching {person} and their Enneagram Type {enneagram_type} for a blog post.\n\n"
         "Search Results:\n{search_results}\n\n"
         "Your task is to organize this information into two clear research summaries:\n\n"
         "1. Person Research: Organize key facts, stories, quotes, and information about {person}'s life, career, personality traits, and recent developments. Focus on what they are known for and interesting stories that surround them. You are looking for what makes them unique and any patterns in their behavior or career.\n\n"
         "2. Enneagram Connection: Analyze how {person}'s behaviors, choices, and known personality traits align with Enneagram Type {enneagram_type}. Provide specific examples from their life that exemplify this type.\n\n"
         "For both sections:\n"
         "- Include specific, factual information with context\n"
         "- Note contradictions or nuances in the information\n"
         "- Include direct quotes where available\n"
         "- Organize information by themes rather than by source\n"
         "- Cite sources where appropriate (using the Result numbers)\n"
        ),
        ("human", "Please organize the research into these two clear categories based on the search results.")
    ])
    
    structured_llm = llm.with_structured_output(ResearchOutput)
    chain = research_prompt | structured_llm
    
    search_results_str = format_search_results(state.search_results)
    research = chain.invoke({
        "person": state.person,
        "enneagram_type": state.enneagram_type,
        "search_results": search_results_str
    })
    
    state.person_research = research.person_research
    state.enneagram_research = research.enneagram_research
    
    return state

# --------------------------------------------------------------
# Create content outline based on research
# --------------------------------------------------------------
class ContentOutlineOutput(BaseModel):
    content_outline: str = Field(
        description="Detailed content outline for the blog with all major sections and key points"
    )

def create_content_outline(state: PersonalityBlogState) -> PersonalityBlogState:
    """
    Create a detailed content outline based on the research.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    
    outline_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are creating a detailed outline for a blog post analyzing {person}'s personality as an Enneagram Type {enneagram_type}.\n\n"
         "Person Research:\n{person_research}\n\n"
         "Enneagram Type Connection:\n{enneagram_research}\n\n"
         "Your task is to create a comprehensive outline that will guide the writing of an engaging, informative blog post diving into {person}'s psyche. The outline should:\n\n"
         "1. Work backwards from {person}'s life and history and let that inform how we dive into their psychology and what we talk about in this article."
         "2. Have a clear, logical structure with H2 and H3 headings that are SEO optimized\n"
         "3. Begin with an introduction that hooks readers with something compelling about {person}\n"
         "4. Include a section explaining {person}'s Enneagram Type {enneagram_type} briefly\n"
         "5. Feature sections that connect {person}'s life experiences, career choices, and behaviors to their Enneagram type\n"
         "6. Include specific stories, quotes, and examples from the research\n"
         "7. Conclude with a brief wrap-up that leaves readers with something to think about\n\n"
         "Make the outline detailed enough that a writer could follow it to create a complete blog post."
        ),
        ("human", "Please create a detailed content outline for this blog post based on the research.")
    ])
    
    structured_llm = llm.with_structured_output(ContentOutlineOutput)
    chain = outline_prompt | structured_llm
    
    outline = chain.invoke({
        "person": state.person,
        "enneagram_type": state.enneagram_type,
        "person_research": state.person_research,
        "enneagram_research": state.enneagram_research
    })
    
    state.content_outline = outline.content_outline
    state.status = WorkflowState.OUTLINE_CREATED
    
    return state

# --------------------------------------------------------------
# Handle outline approval (conditional node)
# --------------------------------------------------------------
def check_outline_approval(state: PersonalityBlogState) -> str:
    """
    Check if the outline has been approved. This is a router node.
    """
    if state.outline_approved:
        state.status = WorkflowState.OUTLINE_APPROVED
        return "approved"
    else:
        return "not_approved"

# --------------------------------------------------------------
# Generate blog content based on approved outline
# --------------------------------------------------------------
def generate_blog_content(state: PersonalityBlogState) -> PersonalityBlogState:
    """
    Generate the full blog content based on the approved outline and research using Claude model.
    """
    # Using Claude model instead of GPT-4
    llm = ChatAnthropic(model="claude-3-7-sonnet-20250219", temperature=0.3)
    
    blog_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "[PERSON] = {person}\n"
         "[TYPE] = {enneagram_type}\n"
         "I write many articles analyzing celebrity personalities and I want the following article's analysis to be unique to the person.\n" 
         "Task: Write a blog post about [PERSON]'s Enneagram personality type ([TYPE]) that sparks curiosity about psychology, personality, and the Enneagram. The post should be evergreen content people want to share.\n"
         "Tone & Voice:\n"
         "* Analytical yet conversational and unbiased.\n"
         "* Speaking as if to one reader (second-person or direct address).\n"
         "* Humanize [PERSON] by exploring their inner world.\n"
         "* Keep language simple, clear, and a bit informal—avoid overly formal expressions.\n"
         "Style & Structure:\n"
         "1. Overall Goal\n"
         "   * Inspire readers to respect [PERSON] and feel curious about the Enneagram.\n"
         "   * Assume readers already know and admire [PERSON].\n"
         "   * Don't overexplain the Enneagram; keep it light but intriguing.\n"
         "   * Create a sense of FOMO for learning more about Enneagram.\n"
         "2. Introduction\n"
         "   * Begin with a quote that reveals something about [PERSON]'s personality (either something [PERSON] said or someone said about them).\n"
         "   * Write 2–3 sentences next that spark curiosity and hint at the deeper analysis to come.\n"
         "3. Use Markdown Headings and subheadings that are SEO optimized\n"
         "   * Write short paragraphs (preferably under 700 characters each).\n"
         "   * Include relevant SEO keywords like [PERSON]'s name and related topics.\n"
         "4. Follow this approved outline: \n{content_outline}\n\n"
         "5. Conclusion\n"
         "   * End with a concise wrap-up (no more than 150 words).\n"
         "      1. Transition and briefly summarize the main takeaway.\n"
         "      2. Pose a direct question to spark the reader's curiosity about [PERSON]'s personality or the Enneagram.\n"
         "Additional Notes:\n"
         "* Use a \"show, don't tell\" approach with examples, stories, quotes, and anecdotes from the research.\n"
         "* Vary sentence lengths—short and punchy, then more descriptive.\n"
         "* Make sure each paragraph's sentences rely on each other (no filler).\n"
         "* Don't use too many adjectives or adverbs. Avoid words like \"enigmatic,\" \"delve,\" \"unravel,\" \"enigma,\" \"quintessential,\" or \"unveiling.\"\n\n"
         "Use this research to inform your writing but dont let it limit your writing:\n\n"
         "Person Research:\n{person_research}\n\n"
         "Enneagram Connection:\n{enneagram_research}\n\n"
        ),
        ("human", "Please write the complete blog post now following the approved outline and using the research provided.")
    ])
    
    chain = blog_prompt | llm
    
    blog_response = chain.invoke({
        "person": state.person,
        "enneagram_type": state.enneagram_type,
        "content_outline": state.content_outline,
        "person_research": state.person_research,
        "enneagram_research": state.enneagram_research
    })
    
    state.final_blog = blog_response.content
    state.status = WorkflowState.BLOG_GENERATED
    
    return state

# --------------------------------------------------------------
# Update and approve outline (external input function)
# --------------------------------------------------------------
def update_and_approve_outline(state: PersonalityBlogState, updated_outline: str = None) -> PersonalityBlogState:
    """
    Update the outline if changes are provided, then mark it as approved.
    
    Args:
        state: The current state object
        updated_outline: Optional updated outline content. If None, the original outline is used.
    """
    if updated_outline is not None:
        state.content_outline = updated_outline
    
    state.outline_approved = True
    return state

# ----------------------------------------------------------------
# BUILD THE GRAPH
# ----------------------------------------------------------------
builder = StateGraph(PersonalityBlogState)

# Add nodes
builder.add_node("generate_search_queries", generate_search_queries)
builder.add_node("search_web", search_web)
builder.add_node("organize_research", organize_research)
builder.add_node("create_content_outline", create_content_outline)
builder.add_node("check_outline_approval", check_outline_approval)
builder.add_node("update_and_approve_outline", update_and_approve_outline)
builder.add_node("generate_blog_content", generate_blog_content)

# Define transitions
builder.add_edge(START, "generate_search_queries")
builder.add_edge("generate_search_queries", "search_web")
builder.add_edge("search_web", "organize_research")
builder.add_edge("organize_research", "create_content_outline")
    
    # Conditional path based on outline approval
builder.add_conditional_edges(
    "create_content_outline",
    check_outline_approval,
    {
        "approved": "generate_blog_content",
        "not_approved": END  # Stop here until approval
    }
)

builder.add_edge("update_and_approve_outline", "check_outline_approval")
builder.add_edge("generate_blog_content", END)
    
    # Compile the graph
graph = builder.compile()


# Example usage
if __name__ == "__main__":
    # Step 1: Start the blog creation process
    initial_state = PersonalityBlogState(
        person="Margot Robbie",
        enneagram_type=2
    )
    
    # Run the graph
    result = graph.invoke(initial_state)
    print("\nContent Outline (for approval):\n")
    print(result["content_outline"])
    
    # In a real application, you would wait for user approval here
    
    
    print("\nFinal Blog Content:\n")
    print(result["final_blog"])