"""Enhanced Blog Update System for 9takes Personality Blogs

This module provides a complete workflow for fetching, analyzing, researching,
and updating personality blogs with new content while maintaining style consistency.
"""

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Literal, Optional

# from langchain_anthropic import ChatAnthropic  # Commented out - using OpenAI instead

# Core imports
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver

# LangGraph imports
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

# Supabase client
try:
    from supabase import Client, create_client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("Warning: Supabase not installed. Install with: pip install supabase")

# MCP Puppeteer support (optional)
try:
    import mcp  # noqa: F401
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


class WorkflowStatus(str, Enum):
    """Tracks the current status of the blog update workflow"""
    INITIALIZING = "initializing"
    FETCHING = "fetching"
    ANALYZING = "analyzing"
    RESEARCHING = "researching"
    OUTLINING = "outlining"
    AWAITING_APPROVAL = "awaiting_approval"
    REWRITING = "rewriting"
    SAVING = "saving"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class BlogUpdateState:
    """Complete state management for blog update workflow"""
    
    # Input
    blog_identifier: str  # Can be URL or Supabase ID
    
    # Blog Content
    current_content: Optional[str] = None
    current_markdown: Optional[str] = None
    blog_metadata: Optional[Dict] = field(default_factory=dict)
    
    # Analysis
    person: Optional[str] = None
    enneagram_type: Optional[str] = None
    current_outline: Optional[str] = None
    suggestions: Optional[str] = None
    
    # Research
    search_queries: List[str] = field(default_factory=list)
    search_results: List[Dict] = field(default_factory=list)
    research_summary: Optional[str] = None
    
    # Generation
    proposed_outline: Optional[str] = None
    approved_outline: Optional[str] = None
    rewritten_content: Optional[str] = None
    rewritten_markdown: Optional[str] = None
    
    # Workflow
    workflow_status: WorkflowStatus = WorkflowStatus.INITIALIZING
    error_message: Optional[str] = None
    approval_history: List[Dict] = field(default_factory=list)
    
    # Configuration
    search_method: str = "tavily"  # tavily, perplexity, mcp_puppeteer
    require_approval: bool = True
    save_to_supabase: bool = True
    export_markdown: bool = True


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_supabase_client() -> Optional[Client]:
    """Initialize Supabase client if credentials are available"""
    if not SUPABASE_AVAILABLE:
        return None
    
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    
    if not url or not key:
        print("Warning: Supabase credentials not found in environment")
        return None
    
    return create_client(url, key)


def format_search_results(search_results: List[Dict]) -> str:
    """Format search results for LLM consumption"""
    if not search_results:
        return "No search results found."
    
    formatted = []
    for i, result in enumerate(search_results, 1):
        title = result.get('title', 'No title')
        url = result.get('url', 'No URL')
        content = result.get('content', '')[:500] if result.get('content') else ''
        score = result.get('score', 0)
        
        formatted.append(
            f"**Result {i}** (Relevance: {score:.2f})\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Content: {content}...\n"
        )
    
    return "\n".join(formatted)


async def search_with_mcp_puppeteer(queries: List[str]) -> List[Dict]:
    """Use MCP Puppeteer for advanced web scraping"""
    if not MCP_AVAILABLE:
        raise ValueError("MCP not available. Install with: pip install mcp")
    
    # Implementation would connect to MCP Puppeteer server
    # This is a placeholder for the actual implementation
    results = []
    for query in queries:
        # In practice, this would use MCP to control a browser
        results.append({
            "title": f"MCP Result for: {query}",
            "url": "https://example.com",
            "content": "Placeholder content from MCP Puppeteer",
            "score": 0.9
        })
    return results


# ============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUTS
# ============================================================================

class BlogAnalysis(BaseModel):
    """Structured output for blog content analysis"""
    person: str = Field(description="The person the blog is about")
    enneagram_type: str = Field(description="The Enneagram type (e.g., 'Type 2', '2w3')")
    current_outline: str = Field(description="Hierarchical outline of current blog structure")
    content_summary: str = Field(description="Brief summary of existing content")
    suggestions: str = Field(description="Specific suggestions for improvements")
    missing_topics: List[str] = Field(description="Topics that should be covered but aren't")


class SearchQueries(BaseModel):
    """Structured search queries"""
    queries: List[str] = Field(description="List of search queries")


class OutlineProposal(BaseModel):
    """Structured outline with additions marked"""
    outline: str = Field(description="Complete outline with new sections marked")
    additions_summary: str = Field(description="Summary of what's being added")
    word_count_estimate: int = Field(description="Estimated word count of new content")


# ============================================================================
# WORKFLOW NODES
# ============================================================================

async def fetch_blog_content(state: BlogUpdateState) -> BlogUpdateState:
    """Fetch blog content from URL or Supabase
    
    Supports:
    - Direct URL fetching (https://9takes.com/...)
    - Supabase ID lookup
    - Supabase slug lookup
    """
    state.workflow_status = WorkflowStatus.FETCHING
    
    identifier = state.blog_identifier
    
    # Check if it's a URL
    if identifier.startswith("http"):
        try:
            # Use requests to fetch with timeout
            import requests
            response = requests.get(identifier, timeout=10)
            if response.status_code == 200:
                state.current_content = response.text
                state.blog_metadata = {"source": "web", "url": identifier}
            else:
                state.error_message = f"Failed to fetch URL: HTTP {response.status_code}"
                state.workflow_status = WorkflowStatus.ERROR
        except requests.Timeout:
            state.error_message = "Timeout while fetching URL"
            state.workflow_status = WorkflowStatus.ERROR
        except Exception as e:
            state.error_message = f"Error fetching from URL: {str(e)}"
            state.workflow_status = WorkflowStatus.ERROR
    
    # Try Supabase
    elif SUPABASE_AVAILABLE:
        supabase = get_supabase_client()
        if supabase:
            try:
                # Try as ID first
                response = supabase.table("blogs").select("*").eq("id", identifier).execute()
                
                # If not found, try as slug
                if not response.data:
                    response = supabase.table("blogs").select("*").eq("slug", identifier).execute()
                
                if response.data:
                    blog = response.data[0]
                    state.current_content = blog.get("content", "")
                    state.current_markdown = blog.get("markdown", "")
                    state.blog_metadata = blog
                else:
                    state.error_message = f"Blog not found in Supabase: {identifier}"
                    state.workflow_status = WorkflowStatus.ERROR
            except Exception as e:
                state.error_message = f"Supabase error: {str(e)}"
                state.workflow_status = WorkflowStatus.ERROR
        else:
            state.error_message = "Supabase client not configured"
            state.workflow_status = WorkflowStatus.ERROR
    else:
        state.error_message = "Invalid identifier and Supabase not available"
        state.workflow_status = WorkflowStatus.ERROR
    
    return state


def analyze_content(state: BlogUpdateState) -> BlogUpdateState:
    """Analyze existing blog content using GPT-4"""
    state.workflow_status = WorkflowStatus.ANALYZING
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    analyze_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an expert blog editor and Enneagram specialist analyzing personality blogs.\n"
         "Analyze the following blog content and provide structured insights.\n\n"
         "Blog Content:\n{content}\n\n"
         "Focus on:\n"
         "1. Identifying the person and their Enneagram type\n"
         "2. Mapping the current structure\n"
         "3. Finding gaps in coverage\n"
         "4. Suggesting research-based improvements"
        ),
        ("human", "Analyze this blog and provide structured output.")
    ])
    
    structured_llm = llm.with_structured_output(BlogAnalysis)
    chain = analyze_prompt | structured_llm
    
    analysis = chain.invoke({"content": state.current_content})
    
    # Update state with analysis
    state.person = analysis.person if analysis.person and analysis.person != "N/A" else "Unknown"
    state.enneagram_type = analysis.enneagram_type if analysis.enneagram_type else "Unknown"
    state.current_outline = analysis.current_outline
    state.suggestions = analysis.suggestions
    
    return state


def generate_search_queries(state: BlogUpdateState) -> BlogUpdateState:
    """Generate targeted search queries for research"""
    state.workflow_status = WorkflowStatus.RESEARCHING
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    query_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Generate search queries to find new information about {person} "
         "(Enneagram {enneagram_type}) for a personality blog.\n\n"
         "Current gaps:\n{suggestions}\n\n"
         "Create 5-7 specific queries targeting:\n"
         "- Recent news or developments\n"
         "- Specific stories or anecdotes\n"
         "- Quotes or interviews\n"
         "- Professional milestones\n"
         "- Personality insights"
        ),
        ("human", "Generate search queries.")
    ])
    
    structured_llm = llm.with_structured_output(SearchQueries)
    chain = query_prompt | structured_llm
    
    queries = chain.invoke({
        "person": state.person,
        "enneagram_type": state.enneagram_type,
        "suggestions": state.suggestions
    })
    state.search_queries = queries.queries
    
    return state


async def search_web(state: BlogUpdateState) -> BlogUpdateState:
    """Execute web searches using configured method"""
    search_results = []
    
    if state.search_method == "tavily":
        tavily_search = TavilySearch(max_results=5)
        for query in state.search_queries:
            response = tavily_search.invoke({"query": query})
            if response and 'results' in response:
                search_results.extend(response['results'])
    
    elif state.search_method == "perplexity":
        # Perplexity implementation would go here
        pass
    
    elif state.search_method == "mcp_puppeteer":
        search_results = await search_with_mcp_puppeteer(state.search_queries)
    
    state.search_results = search_results
    state.research_summary = format_search_results(search_results)
    
    return state


def create_outline(state: BlogUpdateState) -> BlogUpdateState:
    """Create an enhanced outline incorporating new research"""
    state.workflow_status = WorkflowStatus.OUTLINING
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    
    outline_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Create an enhanced outline for the {person} Enneagram blog.\n\n"
         "Current Outline:\n{current_outline}\n\n"
         "New Research:\n{research_summary}\n\n"
         "Instructions:\n"
         "- Maintain the existing structure\n"
         "- Mark new additions with [NEW]\n"
         "- Integrate research naturally\n"
         "- Focus on personality insights\n"
         "- Keep SEO optimization"
        ),
        ("human", "Create the enhanced outline.")
    ])
    
    structured_llm = llm.with_structured_output(OutlineProposal)
    chain = outline_prompt | structured_llm
    
    proposal = chain.invoke({
        "person": state.person,
        "current_outline": state.current_outline,
        "research_summary": state.research_summary
    })
    state.proposed_outline = proposal.outline
    
    return state


def human_feedback(state: BlogUpdateState) -> Command[Literal["create_outline", "rewrite_blog"]]:
    """Get human approval for the proposed outline"""
    state.workflow_status = WorkflowStatus.AWAITING_APPROVAL
    
    # Try to use interrupt if in an interactive context
    try:
        feedback = interrupt(
            f"## Proposed Outline for {state.person} Blog\n\n"
            f"{state.proposed_outline}\n\n"
            f"---\n"
            f"Type 'approve' to proceed with rewriting, or provide feedback for revision:"
        )
    except RuntimeError:
        # If not in interactive context, auto-approve for testing
        print(f"\n📝 Auto-approving outline for {state.person} (non-interactive mode)")
        feedback = "approve"
    
    # Log approval history
    state.approval_history.append({
        "timestamp": datetime.now().isoformat(),
        "proposed": state.proposed_outline,
        "feedback": feedback
    })
    
    if isinstance(feedback, str) and feedback.strip().lower() == "approve":
        state.approved_outline = state.proposed_outline
        return Command(goto="rewrite_blog")
    else:
        # Incorporate feedback and regenerate
        state.proposed_outline = feedback if isinstance(feedback, str) else state.proposed_outline
        return Command(goto="create_outline")


def rewrite_blog(state: BlogUpdateState) -> BlogUpdateState:
    """Rewrite the blog with new content using GPT-4"""
    state.workflow_status = WorkflowStatus.REWRITING
    
    # Use GPT-4 for high-quality writing
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are rewriting a blog about {person}'s Enneagram Type {enneagram_type}.\n\n"
         "Original Content:\n{current_content}\n\n"
         "Approved Outline:\n{approved_outline}\n\n"
         "New Research:\n{research_summary}\n\n"
         "Instructions:\n"
         "- Maintain the original voice and style\n"
         "- Seamlessly integrate new information\n"
         "- Keep all existing valuable content\n"
         "- Add [NEW] tags only in comments for tracking\n"
         "- Output in markdown format\n"
         "- Make it engaging and insightful"
        ),
        ("human", "Rewrite the blog following the approved outline.")
    ])
    
    chain = rewrite_prompt | llm
    response = chain.invoke({
        "person": state.person,
        "enneagram_type": state.enneagram_type,
        "current_content": state.current_content,
        "approved_outline": state.approved_outline,
        "research_summary": state.research_summary
    })
    
    state.rewritten_markdown = response.content
    state.rewritten_content = response.content  # Could convert to HTML if needed
    
    return state


async def save_blog(state: BlogUpdateState) -> BlogUpdateState:
    """Save the updated blog to Supabase and/or export as markdown"""
    state.workflow_status = WorkflowStatus.SAVING
    
    # Save to Supabase if configured
    if state.save_to_supabase and SUPABASE_AVAILABLE:
        supabase = get_supabase_client()
        if supabase and state.blog_metadata.get("id"):
            try:
                # Update the blog record
                update_data = {
                    "content": state.rewritten_content,
                    "markdown": state.rewritten_markdown,
                    "updated_at": datetime.now().isoformat(),
                    "version": state.blog_metadata.get("version", 0) + 1,
                    "metadata": {
                        **state.blog_metadata.get("metadata", {}),
                        "last_ai_update": datetime.now().isoformat(),
                        "research_queries": state.search_queries
                    }
                }
                
                response = supabase.table("blogs").update(update_data).eq(
                    "id", state.blog_metadata["id"]
                ).execute()
                
                if response.data:
                    print(f"✅ Blog updated in Supabase: {state.blog_metadata['id']}")
            except Exception as e:
                state.error_message = f"Failed to save to Supabase: {str(e)}"
    
    # Export markdown if configured
    if state.export_markdown:
        # Sanitize the person name for use in filename
        safe_person = state.person.replace(' ', '_').replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
        if not safe_person or safe_person == "N/A":
            safe_person = "Unknown"
        filename = f"blog_{safe_person}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        filepath = f"exports/{filename}"
        
        os.makedirs("exports", exist_ok=True)
        with open(filepath, "w") as f:
            f.write(state.rewritten_markdown)
        print(f"✅ Blog exported to: {filepath}")
    
    state.workflow_status = WorkflowStatus.COMPLETED
    return state


# ============================================================================
# BUILD THE GRAPH
# ============================================================================

def build_blog_update_graph():
    """Build the complete blog update workflow graph"""
    builder = StateGraph(BlogUpdateState)
    
    # Add all nodes
    builder.add_node("fetch_blog_content", fetch_blog_content)
    builder.add_node("analyze_content", analyze_content)
    builder.add_node("generate_search_queries", generate_search_queries)
    builder.add_node("search_web", search_web)
    builder.add_node("create_outline", create_outline)
    builder.add_node("human_feedback", human_feedback)
    builder.add_node("rewrite_blog", rewrite_blog)
    builder.add_node("save_blog", save_blog)
    
    # Define the flow
    builder.add_edge(START, "fetch_blog_content")
    builder.add_edge("fetch_blog_content", "analyze_content")
    builder.add_edge("analyze_content", "generate_search_queries")
    builder.add_edge("generate_search_queries", "search_web")
    builder.add_edge("search_web", "create_outline")
    builder.add_edge("create_outline", "human_feedback")
    # human_feedback uses Command to route
    builder.add_edge("rewrite_blog", "save_blog")
    builder.add_edge("save_blog", END)
    
    # Add memory for checkpointing
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    
    return graph


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def update_blog(blog_identifier: str, config: Optional[Dict] = None):
    """Main entry point for updating a blog
    
    Args:
        blog_identifier: URL, Supabase ID, or slug
        config: Optional configuration overrides
    """
    # Default config if None
    if config is None:
        config = {}
    
    # Initialize state
    state = BlogUpdateState(
        blog_identifier=blog_identifier,
        search_method=config.get("search_method", os.getenv("DEFAULT_SEARCH_METHOD", "tavily")),
        require_approval=config.get("require_approval", True),
        save_to_supabase=config.get("save_to_supabase", True),
        export_markdown=config.get("export_markdown", True)
    )
    
    # Build and run the graph
    graph = build_blog_update_graph()
    
    # Execute with thread ID for checkpointing
    thread_id = f"blog_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config = {"configurable": {"thread_id": thread_id}}
    
    # Run the workflow
    result = await graph.ainvoke(state, config)
    
    return result


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    # Example: Update a blog from 9takes
    blog_id = "https://9takes.com/personality-analysis/Taylor-Swift"
    
    # Or use Supabase ID/slug
    # blog_id = "some-uuid-here"
    # blog_id = "taylor-swift-enneagram"
    
    result = asyncio.run(update_blog(blog_id))
    
    if result.workflow_status == WorkflowStatus.COMPLETED:
        print("✅ Blog successfully updated!")
        print(f"   Person: {result.person}")
        print(f"   Type: {result.enneagram_type}")
    else:
        print(f"❌ Error: {result.error_message}")