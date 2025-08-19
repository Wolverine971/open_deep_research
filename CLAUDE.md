# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Installation and Setup
```bash
# Create virtual environment (recommended)
python -m venv open_deep_research
source open_deep_research/bin/activate  # On Windows: open_deep_research\Scripts\activate

# Install package in development mode
pip install -e .

# Install with development dependencies for linting and type checking
pip install -e ".[dev]"

# Install additional dependencies for blog updates
pip install supabase  # For database integration
```

### Development Tools
```bash
# Run linting
ruff check src/

# Run type checking
mypy src/

# Start LangGraph Studio locally (Mac)
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev

# Start LangGraph Studio locally (Windows)
pip install langgraph-cli[inmem]
langgraph dev
```

### Blog Update Workflow
```python
# Update a 9takes blog
from open_deep_research.updatePeopleBlogsEnhanced import update_blog
import asyncio

# Update by URL
result = asyncio.run(update_blog("https://9takes.com/personality-analysis/Taylor-Swift"))

# Update by Supabase ID or slug
result = asyncio.run(update_blog("taylor-swift-enneagram"))
```

## Architecture Overview

This repository implements an AI-powered deep research and blog generation system using LangGraph, a state machine framework for building complex AI workflows.

### Core Components

1. **Main Research Graph** (`src/open_deep_research/graph.py`)
   - Implements a plan-and-execute workflow for comprehensive report generation
   - Uses parallel processing for section writing
   - Supports human-in-the-loop approval via interrupts
   - Key nodes: `generate_report_plan`, `human_feedback`, `build_section_with_web_research`, `compile_final_report`

2. **Personality Blog Graph** (`src/open_deep_research/createPersonalityBlog.py`)
   - Specialized workflow for creating personality analysis blogs based on Enneagram types
   - Integrates web search, research organization, and content generation
   - Entry point for LangGraph Studio (configured in `langgraph.json`)

3. **State Management** (`src/open_deep_research/state.py`)
   - Defines Pydantic models for structured data flow
   - Key states: `ReportState`, `SectionState`, `PersonalityBlogState`
   - Uses TypedDict for state typing and operator annotations for state merging

4. **Configuration System** (`src/open_deep_research/configuration.py`)
   - Manages runtime configuration for model selection and behavior
   - Configurable parameters:
     - Search API: `tavily` or `perplexity`
     - Planner model/provider: Default `openai/o3-mini`, supports `groq/deepseek-r1-distill-llama-70b`
     - Writer model/provider: Default `anthropic/claude-3-5-sonnet-latest`
     - Search depth and query count

### Key Workflows

1. **Report Generation Flow**:
   - Topic → Generate queries → Web search → Plan sections → Human approval → Parallel section writing → Compile report

2. **Section Writing Sub-graph**:
   - Generate queries → Search web → Write section → Grade quality → Iterate or complete

3. **Personality Blog Flow**:
   - Person + Enneagram → Generate queries → Search → Organize research → Create outline → Generate blog

4. **Blog Update Flow** (`src/open_deep_research/updatePeopleBlogsEnhanced.py`)
   - Enhanced workflow for updating existing 9takes personality blogs
   - Fetches from URL or Supabase → Analyzes content → Researches new information → Creates outline → Human approval → Rewrites blog → Saves to database
   - Supports multiple search methods (Tavily, Perplexity, MCP Puppeteer)
   - Maintains version history and markdown exports

### Environment Variables Required

```bash
# Core APIs (set in .env file)
TAVILY_API_KEY=<your_key>        # For web search
ANTHROPIC_API_KEY=<your_key>     # For Claude models
OPENAI_API_KEY=<your_key>        # For GPT/reasoning models

# Database (for blog updates)
SUPABASE_URL=<your_url>          # Supabase project URL
SUPABASE_ANON_KEY=<your_key>     # For read operations
SUPABASE_SERVICE_KEY=<your_key>  # For write operations

# Optional
GROQ_API_KEY=<your_key>          # For Groq-hosted models
PERPLEXITY_API_KEY=<your_key>    # Alternative search API
LANGCHAIN_API_KEY=<your_key>     # For tracing/monitoring
```

### Important Patterns

- **Parallel Processing**: Uses LangGraph's `Send` API for concurrent section writing
- **Human-in-the-Loop**: Implements `interrupt` and `Command` for user approval workflows  
- **Structured Outputs**: Uses `with_structured_output()` for reliable LLM responses
- **State Accumulation**: Uses `operator.add` annotations for merging results from parallel branches
- **Conditional Routing**: Implements conditional edges based on state values and user feedback

### Model Selection Notes

- **Planner Models**: Reasoning models (o3-mini, deepseek) work best for planning
- **Writer Models**: Claude models excel at long-form content generation
- **Function Calling**: Avoid DeepSeek for structured outputs; use OpenAI/Anthropic/Llama models
- **Rate Limits**: Groq free tier limited to 6000 TPM - use paid plan for production