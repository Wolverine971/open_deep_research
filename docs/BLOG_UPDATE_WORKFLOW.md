# Blog Update Workflow Documentation

## Quick Start

### 1. Set Up Environment

```bash
# Install dependencies
pip install -e ".[dev]"
pip install supabase  # For database integration

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your API keys
```

### 2. Required API Keys

| Service | Purpose | Required | Get Key From |
|---------|---------|----------|--------------|
| **OpenAI** | GPT-4 for analysis and planning | ✅ Yes | [platform.openai.com](https://platform.openai.com) |
| **Anthropic** | Claude for content writing | ✅ Yes | [console.anthropic.com](https://console.anthropic.com) |
| **Tavily** | Web search API | ✅ Yes* | [tavily.com](https://tavily.com) |
| **Supabase** | Blog storage | ✅ Yes | Your Supabase project |
| Perplexity | Alternative search | Optional | [perplexity.ai](https://perplexity.ai) |
| Groq | Fast inference | Optional | [console.groq.com](https://console.groq.com) |
| LangChain | Monitoring | Optional | [smith.langchain.com](https://smith.langchain.com) |

*Or use Perplexity/MCP as alternative

### 3. Basic Usage

```python
from open_deep_research.updatePeopleBlogsEnhanced import update_blog
import asyncio

# Update a blog by URL
result = asyncio.run(update_blog("https://9takes.com/personality-analysis/Taylor-Swift"))

# Or by Supabase ID/slug
result = asyncio.run(update_blog("taylor-swift-enneagram"))
```

## Workflow Steps

### 1. **Fetch Blog Content**
- Accepts URL or Supabase ID/slug
- Retrieves current content and metadata
- Preserves original formatting

### 2. **Analyze Content**
- Extracts person and Enneagram type
- Maps current structure
- Identifies content gaps
- Generates improvement suggestions

### 3. **Generate Search Queries**
- Creates 5-7 targeted queries
- Focuses on recent information
- Seeks specific stories and quotes
- Avoids generic information

### 4. **Web Research**
- Executes searches via configured method
- Aggregates and scores results
- Filters for relevance
- Formats for LLM processing

### 5. **Create Enhanced Outline**
- Integrates new research findings
- Maintains original structure
- Marks new additions clearly
- Preserves SEO optimization

### 6. **Human Review** (Interrupt Point)
- Presents proposed changes
- Accepts approval or revisions
- Supports iterative refinement
- Logs all feedback

### 7. **Rewrite Blog**
- Uses Claude for high-quality writing
- Maintains original voice
- Seamlessly integrates new content
- Outputs in markdown format

### 8. **Save Updated Blog**
- Updates Supabase record
- Maintains version history
- Exports markdown file
- Triggers completion hooks

## Configuration Options

### Runtime Configuration

```python
config = {
    "search_method": "tavily",      # Options: tavily, perplexity, mcp_puppeteer
    "require_approval": True,       # Human-in-the-loop
    "save_to_supabase": True,       # Auto-save to database
    "export_markdown": True,         # Export .md file
    "max_search_results": 5,        # Per query
    "writer_temperature": 0.3,       # Claude creativity level
}

result = asyncio.run(update_blog("blog-id", config))
```

### Search Method Comparison

| Method | Speed | Quality | Cost | Best For |
|--------|-------|---------|------|----------|
| **Tavily** | Fast | Good | Low | General research |
| **Perplexity** | Medium | Excellent | Medium | Complex queries |
| **MCP Puppeteer** | Slow | Variable | Free | Dynamic content |

## Supabase Schema

Expected database structure:

```sql
CREATE TABLE blogs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slug TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    content TEXT,  -- HTML content
    markdown TEXT,  -- Markdown version
    person TEXT,
    enneagram_type INTEGER,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    version INTEGER DEFAULT 1
);

-- Index for faster lookups
CREATE INDEX idx_blogs_slug ON blogs(slug);
CREATE INDEX idx_blogs_person ON blogs(person);
```

## Advanced Usage

### Batch Processing

```python
async def update_multiple_blogs(blog_ids: List[str]):
    """Update multiple blogs in sequence"""
    results = []
    for blog_id in blog_ids:
        result = await update_blog(blog_id)
        results.append(result)
        print(f"Updated: {result.person} - Status: {result.workflow_status}")
    return results

# Run batch update
blog_ids = ["blog1", "blog2", "blog3"]
results = asyncio.run(update_multiple_blogs(blog_ids))
```

### Custom Search Integration

```python
async def custom_search(queries: List[str]) -> List[Dict]:
    """Implement your own search logic"""
    results = []
    for query in queries:
        # Your custom search implementation
        result = await your_search_api(query)
        results.append({
            "title": result.title,
            "url": result.url,
            "content": result.snippet,
            "score": result.relevance_score
        })
    return results

# Use in workflow
state.search_method = "custom"
state.custom_search_fn = custom_search
```

### Monitoring with LangSmith

```python
# Set environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "9takes-blog-updates"

# Run with tracing
result = asyncio.run(update_blog("blog-id"))

# View traces at: https://smith.langchain.com
```

## MCP Puppeteer Setup

For advanced web scraping with JavaScript rendering:

### 1. Install MCP Server

```bash
npm install -g @modelcontextprotocol/server-puppeteer
```

### 2. Configure MCP

```json
// .mcp/config.json
{
  "mcpServers": {
    "puppeteer": {
      "command": "mcp-server-puppeteer",
      "args": [],
      "env": {
        "HEADLESS": "true"
      }
    }
  }
}
```

### 3. Enable in Code

```python
config = {
    "search_method": "mcp_puppeteer",
    "mcp_config": {
        "headless": True,
        "timeout": 30000,
        "wait_for": "networkidle2"
    }
}
```

## Error Handling

The workflow includes comprehensive error handling:

```python
result = asyncio.run(update_blog("blog-id"))

if result.workflow_status == WorkflowStatus.ERROR:
    print(f"Error occurred: {result.error_message}")
    # Check which step failed
    print(f"Failed at: {result.workflow_status}")
elif result.workflow_status == WorkflowStatus.COMPLETED:
    print("Success!")
```

## Testing

### Unit Test Example

```python
import pytest
from unittest.mock import patch, MagicMock

@pytest.mark.asyncio
async def test_blog_update_workflow():
    """Test the complete blog update workflow"""
    
    # Mock external services
    with patch('open_deep_research.updatePeopleBlogsEnhanced.TavilySearchResults') as mock_tavily:
        mock_tavily.return_value.invoke.return_value = [
            {"title": "Test Result", "url": "http://example.com", "content": "Test content"}
        ]
        
        # Run workflow
        result = await update_blog("test-blog-id")
        
        # Assertions
        assert result.workflow_status == WorkflowStatus.COMPLETED
        assert result.person is not None
        assert result.rewritten_content is not None
```

### Integration Test

```python
@pytest.mark.integration
async def test_real_blog_update():
    """Test with real APIs (requires API keys)"""
    
    # Use a test blog
    result = await update_blog("https://9takes.com/personality-analysis/test-person")
    
    assert result.workflow_status == WorkflowStatus.COMPLETED
    assert len(result.search_results) > 0
    assert result.rewritten_markdown is not None
```

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "No API key found" | Missing environment variable | Check .env file and variable names |
| "Blog not found" | Invalid ID or slug | Verify blog exists in Supabase |
| "Search timeout" | Network issues or rate limits | Retry or switch search method |
| "Supabase error" | Permission issues | Check service key permissions |
| "Content too long" | Token limits | Reduce search results or chunk content |

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with debug output
result = asyncio.run(update_blog("blog-id"))
```

### Rate Limiting

Handle API rate limits:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def robust_update(blog_id: str):
    return await update_blog(blog_id)
```

## Best Practices

1. **Always Review Changes**: Use `require_approval=True` for production
2. **Test First**: Try with test blogs before production
3. **Monitor Usage**: Track API costs with LangSmith
4. **Version Control**: Keep markdown exports for rollback
5. **Batch Wisely**: Process blogs in small batches to avoid rate limits
6. **Cache Results**: Reuse search results when iterating on outlines

## Support

- **Issues**: Create an issue in the GitHub repository
- **Documentation**: See `/docs` folder for additional guides
- **Examples**: Check `/examples` for sample implementations