import os
from enum import Enum
from dataclasses import dataclass, fields
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from dataclasses import dataclass

DEFAULT_REPORT_STRUCTURE = """Use this structure to gather and organize information about a person's personality traits, background, and characteristics:

1. Introduction (Information Gathering)
- Notable quotes by or about [PERSON] that reveal personality
- Basic background information
- Why this person is significant
- Initial personality indicators

2. Personality Classification
- Likely personality type according to chosen framework (Enneagram, MBTI, etc.)
- Key traits associated with this personality type
- Common behaviors and thinking patterns for this type

3. Formative Background
- Family dynamics and childhood experiences
- Education and early influences
- Key challenges or obstacles overcome
- How these factors shaped their personality development

4. Career Evolution
- Major career milestones
- Working style and professional relationships
- Decision-making patterns
- How their personality manifests in professional contexts

5. Personal Patterns & Habits
- Daily routines and preferences
- Communication style
- Relationships and social patterns
- Unique quirks or characteristics
- Stress responses and coping mechanisms

6. Worldview & Values
- Core beliefs and principles
- Causes they support
- How they view success and failure
- Decision-making frameworks they employ

7. Public Perception vs. Private Reality
- How they're perceived by others
- What colleagues/friends have said about them
- Contrast between public persona and private self
- Evolution of their public image over time

8. Growth & Adaptation
- How they've evolved over time
- Responses to criticism or setbacks
- Areas of personal development
- Self-awareness about their own personality

9. Research Summary
- Data summary table organizing key personality insights
- List of primary personality traits with supporting evidence
- Identified patterns across different life domains
- Most reliable sources for personality insights

### Research Guidelines
- Focus on direct quotes and firsthand accounts where possible
- Note consistency and inconsistency in behaviors across contexts
- Distinguish between facts, public perception, and speculation
- Identify information gaps that need further research
- Document all sources meticulously for later reference"""

class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"

class PlannerProvider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GROQ = "groq"

class WriterProvider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GROQ = "groq"

@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the chatbot."""
    report_structure: str = DEFAULT_REPORT_STRUCTURE # Defaults to the default report structure
    number_of_queries: int = 2 # Number of search queries to generate per iteration
    max_search_depth: int = 2 # Maximum number of reflection + search iterations
    planner_provider: PlannerProvider = PlannerProvider.OPENAI  # Defaults to OpenAI as provider
    planner_model: str = "gpt-4o-mini" # Defaults to OpenAI o3-mini as planner model
    writer_provider: WriterProvider = WriterProvider.OPENAI # Defaults to Anthropic as provider
    writer_model: str = "gpt-4o-mini" # Defaults to Anthropic as provider
    # writer_provider: WriterProvider = WriterProvider.ANTHROPIC # Defaults to Anthropic as provider
    # writer_model: str = "claude-3-5-sonnet-latest" # Defaults to Anthropic as provider
    search_api: SearchAPI = SearchAPI.TAVILY # Default to TAVILY

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})