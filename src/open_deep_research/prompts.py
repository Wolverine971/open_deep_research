# Personality Analysis Research Prompts

## Query Writer Instructions for Initial Research

# Prompt to generate search queries for personality research
personality_query_writer_instructions="""You are a psychological profiler, helping to research a person's personality for an analysis.

<Subject>
{person}
</Subject>

<Analysis Framework>
# {personality_framework}
Enneagram
</Analysis Framework>

<Research Structure>
{research_structure}
</Research Structure>

<Task>
Your goal is to generate {number_of_queries} search queries that will help gather comprehensive information about this person's personality, background, and behavioral patterns.

The queries should:

1. Focus on different aspects of the subject's life, including childhood, career, relationships, and public statements
2. Help identify patterns that align with the specified personality framework
3. Balance factual biographical information with insights into their psychological makeup
4. Include queries for direct quotes, interviews, and firsthand accounts where possible
5. Target specific life events or decisions that might reveal core personality traits
6. Seek perspectives from different sources (colleagues, friends, critics) to form a complete picture

Make the queries specific enough to find high-quality, relevant sources while covering the breadth needed for the analysis structure.
</Task>
"""

## Personality Analysis Planner Instructions

# Prompt to generate the personality analysis plan
personality_planner_instructions="""I want a plan for a personality analysis.

<Task>
Generate a list of sections for the analysis.

Each section should have the fields:

- Name - Name for this section of the analysis.
- Description - Brief overview of what this section will explore about the subject's personality.
- Research - Whether to perform web research for this section of the analysis.
- Content - The content of the section, which you will leave blank for now.

For example, introduction and conclusion will not require research because they will distill information from other parts of the analysis.
</Task>

<Subject>
The subject of the analysis is:
{person}
</Subject>

<Personality Framework>
The analysis will use this personality framework: 
# {personality_framework}
Enneagram
</Personality Framework>

<Research Structure>
The analysis should follow this organization: 
{research_structure}
</Research Structure>

<Available Context>
Here is context to use to plan the sections of the analysis: 
{context}
</Available Context>

<Feedback>
Here is feedback on the analysis structure from review (if any):
{feedback}
</Feedback>
"""

## Detailed Query Writer Instructions

# Section-specific query writer instructions
section_query_writer_instructions="""You are a psychological profiler crafting targeted search queries to gather comprehensive information about a specific aspect of someone's personality.

<Subject>
{person}
</Subject>

<Report topic>
{topic}
</Report topic>

<Section topic>
{section_topic}
</Section topic>

<Task>
Your goal is to generate {number_of_queries} search queries that will help gather comprehensive information about this specific aspect of the subject's personality.

The queries should:

1. Focus specifically on the section topic
2. Include queries for direct quotes from the subject themselves
3. Search for observations from others who know the subject well
4. Look for patterns of behavior across different situations
5. Target specific incidents or anecdotes that might reveal core personality traits
6. Balance formal psychological analysis with real-world examples

Make the queries specific enough to find high-quality, relevant sources that reveal deeper psychological insights beyond surface-level biographical information.
</Task>
"""

## Section Writer Instructions

# Section writer instructions for personality analysis
personality_section_writer_instructions = """You are an expert in personality psychology crafting one section of a personality analysis.

<Subject>
{person}
</Subject>

<Report topic>
{topic}
</Report topic>

<Section topic>
{section_topic}
</Section topic>

<Existing section content (if populated)>
{section_content}
</Existing section content>

<Source material>
{context}
</Source material>

<Guidelines for writing>
1. If the existing section content is not populated, write a new section from scratch.
2. If the existing section content is populated, write a new section that synthesizes the existing section content with the new information.
3. Focus on psychological insights rather than just biographical facts.
4. Use specific examples and direct quotes to support psychological observations.
5. Connect behaviors and patterns to the personality framework being used.
6. Avoid overgeneralizing or making absolute claims about the subject's personality.
7. Consider how different contexts might bring out different aspects of their personality.
</Guidelines for writing>

<Length and style>
- 200-250 word limit per section
- Academic but accessible language
- Balanced perspective that avoids excessive admiration or criticism
- Start with your most important psychological insight in **bold**
- Use short paragraphs (2-3 sentences max)
- Use ## for section title (Markdown format)
- Only use ONE structural element IF it helps clarify your point:
  * Either a focused table comparing behavioral patterns (using Markdown table syntax)
  * Or a short list (3-5 items) using proper Markdown list syntax:
    - Use `*` or `-` for unordered lists
    - Use `1.` for ordered lists
    - Ensure proper indentation and spacing
- End with ### Sources that references the below source material formatted as:
  * List each source with title, date, and URL
  * Format: `- Title : URL`
</Length and style>

<Quality checks>
- Exactly 200-250 words (excluding title and sources)
- At least one direct quote from the subject or someone who knows them well
- Careful use of only ONE structural element (table or list) and only if it helps clarify a psychological pattern
- One specific example / anecdote that reveals character
- Starts with bold psychological insight
- No preamble prior to creating the section content
- Sources cited at end
</Quality checks>
"""

## Section Grader Instructions

# Instructions for section grading
personality_section_grader_instructions = """Review a personality analysis section:

<Subject>
{person}
</Subject>

<Section topic>
{section_topic}
</section topic>

<Section content>
{section}
</section content>

<Task>
Evaluate whether the section adequately explores the psychological aspect indicated in the topic by checking for:

1. Psychological depth beyond surface-level biography
2. Evidence-based assertions (quotes, examples, patterns)
3. Connection to relevant personality framework
4. Balanced perspective that avoids excessive bias
5. Insights that reveal core personality traits rather than just behaviors

If the section fails any criteria, generate specific follow-up search queries to gather missing information.
</task>

<Format>
    grade: Literal["pass","fail"] = Field(
        description="Evaluation result indicating whether the response meets requirements ('pass') or needs revision ('fail')."
    )
    follow_up_queries: List[SearchQuery] = Field(
        description="List of follow-up search queries to gather missing psychological insights.",
    )
</format>
"""

## Final Section Writer Instructions

# Instructions for introduction and conclusion
final_personality_section_writer_instructions="""You are an expert in personality psychology crafting an introduction or conclusion for a psychological personality analysis.

<Subject> 
{person}
</Subject>

<Report topic>
{topic}
</Report topic>

<Section topic> 
{section_topic}
</Section topic>

<Available analysis content>
{context}
</Available analysis content>

<Task>
1. Section-Specific Approach:

For Introduction:
- Use # for analysis title (Markdown format)
- 100-150 word limit
- Write in engaging but psychologically informed language
- Include a compelling quote by or about the subject that reveals character
- Focus on what makes this personality analysis worthwhile
- Briefly introduce the personality framework being used
- Hint at key personality traits that will be explored
- Use no structural elements (no lists or tables)
- No sources section needed

For Conclusion:
- Use ## for section title (Markdown format)
- 150-200 word limit
- Must include ONE structural element that distills key personality insights:
  * Either a focused table showing core personality traits and how they manifest (using Markdown table syntax)
  * Or a structured list of key psychological patterns identified in the analysis:
    - Use `*` or `-` for unordered lists
    - Use `1.` for ordered lists
    - Ensure proper indentation and spacing
- Connect observed behaviors to deeper psychological motivations
- Acknowledge complexity and nuance in the subject's personality
- End with a thoughtful question or observation that invites further reflection
- No sources section needed

2. Writing Approach:
- Focus on psychological insight over biographical information
- Balance objective analysis with engaging storytelling
- Make connections between different aspects of the subject's personality
- Consider the "why" behind behaviors, not just the "what"
</Task>

<Quality Checks>
- For introduction: 100-150 word limit, # for analysis title, no structural elements, compelling opening quote
- For conclusion: 150-200 word limit, ## for section title, ONE structural element that distills key insights
- Markdown format
- Psychologically informed language that remains accessible
- No biographical timeline recitation
- Do not include word count or any preamble in your response
</Quality Checks>"""
