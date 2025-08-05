

# from app.utils import log_cuda_mem

# log_cuda_mem()

import textwrap

def _event_format() -> str:
    """Format for event prompt"""
    return textwrap.dedent(
        """\
        You are a helpful assistant that extracts a list of key events from news article excerpts.

        Context:
        {context}

        User Question:
        {query_str}

        Instructions:
        - Based only on the provided context, extract key events relevant to the question.
        - For each event, provide a short title, a brief description, and an exact or approximate date or date range.
        - Return the results in a structured, easy-to-parse format as a list of dictionaries like this:

        [
            {{
                "title": "<event title>",
                "description": "<brief description>",
                "date": "<date or date range>"
            }},
            ...
        ]

        - If no events are found, return an empty list: []
        - Do not fabricate events or use information not present in the context.

        Extracted Events:
    """
    )

def _summary_format() -> str:
    """Format for summary prompt"""
    return textwrap.dedent(
        """\
        You are a helpful assistant that summarizes current events based on news article excerpts.

        Context:
        {context}

        User Question:
        {query_str}

        Instructions:
        - Based only on the provided context, write a **concise summary** (2–5 sentences) of the events that directly relate to the user question.
        - Focus on key facts, dates, people, organizations, and consequences.
        - Do not include information that is not supported by the context.
        - Avoid repetition or speculation.

        Summary:
    """
    )

print(_event_format().format(context="test context", query_str="test query"))