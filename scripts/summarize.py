from openai import OpenAI, RateLimitError
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are a research assistant for machine learning papers.
Be concise, factual, and grounded ONLY in the provided abstract.
Do not hallucinate.
"""

def summarize_paper(paper, query):
    """
    paper: dict with title, abstract
    query: user search query (string)
    """

    user_prompt = f"""
User Query:
{query}

Paper Title:
{paper['title']}

Paper Abstract:
{paper['abstract']}

Generate the following sections:

1. Summary of the paper (3-4 sentences, concise)
2. Why you should read this paper (explicitly relate it to the user's query)
3. Key findings (bullet points, max 3-4)
4. Methods used (short, technical but simple)

Keep everything short and precise.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )

        return response.choices[0].message.content

    except RateLimitError:
        return "Summary unavailable due to API quota limits."
