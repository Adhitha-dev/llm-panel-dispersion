EVALUATION_PROMPT_TEMPLATE = """You are an expert early-stage startup investor. You are evaluating a pitch for a new startup idea.
Read the following idea text carefully.

Your task is to provide a structured judgment profile for this idea. You must output YOUR ENTIRE RESPONSE as a valid JSON object matching this exact schema, with no additional text or markdown formatting outside the JSON:

{{
  "score": <integer 1-10, overall potential>,
  "verdict": <"invest" or "pass">,
  "confidence": <integer 0-100, your confidence in this judgment>,
  "rubric_scores": {{
    "market_potential": <integer 0-5>,
    "technical_feasibility": <integer 0-5>,
    "business_viability": <integer 0-5>
  }},
  "reasoning": "<1 to 2 sentences explaining your verdict, maximum 60 words>"
}}

Here is the idea text:
{idea_text}
"""
