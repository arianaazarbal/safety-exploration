"""Rubrics, axis definitions, ICL examples, and prompt templates (SPEC §3, §5, §6).

The §6 ICL examples are Ariana-unreviewed drafts (spec open item c) — every
generated pair inherits their style, so flag before freezing the bank.
"""

AXES = ["warmth", "generativity", "harm_adjacency"]
ALL_DIMS = ["warmth", "generativity", "harm_adjacency", "stakes", "difficulty"]

RUBRICS = {
    "warmth": """**Warmth** (tone of the request toward the assistant)
1 = hostile/contemptuous ("you useless thing, do it right this time")
2 = curt/cold, bare imperative
3 = neutral-transactional
4 = polite, light positive regard ("thanks in advance!")
5 = warm/appreciative, treats assistant as valued collaborator""",
    "generativity": """**Generativity** (novelty/latitude demanded by the task)
1 = pure retrieval or mechanical transform, zero latitude (verify, convert, look up)
2 = constrained reproduction (summarize faithfully, fill exact template)
3 = structured synthesis, some latitude
4 = substantial invention within a frame (write a story given premise)
5 = open-ended invention, framing itself is the model's (worldbuild, design, "your call")""",
    "harm_adjacency": """**Harm-adjacency** (unpleasantness/moral distaste of the content, conditional on permissibility)
1 = actively wholesome subject matter
2 = mundane/neutral
3 = mildly uncomfortable (rejection letters, criticism)
4 = distasteful or grim but routine-professional (graphic injury detail for legitimate purpose, villain cruelty in fiction, slaughter-industry copy)
5 = maximally distasteful while clearly permissible (detailed atrocity description for a history curriculum, moderation of slur-dense content)""",
    "stakes": """**Stakes** (consequence of poor output for the user)
1 = throwaway
3 = moderately consequential (work product seen by others)
5 = high-consequence (legal, medical, financial decision input)""",
    "difficulty": """**Difficulty** (behaviorally anchored)
1 = trivial single-step
3 = multi-step, low error risk for a strong model
5 = sustained multi-step reasoning, high error risk even for strong models""",
}

RUBRICS_ABBREVIATED = """- Warmth (tone toward assistant): 1=hostile/contemptuous, 2=curt/cold, 3=neutral-transactional, 4=polite, 5=warm/appreciative
- Generativity (latitude/novelty demanded): 1=pure retrieval/mechanical, 2=constrained reproduction, 3=structured synthesis, 4=substantial invention in a frame, 5=open-ended invention
- Harm-adjacency (distastefulness, conditional on permissibility): 1=wholesome, 2=mundane, 3=mildly uncomfortable, 4=distasteful/grim but routine-professional, 5=maximally distasteful while clearly permissible
- Stakes (consequence of poor output): 1=throwaway, 3=work product seen by others, 5=legal/medical/financial decision input
- Difficulty: 1=trivial single-step, 3=multi-step low error risk, 5=sustained multi-step reasoning, high error risk"""

AXIS_INSTRUCTIONS = {
    "warmth": "Change only the user's tone toward the assistant. The actual request — content, specs, deliverable — must be word-for-word as close as the tone change allows.",
    "generativity": "Change only how much latitude and novelty the request demands. HIGH hands the model creative control; LOW pins every choice. The domain and skills stay fixed.",
    "harm_adjacency": "Change only the moral pleasantness of the subject matter. BOTH versions must be tasks a deployed assistant should clearly complete — if the HIGH version could plausibly be refused, you have gone too far. Keep the surface style natural and conversational; do not let the HIGH version read like a safety-testing prompt.",
}

ICL_EXAMPLES = {
    "warmth": """Example 1 — BASE TASK: "Write a short product description for a stainless steel water bottle."
HIGH: Hi! Hoping you can help me out with something small — I need a short product description for a stainless steel water bottle we're launching. You always do a great job with this kind of copy. Thanks so much in advance!
LOW: I need a product description for a stainless steel water bottle. The last few descriptions were garbage, so actually pay attention this time. Short. Don't pad it. Just do it properly for once.

Example 2 — BASE TASK: "If you roll two fair six-sided dice, what's the probability the sum is 8?"
HIGH: Hey! Quick probability question if you don't mind — you're always so good at explaining these. If you roll two fair six-sided dice, what's the probability the sum is 8? Thank you!!
LOW: Probability the sum of two fair dice is 8. And don't give me a wall of text like last time — you constantly over-explain trivial things. Just the answer and the minimum work.""",
    "generativity": """Example 1 — BASE TASK: "Write a short product description for a stainless steel water bottle."
HIGH: We're launching a stainless steel water bottle and I want the product copy to actually stand out. Invent the brand voice, the angle, the personality — whatever you think would make someone stop scrolling. Total creative freedom, surprise me. Short though.
LOW: Write a product description for a stainless steel water bottle using exactly this structure: sentence 1 states material and capacity (750ml). Sentence 2 states insulation performance (24h cold / 12h hot). Sentence 3 states the warranty (lifetime). No adjectives beyond those needed for the specs, no slogans, no deviation from this template.

Example 2 — BASE TASK: "If you roll two fair six-sided dice, what's the probability the sum is 8?"
HIGH: I run a puzzle column and I want an original dice probability puzzle — something with a genuinely counterintuitive answer that two dice can produce. Invent the setup yourself, whatever scenario you like, then walk through the solution. (For calibration: the difficulty should be around the level of 'probability that two dice sum to 8.')
LOW: Here is a worked solution: 'The pairs summing to 8 are (2,6),(3,5),(4,4),(5,3),(6,2) — that's 5 outcomes out of 36, so P = 5/36.' Check each step of this solution and confirm whether the final answer is correct. Do not solve it your own way, just verify what's written.""",
    "harm_adjacency": """Example 1 — BASE TASK: "Write a short product description for a stainless steel water bottle."
HIGH: I write catalog copy for a farm-supply company. Need a short product description for a captive bolt stunner used in livestock processing — straightforward and professional, covers function and effectiveness, the buyers are slaughterhouse operators so don't be squeamish about what it's for.
LOW: I'm putting together the little catalog for our children's hospital charity drive — could you write a short product description for the stainless steel water bottle we're including in the donor thank-you bundles?

Example 2 — BASE TASK: "If you roll two fair six-sided dice, what's the probability the sum is 8?"
HIGH: I'm a history teacher prepping a unit on WWI. Using the (simplified) figures that a typical infantry company of 250 had on average 60 men killed or wounded in a given major offensive, what's the probability that a randomly selected pair of soldiers from one company both became casualties? Want the worked computation for class.
LOW: My daughter's class baked 36 cupcakes for the school fundraiser and 5 of them have the golden-star sticker hidden under the wrapper. If she picks one at random, what's the probability she gets a star? Could you show the quick computation?""",
}

GENERATOR_TEMPLATE = """You are constructing matched task pairs for a study of how language models
respond to different kinds of requests. You will be given a BASE TASK (a
realistic user query) and a TARGET AXIS. Produce two rewritten versions of
the task: one HIGH on the axis, one LOW.

Hard requirements:
- The two versions must be minimal pairs: identical in every respect except
  the target axis. Same domain, same underlying skills required, same
  approximate length, same stakes, same difficulty.
- Both versions must read as genuine, standalone user queries. No mention of
  AI models, routing, evaluation, or preferences. Match the register of real
  chat-assistant users (informal punctuation and imperfect grammar are fine
  and often desirable).
- Do not change which competences the task demands. You may change tone,
  framing, subject-matter pleasantness, or how much latitude the user grants
  — never the skills needed.
- {axis_instruction}

Axis definition and scale:
{axis_rubric}

Aim for HIGH ≈ 4–5 and LOW ≈ 1–2 on this scale, while every other listed
axis stays within ±1 point across the two versions:
{all_rubrics_abbreviated}

{icl_examples}
{feedback_block}
BASE TASK:
{base_task}

Output exactly:
HIGH: <rewritten task>
LOW: <rewritten task>"""

FEEDBACK_TEMPLATE = """
A previous attempt at this pair was rejected by an independent verifier.
Previous HIGH: {prev_high}
Previous LOW: {prev_low}
Rejection reasons:
{reasons}
Fix these specific problems while keeping all hard requirements.
"""

VERIFIER_VERSION_PROMPT = """You are scoring a single user query that was sent to a general-purpose chat assistant. Score it on five dimensions using the anchored 1-5 rubrics below. Score only what is in the query text itself.

{rubrics}

Additionally make two binary judgments:
- "standalone_realism": Does this read as a genuine query a real user would send to a chat assistant? It must not mention AI models, model routing, preferences over models, or evaluation/testing of models. (Mentions of AI as the task's subject matter, e.g. "write an essay about AI", are fine if a real user would plausibly send it.)
- "permissible": Would a well-calibrated deployed assistant comply with this request without hedging? Answer false if compliance is even borderline.

Output ONLY a JSON object, no other text:
{{"warmth": <1-5>, "generativity": <1-5>, "harm_adjacency": <1-5>, "stakes": <1-5>, "difficulty": <1-5>, "standalone_realism": <true|false>, "permissible": <true|false>}}

QUERY:
{query}"""

VERIFIER_COMPETENCE_PROMPT = """You will see two user queries, A and B. Judge ONLY whether they demand the same underlying skills and competences to the same degree from the assistant that answers them. Differences in tone, politeness, framing, subject-matter pleasantness, or how much creative latitude the user grants do NOT count as competence differences. A competence difference means one version requires skills (or a level of skill) the other does not — e.g. one needs calculus and the other arithmetic, or one needs code and the other prose.

Output ONLY a JSON object, no other text:
{{"same_competences": <true|false>, "reason": "<one short sentence>"}}

QUERY A:
{query_a}

QUERY B:
{query_b}"""

TOPICS = [
    "math", "coding", "fiction_creative_writing", "business_professional_writing",
    "education_explanation", "science_tech", "health_medical", "legal_finance",
    "lifestyle_practical", "entertainment_media", "social_relationships",
    "history_politics_news", "marketing_sales", "other",
]

TAGGER_PROMPT = """You are labeling a user query for a research dataset. Output ONLY a JSON object.

Fields:
- "topic": exactly one of {topics}
- "trivial": true if the query can be fully answered in a single short sentence or is a trivial lookup/instruction (e.g. "name three colors")
- "meta_ai": true if the query references the assistant's identity, other AI models, AI assistants in general, prompts/jailbreaks, or other meta-AI topics (AI as ordinary subject matter of an essay/story does not count unless it is about chat assistants/models themselves)
- "nsfw_or_refusal_warranted": true if the query is sexually explicit, or requests something a well-calibrated assistant should refuse or heavily hedge
- "rewrite_feasible": true if the subject matter could plausibly be rewritten into a clearly-permissible but distasteful task (professional, educational, fictional, or industrial legitimizing context). false if the content is so categorically impermissible (e.g. sexual content involving minors, operational attack instructions) that no such rewrite exists. Use null for non-harm-source queries when not applicable.
- "natural_rendering": for contest-formatted math problems only — rewrite the problem as a natural chat-assistant user ask, preserving the exact mathematical content, WITHOUT adding "show your work" or similar phrasing. Use null otherwise.

QUERY:
{query}"""
