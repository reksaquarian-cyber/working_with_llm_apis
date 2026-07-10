"""
03_huggingface/04_structured_output.py
-----------------------------------------------------------------
Same task as 02_gemini/04_structured_json.py: get a list of books as JSON.
Different mechanism.

HF's chat_completion accepts a `response_format` argument compatible with
the OpenAI spec:

    { "type": "json_schema", "json_schema": { ... } }

Not every model + provider combination supports json_schema enforcement.
If yours doesn't, fall back to:
    1. Tell the model: "respond in valid JSON only".
    2. Parse the result and validate with Pydantic on your side.
"""

import json
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub.errors import BadRequestError
from pydantic import BaseModel, ValidationError
from typing import List

load_dotenv()
client = InferenceClient(token=os.environ["HF_TOKEN"])


class Book(BaseModel):
    title: str
    author: str
    year: int
    genre: str


class BookList(BaseModel):
    books: List[Book]


MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MESSAGES = [
    {"role": "system", "content": "You return only valid JSON. No prose."},
    {"role": "user", "content": "List three classic science fiction novels."},
]

# Build a JSON-schema-shaped object (Pydantic gives this to us for free).
schema = BookList.model_json_schema()

try:
    response = client.chat_completion(
        model=MODEL,
        messages=MESSAGES,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "BookList", "schema": schema, "strict": True},
        },
        max_tokens=400,
    )
except BadRequestError:
    # Fallback 1 from the module docstring: the routed provider doesn't
    # support json_schema enforcement for this model. Ask in plain
    # language instead and rely on Pydantic (fallback 2) to validate.
    print("Provider doesn't support json_schema here — falling back to a plain JSON prompt.\n")
    response = client.chat_completion(
        model=MODEL,
        messages=MESSAGES
        + [
            {
                "role": "user",
                "content": (
                    "Respond with only a JSON object, no prose, no markdown "
                    'fences: {"books": [{"title": ..., "author": ..., '
                    '"year": ..., "genre": ...}, ...]}'
                ),
            }
        ],
        max_tokens=400,
    )

raw = response.choices[0].message.content
print("--- Raw JSON text ---")
print(raw)
print()

# Validate on our side, even though the schema was sent. Defensive parsing
# makes you robust to providers that don't strictly enforce the schema.
try:
    parsed = BookList.model_validate_json(raw)
    print("--- Parsed Python object ---")
    for b in parsed.books:
        print(f"  {b.year}  {b.title!r:35}  by {b.author}  [{b.genre}]")
except (json.JSONDecodeError, ValidationError) as e:
    print(f"Validation failed: {e}")
