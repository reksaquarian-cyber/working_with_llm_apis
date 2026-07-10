# `04_structured_output.py` — Concept Guide

## WHY (purpose)

The same need as `02_gemini/04_structured_json.py`: get JSON that conforms
to a schema, every call, with no string munging. The HF route is worth
seeing on its own because:

* It uses the **OpenAI-compatible `response_format` spec**, so the same
  block of code works against OpenAI, HF Inference, vLLM with an
  OpenAI-compatible server, and most third-party providers.
* Enforcement strictness varies by provider/model. So the *defensive*
  habit — validate again on your side with Pydantic — matters more here
  than it does on Gemini.
* HF's `InferenceClient` can auto-route a request to any provider hosting
  the model. We tested `meta-llama/Llama-3.1-8B-Instruct` against all four
  providers HF lists for it: `nscale` and `deepinfra` both honour strict
  `json_schema`; `novita` rejects it outright with a 400. The script pins
  `provider="nscale"` so the strict path succeeds reliably instead of
  depending on which provider "auto" happens to route you to.

## WHAT (technical concepts)

| Concept | What it means |
|---|---|
| **OpenAI `response_format`** | A spec for asking models to return JSON. `type: "json_object"` (free-form JSON), `type: "json_schema"` (constrained to a schema). Adopted by HF and many other providers. |
| **`json_schema` block** | `{ "name": "...", "schema": <jsonschema dict>, "strict": true }`. `strict` asks the server to reject any output that doesn't validate. Not all providers honour `strict`. |
| **Pydantic → JSON Schema** | `MyModel.model_json_schema()` returns a dict in JSON-Schema-Draft form. Pydantic generates the schema automatically from your typed class — no hand-writing. |
| **Defensive parsing** | Even when the server claims to enforce the schema, validate again locally. `BaseModel.model_validate_json(raw)` parses + validates; both `json.JSONDecodeError` and `pydantic.ValidationError` should be handled. |
| **Why "system: only valid JSON"** | A safety net for models/providers that *don't* enforce strictly. Cheap insurance — it costs you a few extra tokens. |
| **`provider="nscale"`** | Pins the request to one HF Inference Provider instead of letting HF's router pick one for you. Needed here because strict `json_schema` support is inconsistent across providers for the same model — see WHY above. |
| **`try/except BadRequestError`** | Wraps the strict `response_format` call. If it 400s (e.g., you swap in a model/provider that doesn't support `strict` json_schema), the `except` re-sends the same request as a plain-language JSON instruction instead of crashing. With `provider="nscale"` this branch shouldn't fire for the default model — it's there for when you change either one. |

## HOW (code walkthrough)

```python
from pydantic import BaseModel, ValidationError
from typing import List

class Book(BaseModel):
    title: str; author: str; year: int; genre: str

class BookList(BaseModel):
    books: List[Book]

schema = BookList.model_json_schema()
```

Pydantic does the schema work. You never hand-author JSON Schema.

```python
client = InferenceClient(token=os.environ["HF_TOKEN"], provider="nscale")

try:
    response = client.chat_completion(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {"role": "system", "content": "You return only valid JSON. No prose."},
            {"role": "user",   "content": "List three classic science fiction novels."},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "BookList", "schema": schema, "strict": True},
        },
        max_tokens=400,
    )
except BadRequestError:
    # provider/model doesn't honour strict json_schema — ask in plain
    # language instead and lean on Pydantic to validate.
    response = client.chat_completion(..., response_format=None)  # see script for full fallback

raw = response.choices[0].message.content
```

`response_format` is the OpenAI spec. The `system` message is the safety
net. `raw` is a JSON string — even with strict enforcement, treat it as
text until you've validated. The `try/except` is the important bit: it's
what makes the difference between "this breaks the moment I try a
different model" and "this degrades gracefully."

```python
try:
    parsed = BookList.model_validate_json(raw)
    for b in parsed.books:
        print(f"  {b.year}  {b.title!r}  by {b.author}  [{b.genre}]")
except (json.JSONDecodeError, ValidationError) as e:
    print(f"Validation failed: {e}")
```

Catch *both* exceptions. JSON-decode errors mean the model emitted
non-JSON; validation errors mean it emitted JSON of the wrong shape. In
production, log the raw text and the prompt — that's how you debug the
prompt later.

**Run it:**

```bash
python 03_huggingface/04_structured_output.py
```

Try changing `provider="nscale"` to `provider="novita"` — the strict
request will 400, you'll see the script print its fallback notice, and
generation continues on the plain-JSON-prompt path instead of crashing.
Or drop the `provider` argument entirely to let HF auto-route you to
whichever provider is available at that moment — since not all of them
honour `strict`, this is exactly why defensive parsing belongs in the
recipe, not as a "polish later" item.
