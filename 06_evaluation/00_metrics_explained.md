# Evaluation Metrics, Explained With Analogies

*A plain-English companion to this folder. No code, no jargon — just what
each metric is actually doing and why you'd reach for it. Read this first,
then look at `01_rouge.py` → `04_compare_models.py` for the code.*

## The problem these metrics solve

You asked an AI model to summarize an article, or translate a sentence, or
answer a question. You have the "correct" answer (written by a human) and
the AI's answer. Now: **how good is the AI's answer, as a number?**

You can't just check for an exact match — two people can say the same
correct thing in totally different words. You need a grading method. The
scripts in this folder are four different graders, each with a different
philosophy. Think of them as four teachers grading the same stack of essays,
each with their own grading style.

---

## Teacher #1: ROUGE, the Word-Counter

**Analogy:** Imagine grading a student's summary by laying it next to the
model answer and literally counting how many of the same words show up in
both. That's ROUGE. It doesn't read for understanding — it counts overlap.

It comes in three flavors, like three levels of strictness:

| Flavor | What it counts | Analogy |
|---|---|---|
| **ROUGE-1** | Matching single words, regardless of order | Two shopping lists — how many of the same items are on both, even if listed in a different order |
| **ROUGE-2** | Matching pairs of words that appear next to each other | Same shopping lists, but now you only get credit if two items were written next to each other in *both* lists, in the same order |
| **ROUGE-L** | The longest run of words that appear in the same order in both (skips allowed) | Comparing two people's step-by-step directions to a house — you get credit for the longest stretch where both sets of directions agree on the order of turns, even if one person added an extra turn in between |

**What it's good at:** cheap, fast, consistent. Great for a quick sanity
check.

**What it's blind to:** meaning. If the AI writes "NASA sent people to the
Moon" and the correct answer says "The United States space agency put
astronauts on the lunar surface" — same meaning, almost no shared words —
ROUGE scores it low. The Word-Counter teacher only cares about literal word
overlap, not whether the student actually understood the material.

---

## Teacher #2: BLEU, the Translation Checker

**Analogy:** BLEU was built specifically for grading translations. Picture
a strict language teacher checking a translated sentence phrase by phrase
against one or more "official" correct translations pinned to the wall
(often there's more than one right way to translate something, so the
teacher keeps a few acceptable answer keys).

Key ideas, in plain terms:

- **It checks phrases of increasing length** — single words, then two-word
  chunks, then three, then four — and blends the results. The more of
  these overlapping chunks that match, the higher the score.
- **The "brevity penalty":** a student can't game the system by writing one
  short, safe word and getting full marks for what little they wrote.
  Analogy: a food critic won't give a five-course-meal score to a dish that
  only has one ingredient, even if that one ingredient was correct.
  Short-but-technically-not-wrong answers get docked.
- **Multiple correct answers are allowed:** like a spelling bee accepting
  both "color" and "colour" as correct, BLEU checks the student's answer
  against *all* the acceptable answer keys and gives credit for matching
  any one of them.

**What it's good at:** the standard for grading machine translation, and
you'll see it quoted constantly in papers and leaderboards.

**What it's blind to:** exactly the same weakness as ROUGE — it's counting
matching words and phrases, not meaning. A perfect paraphrase using
different vocabulary scores just as badly as a wrong answer.

---

## Teacher #3: BERTScore, the Meaning-Understander

**Analogy:** This is the teacher who actually reads for understanding
instead of counting words. Instead of comparing the words themselves,
BERTScore converts every word into a "meaning fingerprint" (a list of
numbers that captures what the word means *in that sentence*) and compares
the fingerprints instead of the raw text.

Why that matters:

- **Synonyms are recognized.** "NASA" and "the United States space agency"
  produce very similar fingerprints, so a paraphrase scores highly — the
  exact problem ROUGE and BLEU can't solve.
- **Context changes the fingerprint.** The word "bank" gets a different
  fingerprint in "river bank" versus "bank account." A simple dictionary
  look-up would treat both as the same word; this teacher is smart enough
  to tell them apart based on the sentence around it.
- **It's slower and more expensive.** Understanding meaning takes more
  effort than counting words — literally, in this case: it needs a large
  pretrained language model to generate those fingerprints, so it's slower
  and needs a one-time model download compared to the other two.

**What it's good at:** catching correct answers that use different words
than the reference — the modern-LLM blind spot in ROUGE/BLEU.

**What it's blind to:** it's still a metric, not a human. It can be fooled
by fluent-sounding nonsense and doesn't replace an actual read-through for
anything high-stakes.

---

## The three numbers every grader reports: Precision, Recall, F1

All three teachers above report their scores using the same three numbers.
Here's the plain-English version, using a fishing analogy:

- **Precision** — "Of everything you caught, how much was actually a fish
  (and not an old boot)?" In grading terms: of everything the AI's answer
  said, how much of it was actually correct / present in the reference?
- **Recall** — "Of all the fish that were actually in the pond, how many
  did you catch?" In grading terms: of everything the *correct* answer
  says, how much did the AI's answer manage to include?
- **F1** — A single number that balances the two, so you can't win by only
  being good at one. An answer that's short but 100% correct (high
  precision, low recall) and an answer that says everything but pads it
  with junk (high recall, low precision) both get punished. F1 rewards
  hitting both at once — this is usually the one number you report.

---

## Putting them side by side

| | ROUGE | BLEU | BERTScore |
|---|---|---|---|
| **Grading style** | Counts matching words/phrases | Counts matching phrases + penalizes short answers | Compares *meaning*, not exact words |
| **Understands paraphrases?** | No | No | Yes |
| **Speed / cost** | Very fast, free | Very fast, free | Slower, needs a downloaded model |
| **Typical use** | Summarization | Translation | Anything where wording can legitimately vary |
| **Biggest weakness** | Blind to meaning | Blind to meaning | Still just a metric, not human judgement |

**Rule of thumb:** start with ROUGE or BLEU because they're free and fast.
The moment you suspect the model is being marked down for saying the right
thing in different words, bring in BERTScore to check.

---

## The fourth file: comparing whole models, not just one answer

`04_compare_models.py` isn't a new grading style — it's a recipe for using
these graders responsibly when you're choosing between AI models.

**Analogy:** think of it as a cooking contest. Every contestant (AI model)
gets the *exact same* recipe card (the same prompt) — that's the only fair
way to compare them. A judge tastes and scores each dish against a "perfect
dish" cooked by a human expert (the reference answer). Crucially, the
contest organizer writes down every dish's score *and keeps a photo of the
dish itself* — because a number alone can be misleading. A contestant can
score well on paper (the plate looks right) while getting the actual
cooking wrong, so someone still has to taste it before declaring a winner.

That's the discipline this script encodes: same prompt for every model,
score with a metric, but always read the actual generated text before
trusting the number.

---

## Where to go next

- `01_rouge.py` / `01_rouge.md` — ROUGE in code
- `02_bleu.py` / `02_bleu.md` — BLEU in code
- `03_bertscore.py` / `03_bertscore.md` — BERTScore in code
- `04_compare_models.py` / `04_compare_models.md` — the full comparison pipeline
