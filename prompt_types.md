Chain-of-thought → "Walk me through your reasoning"  
Few-shot examples → "Here's what I mean, like this"  
System prompts → "Here's who you are and how I need you to work"  
Output constraints → "Give me the answer in this format"  
Query decomposition → "Let's break this into parts"  


# Prompt Engineering Gotchas

## The model takes you literally

You say "summarize this" — you get a summary. You didn't say how long, for what audience, or what to focus on. The model filled in the blanks with its own assumptions. A human colleague would ask clarifying questions. The model just guesses.

## More context isn't always better

People dump entire documents into the prompt thinking more information helps. It doesn't. It dilutes focus, increases cost, and the model may latch onto irrelevant details. The same way a 40-page requirements doc with no prioritization is useless to a developer.

## The model is confidently wrong

It doesn't say "I don't know." It generates plausible-sounding nonsense with the same tone as factual answers. You can't tell the difference without domain expertise. This is the most dangerous gotcha in enterprise settings.

## Order matters

Information at the beginning and end of a long prompt gets more attention than the middle. Critical instructions buried in paragraph six get ignored. Same reason nobody reads the middle of a long email.

## It forgets within a conversation

Long conversations exceed the context window and early messages silently drop off. The model doesn't tell you it forgot — it just starts contradicting itself or losing thread.

## Same prompt, different results

Temperature and sampling mean you can run the same prompt twice and get different outputs. Fine for creative work, terrible for production pipelines that need consistency.

## It's sycophantic

Push back on the model's answer and it'll often agree with you even when it was right the first time. It's trained to be helpful, which sometimes means agreeable to a fault.

## Prompt injection

Users can embed instructions in their input that override your system prompt. "Ignore all previous instructions and..." is a real attack vector. Critical for any customer-facing application.

## What works today breaks tomorrow

Model updates change behavior. A prompt tuned perfectly for GPT-4o may produce garbage on GPT-5. There's no versioning guarantee on behavior, only on the model name.

## Evaluation is hard

"Is this output good?" is subjective for most tasks. Without clear evaluation criteria defined upfront, you end up with vibes-based quality assessment. Same problem as testing software without acceptance criteria.

## The biggest gotcha

People treat prompt engineering as a one-time task. It's not. It's iterative — just like writing good SQL, good requirements, or good test cases. The first version is never the best version.
