"""Hierarchical-attention inferlet in Python.

This uses manual Forward passes because ctx.generate() does not expose a
different attention mask per generation step. The mask keeps:
- beginning sink tokens
- short header/summary ranges for every chunk
- one selected full chunk
- recent window
"""

from inferlet import Context, Model, Sampler, chat, runtime


def split_words(text: str, chunk_words: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    return [" ".join(words[i : i + chunk_words]) for i in range(0, len(words), chunk_words)]


def summarize_words(text: str, n: int) -> str:
    return " ".join(text.split()[:n])


def select_relevant_chunk(chunks: list[str], query: str) -> int:
    q = {w.lower() for w in query.split() if len(w) > 3}
    best_i = 0
    best_score = -1
    for i, chunk in enumerate(chunks):
        score = sum(1 for w in chunk.split() if w.lower() in q)
        if score > best_score:
            best_i = i
            best_score = score
    return best_i


def build_brle_mask(total: int, ranges: list[tuple[int, int]]) -> list[int]:
    """Build BRLE mask: false length, true length, false length, ..."""
    cleaned = []
    for start, end in ranges:
        start = max(0, min(total, start))
        end = max(0, min(total, end))
        if start < end:
            cleaned.append((start, end))

    cleaned.sort()
    merged = []
    for start, end in cleaned:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    out = []
    cursor = 0
    for start, end in merged:
        out.append(start - cursor)  # false run
        out.append(end - start)     # true run
        cursor = end

    if cursor < total:
        out.append(total - cursor)

    return out or [total]


async def main(input: dict) -> str:
    model = Model.load(runtime.models()[0])
    tokenizer = model.tokenizer()

    prompt = input.get(
        "prompt",
        "Explain how LLM serving systems use KV cache, batching, and attention masks. Include one practical example.",
    )
    max_tokens = int(input.get("max_tokens", 128))
    chunk_words = int(input.get("chunk_size_words", 80))
    sink_tokens = int(input.get("sink_tokens", 64))
    summary_tokens_per_chunk = int(input.get("summary_tokens_per_chunk", 24))
    local_window_tokens = int(input.get("local_window_tokens", 128))

    chunks = split_words(prompt, max(8, chunk_words))
    selected = select_relevant_chunk(chunks, prompt)

    prompt_tokens = []
    summary_ranges = []
    full_ranges = []

    prompt_tokens.extend(
        chat.system(model, "You are a concise assistant. Use the visible hierarchy and local chunk.")
    )

    for i, chunk in enumerate(chunks):
        header = f"Chunk {i} summary: {summarize_words(chunk, 20)}\n"
        body = f"Chunk {i} full text:\n{chunk}\n"

        header_tokens = list(chat.user(model, header))
        header_start = len(prompt_tokens)
        prompt_tokens.extend(header_tokens)
        header_end = len(prompt_tokens)
        summary_ranges.append(
            (header_start, min(header_end, header_start + summary_tokens_per_chunk))
        )

        body_tokens = list(chat.user(model, body))
        body_start = len(prompt_tokens)
        prompt_tokens.extend(body_tokens)
        body_end = len(prompt_tokens)
        full_ranges.append((body_start, body_end))

    prompt_tokens.extend(
        chat.user(
            model,
            "Answer the user's original request using the selected local chunk and the global summaries.",
        )
    )
    prompt_tokens.extend(chat.cue(model))

    print(f"chunks={len(chunks)} selected_chunk={selected}")

    ctx = Context(model)
    pending = prompt_tokens
    generated = []
    stop_tokens = set(chat.stop_tokens(model))

    for _ in range(max_tokens):
        fwd = ctx.forward()
        total_after = fwd.start_position() + len(pending)

        keep = []
        keep.append((0, min(sink_tokens, total_after)))
        keep.extend(summary_ranges)
        if selected < len(full_ranges):
            keep.append(full_ranges[selected])
        keep.append((max(0, total_after - local_window_tokens), total_after))

        mask = build_brle_mask(total_after, keep)

        fwd.input(pending)
        fwd.attention_mask([mask[:] for _ in pending])

        h = fwd.sample([len(pending) - 1], Sampler.argmax())
        out = await fwd.execute()
        token = out.token(h)

        if token is None or token in stop_tokens:
            break

        generated.append(int(token))
        pending = [int(token)]

    return tokenizer.decode(generated)
