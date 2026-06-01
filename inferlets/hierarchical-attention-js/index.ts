// Hierarchical-attention inferlet in TypeScript.
//
// During generation, the attention mask keeps:
//   1. beginning sink tokens
//   2. short header/summary ranges for every chunk
//   3. one selected full chunk
//   4. recent sliding window
//
// This is an MVP that proves the programmable attention-mask path.

import { Context, Model, Sampler, chat, runtime } from 'inferlet';

interface Input {
    prompt?: string;
    max_tokens?: number;
    chunk_size_words?: number;
    sink_tokens?: number;
    summary_tokens_per_chunk?: number;
    local_window_tokens?: number;
}

interface Range {
    start: number;
    end: number;
}

function splitWords(text: string, chunkWords: number): string[] {
    const words = text.split(/\s+/).filter(Boolean);
    if (words.length === 0) return [''];
    const chunks: string[] = [];
    for (let i = 0; i < words.length; i += chunkWords) {
        chunks.push(words.slice(i, i + chunkWords).join(' '));
    }
    return chunks;
}

function summarizeWords(text: string, n: number): string {
    return text.split(/\s+/).filter(Boolean).slice(0, n).join(' ');
}

function selectRelevantChunk(chunks: string[], query: string): number {
    const q = new Set(
        query
            .split(/\s+/)
            .map((w) => w.toLowerCase())
            .filter((w) => w.length > 3),
    );

    let best = 0;
    let bestScore = -1;

    chunks.forEach((chunk, i) => {
        let score = 0;
        for (const w of chunk.split(/\s+/)) {
            if (q.has(w.toLowerCase())) score++;
        }
        if (score > bestScore) {
            best = i;
            bestScore = score;
        }
    });

    return best;
}

function buildBrleMask(total: number, ranges: Range[]): number[] {
    const cleaned = ranges
        .map((r) => ({
            start: Math.max(0, Math.min(total, r.start)),
            end: Math.max(0, Math.min(total, r.end)),
        }))
        .filter((r) => r.start < r.end)
        .sort((a, b) => a.start - b.start);

    const merged: Range[] = [];
    for (const r of cleaned) {
        const last = merged[merged.length - 1];
        if (last && r.start <= last.end) {
            last.end = Math.max(last.end, r.end);
        } else {
            merged.push({ ...r });
        }
    }

    const out: number[] = [];
    let cursor = 0;

    for (const r of merged) {
        out.push(r.start - cursor); // false run
        out.push(r.end - r.start);  // true run
        cursor = r.end;
    }

    if (cursor < total) out.push(total - cursor);
    return out.length ? out : [total];
}

export async function main(input: Input): Promise<string> {
    const model = Model.load(runtime.models()[0]);
    const tokenizer = model.tokenizer();

    const prompt =
        input.prompt ??
        'Explain how LLM serving systems use KV cache, batching, and attention masks. Include one practical example.';

    const maxTokens = input.max_tokens ?? 128;
    const chunkWords = Math.max(8, input.chunk_size_words ?? 80);
    const sinkTokens = input.sink_tokens ?? 64;
    const summaryTokensPerChunk = input.summary_tokens_per_chunk ?? 24;
    const localWindowTokens = input.local_window_tokens ?? 128;

    const chunks = splitWords(prompt, chunkWords);
    const selected = selectRelevantChunk(chunks, prompt);

    const promptTokens: number[] = [];
    const summaryRanges: Range[] = [];
    const fullRanges: Range[] = [];

    promptTokens.push(
        ...Array.from(
            chat.system(
                model,
                'You are a concise assistant. Use the visible hierarchy and local chunk.',
            ),
        ),
    );

    chunks.forEach((chunk, i) => {
        const header = `Chunk ${i} summary: ${summarizeWords(chunk, 20)}\n`;
        const body = `Chunk ${i} full text:\n${chunk}\n`;

        const headerTokens = Array.from(chat.user(model, header));
        const headerStart = promptTokens.length;
        promptTokens.push(...headerTokens);
        const headerEnd = promptTokens.length;

        summaryRanges.push({
            start: headerStart,
            end: Math.min(headerEnd, headerStart + summaryTokensPerChunk),
        });

        const bodyTokens = Array.from(chat.user(model, body));
        const bodyStart = promptTokens.length;
        promptTokens.push(...bodyTokens);
        const bodyEnd = promptTokens.length;

        fullRanges.push({ start: bodyStart, end: bodyEnd });
    });

    promptTokens.push(
        ...Array.from(
            chat.user(
                model,
                "Answer the user's original request using the selected local chunk and the global summaries.",
            ),
        ),
    );
    promptTokens.push(...Array.from(chat.cue(model)));

    console.log(`chunks=${chunks.length} selected_chunk=${selected}`);

    const ctx = new Context(model);
    let pending = promptTokens;
    const generated: number[] = [];
    const stopTokens = new Set<number>(Array.from(chat.stopTokens(model)));

    for (let i = 0; i < maxTokens; ++i) {
        const fwd = ctx.forward();
        const totalAfter = fwd.startPosition() + pending.length;

        const keep: Range[] = [];
        keep.push({ start: 0, end: Math.min(sinkTokens, totalAfter) });
        keep.push(...summaryRanges);
        if (selected < fullRanges.length) keep.push(fullRanges[selected]);
        keep.push({
            start: Math.max(0, totalAfter - localWindowTokens),
            end: totalAfter,
        });

        const mask = buildBrleMask(totalAfter, keep);

        fwd.input(new Uint32Array(pending));
        fwd.attentionMask(pending.map(() => new Uint32Array(mask)));

        const h = fwd.sample([pending.length - 1], Sampler.argmax());
        const out = await fwd.execute();
        const token = out.token(h);

        if (token === undefined || stopTokens.has(token)) break;

        generated.push(token);
        pending = [token];
    }

    return tokenizer.decode(new Uint32Array(generated));
}
