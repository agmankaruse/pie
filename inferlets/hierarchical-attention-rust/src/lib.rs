//! Hierarchical-attention inferlet.
//!
//! This demonstrates a missing paper-style optimization:
//! custom attention masks that keep a compressed hierarchy visible.
//!
//! The prompt is split into chunks. During generation, each token can attend to:
//!   1. the beginning sink tokens
//!   2. short header/summary ranges for every chunk
//!   3. one selected full chunk
//!   4. a recent sliding window
//!
//! This is an MVP: summaries are chunk headers, relevance is lexical overlap.
//! The point is to demonstrate Pie's programmable attention-mask interface.

use inferlet::{Context, Result, model::Model, runtime, sample::Sampler};
use serde::Deserialize;
use std::collections::HashSet;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_chunk_words")]
    chunk_size_words: usize,
    #[serde(default = "default_sink")]
    sink_tokens: u32,
    #[serde(default = "default_summary")]
    summary_tokens_per_chunk: u32,
    #[serde(default = "default_window")]
    local_window_tokens: u32,
}

#[derive(Clone, Copy, Debug)]
struct Range {
    start: u32,
    end: u32,
}

fn default_prompt() -> String {
    "Explain how LLM serving systems use KV cache, batching, and attention masks. Include one practical example.".into()
}
fn default_max_tokens() -> usize { 128 }
fn default_chunk_words() -> usize { 80 }
fn default_sink() -> u32 { 64 }
fn default_summary() -> u32 { 24 }
fn default_window() -> u32 { 128 }

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let model = Model::load(runtime::models().first().ok_or("No models available")?)?;
    let stop_tokens = inferlet::chat::stop_tokens(&model);

    let chunks = split_words(&input.prompt, input.chunk_size_words.max(8));
    let selected_chunk = select_relevant_chunk(&chunks, &input.prompt);

    // Build one prompt token stream and remember ranges for chunk headers/full text.
    let mut prompt_tokens = Vec::new();
    let mut summary_ranges = Vec::new();
    let mut full_ranges = Vec::new();

    prompt_tokens.extend(inferlet::chat::system(
        &model,
        "You are a concise assistant. Use the visible hierarchy and local chunk.",
    ));

    for (i, chunk) in chunks.iter().enumerate() {
        let header = format!("Chunk {} summary: {}\n", i, summarize_words(chunk, 20));
        let body = format!("Chunk {} full text:\n{}\n", i, chunk);

        let header_tokens = inferlet::chat::user(&model, &header);
        let header_start = prompt_tokens.len() as u32;
        prompt_tokens.extend(header_tokens);
        let header_end = prompt_tokens.len() as u32;

        // Keep only the first N tokens of each header as global summaries.
        summary_ranges.push(Range {
            start: header_start,
            end: header_start + input.summary_tokens_per_chunk.min(header_end - header_start),
        });

        let body_tokens = inferlet::chat::user(&model, &body);
        let body_start = prompt_tokens.len() as u32;
        prompt_tokens.extend(body_tokens);
        let body_end = prompt_tokens.len() as u32;

        full_ranges.push(Range { start: body_start, end: body_end });
    }

    prompt_tokens.extend(inferlet::chat::user(
        &model,
        "Answer the user's original request using the selected local chunk and the global summaries.",
    ));
    prompt_tokens.extend(inferlet::chat::cue(&model));

    println!("chunks={} selected_chunk={}", chunks.len(), selected_chunk);

    let mut ctx = Context::new(&model)?;
    let mut pending = prompt_tokens;
    let mut generated = Vec::new();

    for _ in 0..input.max_tokens {
        let mut fwd = ctx.forward();
        let total_seq_after = fwd.start_position() + pending.len() as u32;

        let mut keep = Vec::new();

        // 1. Sink: beginning of prompt remains globally visible.
        keep.push(Range { start: 0, end: input.sink_tokens.min(total_seq_after) });

        // 2. Summary/header tokens from every chunk.
        keep.extend(summary_ranges.iter().copied());

        // 3. Full selected local chunk.
        if let Some(r) = full_ranges.get(selected_chunk) {
            keep.push(*r);
        }

        // 4. Recent window over committed/generated sequence.
        let win_start = total_seq_after.saturating_sub(input.local_window_tokens);
        keep.push(Range { start: win_start, end: total_seq_after });

        let mask = build_brle_mask(total_seq_after, &keep);
        fwd.input(&pending);

        // Use one shared mask per query token in this pass.
        let masks: Vec<Vec<u32>> = (0..pending.len()).map(|_| mask.clone()).collect();
        fwd.attention_mask(&masks);

        let h = fwd.sample(&[(pending.len() - 1) as u32], Sampler::Argmax);
        let out = fwd.execute().await?;
        let token = out.token(h).ok_or("no sampled token")?;

        if stop_tokens.contains(&token) {
            break;
        }

        generated.push(token);
        pending = vec![token];
    }

    Ok(model.tokenizer().decode(&generated)?)
}

fn split_words(text: &str, chunk_words: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() { return vec![String::new()]; }

    words.chunks(chunk_words)
        .map(|w| w.join(" "))
        .collect()
}

fn summarize_words(text: &str, n: usize) -> String {
    text.split_whitespace()
        .take(n)
        .collect::<Vec<_>>()
        .join(" ")
}

fn select_relevant_chunk(chunks: &[String], query: &str) -> usize {
    let q: HashSet<String> = query
        .split_whitespace()
        .map(|w| w.to_lowercase())
        .filter(|w| w.len() > 3)
        .collect();

    chunks
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let score = c
                .split_whitespace()
                .map(|w| w.to_lowercase())
                .filter(|w| q.contains(w))
                .count();
            (i, score)
        })
        .max_by_key(|(_, score)| *score)
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn build_brle_mask(total: u32, ranges: &[Range]) -> Vec<u32> {
    // BRLE starts with a false-run length, then true-run length, alternating.
    // [0, total] means "all true".
    let mut ranges = ranges.to_vec();
    ranges.sort_by_key(|r| r.start);

    let mut merged: Vec<Range> = Vec::new();
    for mut r in ranges {
        r.start = r.start.min(total);
        r.end = r.end.min(total);
        if r.start >= r.end { continue; }

        if let Some(last) = merged.last_mut() {
            if r.start <= last.end {
                last.end = last.end.max(r.end);
            } else {
                merged.push(r);
            }
        } else {
            merged.push(r);
        }
    }

    let mut out = Vec::new();
    let mut cursor = 0u32;

    for r in merged {
        out.push(r.start.saturating_sub(cursor)); // false
        out.push(r.end - r.start);                // true
        cursor = r.end;
    }

    if cursor < total {
        out.push(total - cursor); // final false
    }

    if out.is_empty() {
        vec![total]
    } else {
        out
    }
}
