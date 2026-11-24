use serde_json::{json, Value};

use crate::ir::{CanonicalStreamEvent, ToolChoice, ToolDefinition};

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
const GEMMA4_THINK_OPEN: &str = "<|channel>thought";
const GEMMA4_THINK_CLOSE: &str = "<channel|>";
const TOOL_OPEN: &str = "<tool_call>";
const TOOL_CLOSE: &str = "</tool_call>";
const FUNC_OPEN: &str = "<function=";
const FUNC_CLOSE: &str = "</function>";
const MISTRAL_TOOLS_OPEN: &str = "[TOOL_CALLS]";

/// Format tool definitions into a deterministic system prompt block for native GGUF models
/// (compatible with Qwen / Hermes / Llama / Gemma instruction-tuned models).
pub fn format_tools_system_block(
    tools: &[ToolDefinition],
    choice: Option<&ToolChoice>,
) -> Option<String> {
    if tools.is_empty() || matches!(choice, Some(ToolChoice::None)) {
        return None;
    }

    let mut defs = Vec::with_capacity(tools.len());
    for t in tools {
        defs.push(json!({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description.as_deref().unwrap_or(""),
                "parameters": t.input_schema,
            }
        }));
    }

    let defs_lines = defs
        .iter()
        .filter_map(|d| serde_json::to_string(d).ok())
        .collect::<Vec<_>>()
        .join("\n");

    let policy = match choice {
        Some(ToolChoice::Required) => {
            "You MUST call at least one function from the tools list above."
        }
        Some(ToolChoice::Named { name }) => {
            return Some(format!(
                "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
                 You are provided with function signatures within <tools></tools> XML tags:\n\
                 <tools>\n{defs_lines}\n</tools>\n\n\
                 You MUST call the function `{name}` using <tool_call></tool_call> XML tags:\n\
                 <tool_call>\n{{\"name\": \"{name}\", \"arguments\": <args-json-object>}}\n</tool_call>"
            ));
        }
        _ => "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:",
    };

    Some(format!(
        "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
         You are provided with function signatures within <tools></tools> XML tags:\n\
         <tools>\n{defs_lines}\n</tools>\n\n\
         {policy}\n\
         <tool_call>\n{{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n</tool_call>"
    ))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParseState {
    Normal,
    InThink,
    InToolCall,
    InFuncCall,
    InMistralTools,
}

/// Incremental streaming parser that separates `<think>...</think>` into `ThinkingDelta`,
/// `<tool_call>...</tool_call>` into `ToolCallStart` + `ToolCallDelta`, and everything else into `TextDelta`.
pub struct NativeOutputParser {
    state: ParseState,
    buf: String,
    tool_buf: String,
    tool_count: usize,
}

impl NativeOutputParser {
    pub fn new() -> Self {
        Self {
            state: ParseState::Normal,
            buf: String::new(),
            tool_buf: String::new(),
            tool_count: 0,
        }
    }

    pub fn had_tool_calls(&self) -> bool {
        self.tool_count > 0
    }

    pub fn push(&mut self, chunk: &str) -> Vec<CanonicalStreamEvent> {
        self.buf.push_str(chunk);
        let mut events = Vec::new();

        loop {
            match self.state {
                ParseState::Normal => {
                    let next_think = self.buf.find(THINK_OPEN);
                    let next_gemma_think = self.buf.find(GEMMA4_THINK_OPEN);
                    let next_tool = self.buf.find(TOOL_OPEN);
                    let next_func = self.buf.find(FUNC_OPEN);
                    let next_mistral = self.buf.find(MISTRAL_TOOLS_OPEN);

                    // Pick earliest opening tag among <think>, <|channel>thought, <tool_call>, <function=, and [TOOL_CALLS]
                    let earliest = [
                        next_think.map(|i| (i, 0u8)),
                        next_tool.map(|i| (i, 1u8)),
                        next_func.map(|i| (i, 2u8)),
                        next_mistral.map(|i| (i, 3u8)),
                        next_gemma_think.map(|i| (i, 4u8)),
                    ]
                    .into_iter()
                    .flatten()
                    .min_by_key(|(i, _)| *i);

                    match earliest {
                        Some((ti, 0)) => {
                            let before = self.buf[..ti].to_string();
                            self.buf.drain(..ti + THINK_OPEN.len());
                            if !before.is_empty() {
                                events.push(CanonicalStreamEvent::TextDelta(before));
                            }
                            self.state = ParseState::InThink;
                        }
                        Some((gi, 4)) => {
                            let before = self.buf[..gi].to_string();
                            self.buf.drain(..gi + GEMMA4_THINK_OPEN.len());
                            if !before.is_empty() {
                                events.push(CanonicalStreamEvent::TextDelta(before));
                            }
                            self.state = ParseState::InThink;
                        }
                        Some((ci, 1)) => {
                            let before = self.buf[..ci].to_string();
                            self.buf.drain(..ci + TOOL_OPEN.len());
                            if !before.is_empty() {
                                events.push(CanonicalStreamEvent::TextDelta(before));
                            }
                            self.tool_buf.clear();
                            self.state = ParseState::InToolCall;
                        }
                        Some((fi, 2)) => {
                            let before = self.buf[..fi].to_string();
                            self.buf.drain(..fi + FUNC_OPEN.len());
                            if !before.is_empty() {
                                events.push(CanonicalStreamEvent::TextDelta(before));
                            }
                            self.tool_buf.clear();
                            self.state = ParseState::InFuncCall;
                        }
                        Some((mi, 3)) => {
                            let before = self.buf[..mi].to_string();
                            self.buf.drain(..mi + MISTRAL_TOOLS_OPEN.len());
                            if !before.is_empty() {
                                events.push(CanonicalStreamEvent::TextDelta(before));
                            }
                            self.tool_buf.clear();
                            self.state = ParseState::InMistralTools;
                        }
                        _ => {
                            let keep = longest_tag_prefix(
                                &self.buf,
                                &[
                                    THINK_OPEN,
                                    GEMMA4_THINK_OPEN,
                                    TOOL_OPEN,
                                    FUNC_OPEN,
                                    MISTRAL_TOOLS_OPEN,
                                ],
                            );
                            let emit_len = self.buf.len().saturating_sub(keep);
                            if emit_len > 0 {
                                let out = self.buf[..emit_len].to_string();
                                self.buf.drain(..emit_len);
                                events.push(CanonicalStreamEvent::TextDelta(out));
                            }
                            break;
                        }
                    }
                }
                ParseState::InThink => {
                    // Recover cleanly if a thinking model emits `<tool_call>` or `<function=`
                    // while forgetting to close `</think>` / `<channel|>` first.
                    let next_close = self.buf.find(THINK_CLOSE);
                    let next_gemma_close = self.buf.find(GEMMA4_THINK_CLOSE);
                    let next_tool = self.buf.find(TOOL_OPEN);
                    let next_func = self.buf.find(FUNC_OPEN);
                    let earliest = [
                        next_close.map(|i| (i, 0u8)),
                        next_tool.map(|i| (i, 1u8)),
                        next_func.map(|i| (i, 2u8)),
                        next_gemma_close.map(|i| (i, 3u8)),
                    ]
                    .into_iter()
                    .flatten()
                    .min_by_key(|(i, _)| *i);

                    match earliest {
                        Some((end, 0)) => {
                            let think_txt = self.buf[..end].to_string();
                            self.buf.drain(..end + THINK_CLOSE.len());
                            if !think_txt.is_empty() {
                                events.push(CanonicalStreamEvent::ThinkingDelta(think_txt));
                            }
                            self.state = ParseState::Normal;
                        }
                        Some((end, 3)) => {
                            let think_txt = self.buf[..end].to_string();
                            self.buf.drain(..end + GEMMA4_THINK_CLOSE.len());
                            if !think_txt.is_empty() {
                                events.push(CanonicalStreamEvent::ThinkingDelta(think_txt));
                            }
                            self.state = ParseState::Normal;
                        }
                        Some((ci, 1)) => {
                            let think_txt = self.buf[..ci].to_string();
                            self.buf.drain(..ci + TOOL_OPEN.len());
                            if !think_txt.is_empty() {
                                events.push(CanonicalStreamEvent::ThinkingDelta(think_txt));
                            }
                            self.tool_buf.clear();
                            self.state = ParseState::InToolCall;
                        }
                        Some((fi, 2)) => {
                            let think_txt = self.buf[..fi].to_string();
                            self.buf.drain(..fi + FUNC_OPEN.len());
                            if !think_txt.is_empty() {
                                events.push(CanonicalStreamEvent::ThinkingDelta(think_txt));
                            }
                            self.tool_buf.clear();
                            self.state = ParseState::InFuncCall;
                        }
                        _ => {
                            let keep = longest_tag_prefix(
                                &self.buf,
                                &[THINK_CLOSE, GEMMA4_THINK_CLOSE, TOOL_OPEN, FUNC_OPEN],
                            );
                            let emit_len = self.buf.len().saturating_sub(keep);
                            if emit_len > 0 {
                                let out = self.buf[..emit_len].to_string();
                                self.buf.drain(..emit_len);
                                events.push(CanonicalStreamEvent::ThinkingDelta(out));
                            }
                            break;
                        }
                    }
                }
                ParseState::InToolCall => {
                    if let Some(end) = self.buf.find(TOOL_CLOSE) {
                        self.tool_buf.push_str(&self.buf[..end]);
                        self.buf.drain(..end + TOOL_CLOSE.len());
                        self.emit_tool_call(&mut events);
                        self.state = ParseState::Normal;
                    } else {
                        let keep = longest_tag_prefix(&self.buf, &[TOOL_CLOSE]);
                        let take_len = self.buf.len().saturating_sub(keep);
                        if take_len > 0 {
                            self.tool_buf.push_str(&self.buf[..take_len]);
                            self.buf.drain(..take_len);
                        }
                        break;
                    }
                }
                ParseState::InFuncCall => {
                    if let Some(end) = self.buf.find(FUNC_CLOSE) {
                        self.tool_buf.push_str(&self.buf[..end]);
                        self.buf.drain(..end + FUNC_CLOSE.len());
                        self.emit_func_call(&mut events);
                        self.state = ParseState::Normal;
                    } else {
                        let keep = longest_tag_prefix(&self.buf, &[FUNC_CLOSE]);
                        let take_len = self.buf.len().saturating_sub(keep);
                        if take_len > 0 {
                            self.tool_buf.push_str(&self.buf[..take_len]);
                            self.buf.drain(..take_len);
                        }
                        break;
                    }
                }
                ParseState::InMistralTools => {
                    self.tool_buf.push_str(&self.buf);
                    self.buf.clear();
                    break;
                }
            }
        }

        events
    }

    pub fn finish(&mut self) -> Vec<CanonicalStreamEvent> {
        let mut events = Vec::new();
        match self.state {
            ParseState::Normal => {
                if !self.buf.is_empty() {
                    events.push(CanonicalStreamEvent::TextDelta(std::mem::take(&mut self.buf)));
                }
            }
            ParseState::InThink => {
                if !self.buf.is_empty() {
                    events.push(CanonicalStreamEvent::ThinkingDelta(std::mem::take(
                        &mut self.buf,
                    )));
                }
            }
            ParseState::InToolCall => {
                self.tool_buf.push_str(&self.buf);
                self.buf.clear();
                self.emit_tool_call(&mut events);
            }
            ParseState::InFuncCall => {
                self.tool_buf.push_str(&self.buf);
                self.buf.clear();
                self.emit_func_call(&mut events);
            }
            ParseState::InMistralTools => {
                self.tool_buf.push_str(&self.buf);
                self.buf.clear();
                self.emit_mistral_tools(&mut events);
            }
        }
        events
    }

    fn emit_tool_call(&mut self, events: &mut Vec<CanonicalStreamEvent>) {
        let raw = self.tool_buf.trim().to_string();
        self.tool_buf.clear();
        if raw.is_empty() {
            return;
        }
        // Handle nested `<tool_call><function=NAME>{...}</function></tool_call>` (MiMo-V2.6 / Qwen3.5)
        if let Some(rest) = raw.strip_prefix(FUNC_OPEN) {
            let inner = rest.strip_suffix(FUNC_CLOSE).unwrap_or(rest).trim();
            self.tool_buf = inner.to_string();
            self.emit_func_call(events);
            return;
        }
        if let Some(parsed) = parse_json_lenient(&raw) {
            if let Some(name) = parsed.get("name").and_then(Value::as_str) {
                let idx = self.tool_count;
                self.tool_count += 1;
                let id = format!("call_native_{}", idx + 1);
                let args = parsed
                    .get("arguments")
                    .or_else(|| parsed.get("input"))
                    .cloned()
                    .unwrap_or_else(|| json!({}));
                let args_str = serde_json::to_string(&args).unwrap_or_else(|_| "{}".to_string());
                events.push(CanonicalStreamEvent::ToolCallStart {
                    index: idx,
                    id,
                    name: name.to_string(),
                });
                events.push(CanonicalStreamEvent::ToolCallDelta {
                    index: idx,
                    arguments_delta: args_str,
                });
                return;
            }
        }
        // Fallback: if invalid JSON inside <tool_call>, return raw text rather than dropping it
        events.push(CanonicalStreamEvent::TextDelta(raw));
    }

    /// Parse `<function=tool_name>{...json args...}</function>` OR
    /// `<function=tool_name><parameter=key>val</parameter>...</function>`
    /// formats used by MiMo-V2.6, Qwen3.5, and Qwen3-Coder.
    fn emit_func_call(&mut self, events: &mut Vec<CanonicalStreamEvent>) {
        let raw = self.tool_buf.trim().to_string();
        self.tool_buf.clear();
        if let Some((name_part, args_part)) = raw.split_once('>') {
            let name = name_part.trim();
            let args_raw = args_part.trim();
            if !name.is_empty() {
                let args_val = parse_json_lenient(args_raw)
                    .or_else(|| parse_xml_parameters(args_raw));
                if let Some(args_val) = args_val {
                    let idx = self.tool_count;
                    self.tool_count += 1;
                    let id = format!("call_native_{}", idx + 1);
                    let args_str =
                        serde_json::to_string(&args_val).unwrap_or_else(|_| "{}".to_string());
                    events.push(CanonicalStreamEvent::ToolCallStart {
                        index: idx,
                        id,
                        name: name.to_string(),
                    });
                    events.push(CanonicalStreamEvent::ToolCallDelta {
                        index: idx,
                        arguments_delta: args_str,
                    });
                    return;
                }
            }
        }
        if !raw.is_empty() {
            events.push(CanonicalStreamEvent::TextDelta(format!("<function={raw}")));
        }
    }

    /// Parse Mistral/Devstral `[TOOL_CALLS] [{"name":"...","arguments":{...}}]` or
    /// `[TOOL_CALLS] func_name[ARGS]{...}` format.
    fn emit_mistral_tools(&mut self, events: &mut Vec<CanonicalStreamEvent>) {
        let raw = self.tool_buf.trim().to_string();
        self.tool_buf.clear();
        if raw.is_empty() {
            return;
        }
        // Format A: JSON array `[{"name": "...", "arguments": {...}}]` or single object `{...}`
        if let Ok(parsed) = serde_json::from_str::<Value>(&raw) {
            let items: Vec<&Value> = match &parsed {
                Value::Array(arr) => arr.iter().collect(),
                Value::Object(_) => vec![&parsed],
                _ => Vec::new(),
            };
            let mut emitted = false;
            for item in items {
                if let Some(name) = item.get("name").and_then(Value::as_str) {
                    let idx = self.tool_count;
                    self.tool_count += 1;
                    let id = format!("call_native_{}", idx + 1);
                    let args = item
                        .get("arguments")
                        .or_else(|| item.get("parameters"))
                        .cloned()
                        .unwrap_or_else(|| json!({}));
                    let args_str =
                        serde_json::to_string(&args).unwrap_or_else(|_| "{}".to_string());
                    events.push(CanonicalStreamEvent::ToolCallStart {
                        index: idx,
                        id,
                        name: name.to_string(),
                    });
                    events.push(CanonicalStreamEvent::ToolCallDelta {
                        index: idx,
                        arguments_delta: args_str,
                    });
                    emitted = true;
                }
            }
            if emitted {
                return;
            }
        }
        // Format B (Mistral v3 / Ministral-3): one or more `func_name[ARGS]{"key":"val"}`
        // segments separated by `[TOOL_CALLS]` (also strips duplicate leading `[TOOL_CALLS]`).
        let mut emitted_any = false;
        for segment in raw.split(MISTRAL_TOOLS_OPEN) {
            let seg = segment.trim();
            if seg.is_empty() {
                continue;
            }
            if let Some((name_part, args_part)) = seg.split_once("[ARGS]") {
                let name = name_part.trim().trim_start_matches(MISTRAL_TOOLS_OPEN).trim();
                let args_raw = args_part.trim();
                if !name.is_empty() {
                    if let Some(args_val) = parse_json_lenient(args_raw) {
                        let idx = self.tool_count;
                        self.tool_count += 1;
                        let id = format!("call_native_{}", idx + 1);
                        let args_str =
                            serde_json::to_string(&args_val).unwrap_or_else(|_| "{}".to_string());
                        events.push(CanonicalStreamEvent::ToolCallStart {
                            index: idx,
                            id,
                            name: name.to_string(),
                        });
                        events.push(CanonicalStreamEvent::ToolCallDelta {
                            index: idx,
                            arguments_delta: args_str,
                        });
                        emitted_any = true;
                    }
                }
            }
        }
        if emitted_any {
            return;
        }
        // Fallback: if `[TOOL_CALLS]` was followed by prose + `<tool_call>...</tool_call>`,
        // re-parse the segment so embedded `<tool_call>` tags are still extracted cleanly.
        if raw.contains(TOOL_OPEN) || raw.contains(FUNC_OPEN) {
            let mut sub = NativeOutputParser::new();
            sub.tool_count = self.tool_count;
            let mut sub_events = sub.push(&raw);
            sub_events.extend(sub.finish());
            self.tool_count = sub.tool_count;
            events.extend(sub_events);
            return;
        }
        events.push(CanonicalStreamEvent::TextDelta(raw));
    }
}

/// Fallback extractor for models (e.g. MiMo-V2.6, Qwen, Llama) that occasionally emit a
/// bare JSON tool call `{"name": "...", "arguments": {...}}` without `<tool_call>` tags
/// after several turns of tool use. Only matches when `"name"` is in `allowed_tools`
/// to prevent false positives on ordinary JSON prose.
pub fn extract_bare_tool_calls(
    raw_output: &str,
    allowed_tools: &[&str],
) -> Vec<(String, String)> {
    if allowed_tools.is_empty() {
        return Vec::new();
    }
    // Strip any `<think>...</think>` prefix so we only inspect the visible answer.
    let body = if let Some(idx) = raw_output.rfind(THINK_CLOSE) {
        &raw_output[idx + THINK_CLOSE.len()..]
    } else {
        raw_output
    };
    let trimmed = body.trim();
    // Unwrap optional ```json ... ``` code fences.
    let candidate = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .and_then(|s| s.strip_suffix("```"))
        .map(str::trim)
        .unwrap_or(trimmed);

    // Find the first `{` or `[` to tolerate a short preamble sentence before the JSON object.
    let json_slice = match candidate.find(|c| c == '{' || c == '[') {
        Some(start) => &candidate[start..],
        None => return Vec::new(),
    };

    let parsed: Value = match parse_json_lenient(json_slice) {
        Some(v) => v,
        None => return Vec::new(),
    };

    let items: Vec<&Value> = match &parsed {
        Value::Object(_) => vec![&parsed],
        Value::Array(arr) => arr.iter().collect(),
        _ => return Vec::new(),
    };

    let mut out = Vec::new();
    for item in items {
        let Some(name) = item.get("name").and_then(Value::as_str) else {
            continue;
        };
        if !allowed_tools.contains(&name) {
            continue;
        }
        let Some(args) = item
            .get("arguments")
            .or_else(|| item.get("parameters"))
            .or_else(|| item.get("input"))
        else {
            continue;
        };
        if args.is_object() {
            let args_str = serde_json::to_string(args).unwrap_or_else(|_| "{}".to_string());
            out.push((name.to_string(), args_str));
        }
    }
    out
}

/// Strict `serde_json::from_str` first; if that fails (common on 7B–9B models writing
/// unescaped shell quotes like `echo "EXIT=$?"` or dropping a closing `}` / adding a stray `)`),
/// attempt a deterministic structural repair and re-parse.
fn parse_json_lenient(raw: &str) -> Option<Value> {
    if let Ok(v) = serde_json::from_str::<Value>(raw) {
        return Some(v);
    }
    let repaired = repair_broken_json(raw);
    serde_json::from_str::<Value>(&repaired).ok()
}

fn repair_broken_json(raw: &str) -> String {
    // Strip stray trailing punctuation outside the JSON object/array (e.g. `]})` -> `]}`)
    let trimmed = raw
        .trim()
        .trim_end_matches(|c: char| !matches!(c, '}' | ']' | '"' | '0'..='9' | 'e' | 'l'));

    let chars: Vec<char> = trimmed.chars().collect();
    let mut out = String::with_capacity(trimmed.len() + 16);
    let mut in_string = false;
    let mut escaped = false;
    let mut open_braces: i32 = 0;
    let mut open_brackets: i32 = 0;

    let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];
        if in_string {
            if escaped {
                out.push(ch);
                escaped = false;
            } else if ch == '\\' {
                out.push(ch);
                escaped = true;
            } else if ch == '"' {
                // Decide whether this `"` genuinely closes the JSON string or is an unescaped quote inside it.
                let mut j = i + 1;
                while j < chars.len() && chars[j].is_whitespace() {
                    j += 1;
                }
                let is_closing = if j >= chars.len() {
                    true
                } else {
                    match chars[j] {
                        ':' | '}' | ']' => true,
                        ',' => {
                            // After `,`, a valid JSON separator is followed by a value/key start
                            let mut k = j + 1;
                            while k < chars.len() && chars[k].is_whitespace() {
                                k += 1;
                            }
                            k < chars.len()
                                && matches!(
                                    chars[k],
                                    '"' | '{' | '[' | '-' | '0'..='9' | 't' | 'f' | 'n'
                                )
                        }
                        _ => false,
                    }
                };
                if is_closing {
                    out.push('"');
                    in_string = false;
                } else {
                    out.push('\\');
                    out.push('"');
                }
            } else if ch == '\n' {
                out.push_str("\\n");
            } else if ch == '\r' {
                out.push_str("\\r");
            } else if ch == '\t' {
                out.push_str("\\t");
            } else {
                out.push(ch);
            }
        } else {
            match ch {
                '"' => {
                    in_string = true;
                    out.push(ch);
                }
                '{' => {
                    open_braces += 1;
                    out.push(ch);
                }
                '}' => {
                    open_braces -= 1;
                    out.push(ch);
                }
                '[' => {
                    open_brackets += 1;
                    out.push(ch);
                }
                ']' => {
                    open_brackets -= 1;
                    out.push(ch);
                }
                _ => out.push(ch),
            }
        }
        i += 1;
    }

    if in_string {
        out.push('"');
    }
    while open_brackets > 0 {
        out.push(']');
        open_brackets -= 1;
    }
    while open_braces > 0 {
        out.push('}');
        open_braces -= 1;
    }
    out
}

fn parse_xml_parameters(raw: &str) -> Option<Value> {
    const P_OPEN: &str = "<parameter=";
    const P_CLOSE: &str = "</parameter>";
    let mut map = serde_json::Map::new();
    let mut rest = raw;

    while let Some(start) = rest.find(P_OPEN) {
        let after_open = &rest[start + P_OPEN.len()..];
        let (key_part, after_gt) = after_open.split_once('>')?;
        let key = key_part.trim();
        let (val_part, remaining) = match after_gt.find(P_CLOSE) {
            Some(end) => (&after_gt[..end], &after_gt[end + P_CLOSE.len()..]),
            None => (after_gt, ""),
        };
        let trimmed = val_part.trim();
        // Preserve strings unless the value is a JSON boolean/number/array/object
        let parsed = match serde_json::from_str::<Value>(trimmed) {
            Ok(v @ (Value::Bool(_) | Value::Number(_) | Value::Array(_) | Value::Object(_))) => v,
            _ => Value::String(trimmed.to_string()),
        };
        if !key.is_empty() {
            map.insert(key.to_string(), parsed);
        }
        rest = remaining;
    }

    if map.is_empty() && !raw.trim().is_empty() {
        None
    } else {
        Some(Value::Object(map))
    }
}

fn longest_tag_prefix(buf: &str, tags: &[&str]) -> usize {
    let mut best = 0usize;
    for tag in tags {
        let max_check = buf.len().min(tag.len().saturating_sub(1));
        for len in (1..=max_check).rev() {
            if buf.is_char_boundary(buf.len() - len) && buf.ends_with(&tag[..len]) {
                best = best.max(len);
                break;
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_think_text_and_tool_calls_across_chunk_boundaries() {
        let mut parser = NativeOutputParser::new();
        let chunks = [
            "<thi",
            "nk>Cek kode ",
            "benefit</thi",
            "nk>Saya cek dulu.<tool_",
            "call>{\"name\":\"lookup_benefit\",",
            "\"arguments\":{\"code\":\"06\"}}</tool_call>",
        ];

        let mut thinking = String::new();
        let mut text = String::new();
        let mut tool_name = String::new();
        let mut tool_args = String::new();

        for c in chunks {
            for ev in parser.push(c) {
                match ev {
                    CanonicalStreamEvent::ThinkingDelta(d) => thinking.push_str(&d),
                    CanonicalStreamEvent::TextDelta(d) => text.push_str(&d),
                    CanonicalStreamEvent::ToolCallStart { name, .. } => tool_name = name,
                    CanonicalStreamEvent::ToolCallDelta { arguments_delta, .. } => {
                        tool_args.push_str(&arguments_delta)
                    }
                    _ => {}
                }
            }
        }
        for ev in parser.finish() {
            if let CanonicalStreamEvent::TextDelta(d) = ev {
                text.push_str(&d);
            }
        }

        assert_eq!(thinking, "Cek kode benefit");
        assert_eq!(text, "Saya cek dulu.");
        assert_eq!(tool_name, "lookup_benefit");
        assert_eq!(tool_args, "{\"code\":\"06\"}");
        assert!(parser.had_tool_calls());
    }

    #[test]
    fn parses_mimo_qwen35_xml_parameter_tool_calls() {
        let mut parser = NativeOutputParser::new();
        let input = "<tool_call><function=list_directory><parameter=path>/Users/test/agentcode</parameter></function></tool_call>";
        let mut events = parser.push(input);
        events.extend(parser.finish());
        assert!(parser.had_tool_calls());
        let mut name = String::new();
        let mut args = String::new();
        for ev in events {
            match ev {
                CanonicalStreamEvent::ToolCallStart { name: n, .. } => name = n,
                CanonicalStreamEvent::ToolCallDelta { arguments_delta, .. } => args.push_str(&arguments_delta),
                _ => {}
            }
        }
        assert_eq!(name, "list_directory");
        assert_eq!(args, "{\"path\":\"/Users/test/agentcode\"}");
    }

    #[test]
    fn extracts_bare_json_tool_call_only_for_declared_tools() {
        let raw = "<think>Check git status</think>\n{\"id\":\"call_native_1\",\"name\":\"terminal\",\"arguments\":{\"command\":\"git status\",\"cd\":\"/tmp\"}}";
        let calls = extract_bare_tool_calls(raw, &["list_directory", "terminal"]);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "terminal");
        assert!(calls[0].1.contains("\"command\":\"git status\""));

        // Must NOT trigger for undeclared tool names
        let none = extract_bare_tool_calls(raw, &["list_directory"]);
        assert!(none.is_empty());
    }

    #[test]
    fn repairs_unescaped_shell_quotes_and_stray_closing_paren_in_tool_json() {
        let raw = r#"The module cache has the key packages cached.
{"name":"terminal","arguments":{"command":"GOFLAGS=-mod=mod go mod download github.com/go-redis/redis/v8 2>&1 | head -20; echo "EXIT=$?"","cd":"/Users/agungpambudi/Development/iseng/agentcode/api","timeout_ms":60000,"allow_hosts":["proxy.golang.org","sum.golang.org"]})"#;
        let calls = extract_bare_tool_calls(raw, &["terminal", "list_directory"]);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "terminal");
        let args: Value = serde_json::from_str(&calls[0].1).expect("repaired args valid json");
        assert_eq!(
            args["command"].as_str().unwrap(),
            r#"GOFLAGS=-mod=mod go mod download github.com/go-redis/redis/v8 2>&1 | head -20; echo "EXIT=$?""#
        );
        assert_eq!(args["timeout_ms"].as_i64().unwrap(), 60000);
    }

    #[test]
    fn parses_multiple_and_duplicate_mistral_tool_calls_and_embedded_xml() {
        let mut p1 = NativeOutputParser::new();
        let mut ev1 = p1.push(
            "[TOOL_CALLS][TOOL_CALLS]find_path[ARGS]{\"glob\":\"*.go\"}[TOOL_CALLS]list_directory[ARGS]{\"path\":\".\"}",
        );
        ev1.extend(p1.finish());
        let names: Vec<String> = ev1
            .into_iter()
            .filter_map(|e| match e {
                CanonicalStreamEvent::ToolCallStart { name, .. } => Some(name),
                _ => None,
            })
            .collect();
        assert_eq!(names, vec!["find_path", "list_directory"]);

        let mut p2 = NativeOutputParser::new();
        let mut ev2 = p2.push(
            "[TOOL_CALLS]I will list the root directory.<tool_call>\n{\"name\":\"list_directory\",\"arguments\":{\"path\":\"agentcode\"}}\n</tool_call>",
        );
        ev2.extend(p2.finish());
        assert!(p2.had_tool_calls());
        let names2: Vec<String> = ev2
            .into_iter()
            .filter_map(|e| match e {
                CanonicalStreamEvent::ToolCallStart { name, .. } => Some(name),
                _ => None,
            })
            .collect();
        assert_eq!(names2, vec!["list_directory"]);
    }
}
