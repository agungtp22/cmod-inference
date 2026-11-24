use serde_json::{json, Value};

use crate::ir::{CanonicalStreamEvent, ToolChoice, ToolDefinition};

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
const TOOL_OPEN: &str = "<tool_call>";
const TOOL_CLOSE: &str = "</tool_call>";
const FUNC_OPEN: &str = "<function=";
const FUNC_CLOSE: &str = "</function>";

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
                    let next_tool = self.buf.find(TOOL_OPEN);
                    let next_func = self.buf.find(FUNC_OPEN);

                    // Pick earliest opening tag among <think>, <tool_call>, and <function=
                    let earliest = [
                        next_think.map(|i| (i, 0u8)),
                        next_tool.map(|i| (i, 1u8)),
                        next_func.map(|i| (i, 2u8)),
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
                        _ => {
                            let keep =
                                longest_tag_prefix(&self.buf, &[THINK_OPEN, TOOL_OPEN, FUNC_OPEN]);
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
                    if let Some(end) = self.buf.find(THINK_CLOSE) {
                        let think_txt = self.buf[..end].to_string();
                        self.buf.drain(..end + THINK_CLOSE.len());
                        if !think_txt.is_empty() {
                            events.push(CanonicalStreamEvent::ThinkingDelta(think_txt));
                        }
                        self.state = ParseState::Normal;
                    } else {
                        let keep = longest_tag_prefix(&self.buf, &[THINK_CLOSE]);
                        let emit_len = self.buf.len().saturating_sub(keep);
                        if emit_len > 0 {
                            let out = self.buf[..emit_len].to_string();
                            self.buf.drain(..emit_len);
                            events.push(CanonicalStreamEvent::ThinkingDelta(out));
                        }
                        break;
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
        if let Ok(parsed) = serde_json::from_str::<Value>(&raw) {
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
                let args_val = serde_json::from_str::<Value>(args_raw)
                    .ok()
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
}
