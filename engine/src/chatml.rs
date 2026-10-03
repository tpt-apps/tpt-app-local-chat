//! ChatML prompt formatting.
//!
//! Every model this app ships uses the ChatML template (`<|im_start|>role
//! \n content <|im_end|>\n`), which is exactly what its GGUF
//! `tokenizer.chat_template` renders for a plain message list — the
//! template is embedded for transparency, but rendering is hardcoded so
//! the engine stays std-only and deterministically testable.

use crate::ChatHistory;

/// The default system prompt (matches the model's own template default).
pub const DEFAULT_SYSTEM_PROMPT: &str =
    "You are a helpful AI assistant named SmolLM, trained by Hugging Face";

/// Kept as a compile-time assertion that the template assumption is
/// documented where it is used (see `lib.rs` docs).
pub const CHAT_ML_TEMPLATE_NOTE: &str = "ChatML: <|im_start|>{role}\\n{content}<|im_end|>\\n";

/// Renders the history into the ready-to-tokenize prompt, ending with the
/// assistant generation header. An optional single system message is
/// honored; when none was given, the model-template's default system line
/// is inserted first.
pub fn render(history: &ChatHistory) -> String {
    let mut system = String::new();
    let mut turns = String::with_capacity(512);
    for message in &history.messages {
        match message.role.as_str() {
            "system" => {
                system.push_str("<|im_start|>system\n");
                system.push_str(&message.content);
                system.push_str("<|im_end|>\n");
            }
            role @ ("user" | "assistant") => {
                turns.push_str("<|im_start|>");
                turns.push_str(role);
                turns.push('\n');
                turns.push_str(&message.content);
                turns.push_str("<|im_end|>\n");
            }
            _ => {
                // Unknown roles render as ChatML user turns rather than
                // corrupting the template.
                turns.push_str("<|im_start|>user\n");
                turns.push_str(&message.content);
                turns.push_str("<|im_end|>\n");
            }
        }
    }
    if system.is_empty() {
        system.push_str("<|im_start|>system\n");
        system.push_str(DEFAULT_SYSTEM_PROMPT);
        system.push_str("<|im_end|>\n");
    }
    let mut out = system;
    out.push_str(&turns);
    out.push_str("<|im_start|>assistant\n");
    out
}
