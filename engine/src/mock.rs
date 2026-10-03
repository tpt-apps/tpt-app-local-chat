//! Deterministic mock engine: same [`ChatEngine`] surface as the real one,
//! no weights, instant. Hosts unit-test the chat plumbing (history
//! formatting, streaming, stop handling) against this.

use crate::chatml;
use crate::error::EngineError;
use crate::{ChatEngine, ChatHistory, GenParams, TokenEvent};

/// A mock that replies with a fixed template built from the last user
/// message. Deterministic: same history → same stream, every time.
pub struct MockChatEngine {
    /// Prepended to the echo so tests can assert on it.
    pub marker: String,
}

impl MockChatEngine {
    pub fn new() -> Self {
        Self {
            marker: String::from("[mock]"),
        }
    }

    /// Renders the history through the same ChatML template as the real
    /// engine (the mock exists to test that plumbing).
    pub fn render_prompt(&self, history: &ChatHistory) -> String {
        chatml::render(history)
    }

    fn reply_text(&self, history: &ChatHistory) -> String {
        let last_user = history
            .messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .map(|m| m.content.as_str())
            .unwrap_or("");
        if last_user.is_empty() {
            format!("{} Hello! How can I help?", self.marker)
        } else {
            format!(
                "{} You said: {last_user}",
                self.marker
            )
        }
    }
}

impl Default for MockChatEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ChatEngine for MockChatEngine {
    fn chat(
        &self,
        history: &ChatHistory,
        params: &GenParams,
        on_token: &mut dyn FnMut(TokenEvent) -> Result<(), EngineError>,
    ) -> Result<String, EngineError> {
        if history.messages.is_empty() {
            return Err(EngineError::EmptyPrompt);
        }
        let reply = self.reply_text(history);
        // Stream word-by-word, honoring max_tokens (words, here) and the
        // first stop sequence — the real engine's contract in miniature.
        let mut emitted = String::new();
        for (i, word) in reply.split(' ').enumerate() {
            if i as u32 >= params.max_tokens {
                break;
            }
            let piece = if i == 0 {
                word.to_string()
            } else {
                format!(" {word}")
            };
            emitted.push_str(&piece);
            let mut stop_hit = false;
            for stop in &params.stop_sequences {
                if !stop.is_empty() && emitted.contains(stop.as_str()) {
                    stop_hit = true;
                    break;
                }
            }
            let done = stop_hit || i as u32 + 1 == params.max_tokens || {
                i + 1 == reply.split(' ').count()
            };
            on_token(TokenEvent {
                token: piece,
                done,
            })?;
            if stop_hit {
                break;
            }
        }
        Ok(emitted)
    }

    fn prompt_token_count(&self, history: &ChatHistory) -> Result<usize, EngineError> {
        // Words as a stand-in for tokens: deterministic and good enough for
        // mock-level tests of callers that just need a nonzero estimate.
        let prompt = chatml::render(history);
        Ok(prompt.split_whitespace().count())
    }
}
