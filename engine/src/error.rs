//! Engine errors — deliberately plain (no thiserror) to stay std-only.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineError {
    /// The byte slice is not a GGUF v2/v3 file.
    BadMagic,
    /// GGUF version outside the supported 2..=3 range.
    UnsupportedVersion(u32),
    /// Malformed container: metadata or tensor directory runs past the data.
    Truncated(&'static str),
    /// A metadata key the model requires is missing.
    MissingMeta(String),
    /// The file's architecture is not the supported LLaMA family.
    UnsupportedArchitecture(String),
    /// A tensor uses a quantization this build cannot dequantize.
    UnsupportedQuant { tensor: String, quant: u32 },
    /// A tensor's element count is not a whole number of quant blocks.
    BadTensorShape { tensor: String },
    /// The tokenizer metadata is missing or malformed.
    BadTokenizer(&'static str),
    /// Prompt (+ generation budget) would exceed the runtime context cap.
    ContextOverflow { needed: usize, cap: usize },
    /// The conversation ended up empty / malformed.
    EmptyPrompt,
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadMagic => write!(f, "not a GGUF file (bad magic)"),
            Self::UnsupportedVersion(v) => {
                write!(f, "unsupported GGUF version {v} (need 2 or 3)")
            }
            Self::Truncated(what) => write!(f, "truncated GGUF: {what}"),
            Self::MissingMeta(key) => write!(f, "missing required GGUF metadata: {key}"),
            Self::UnsupportedArchitecture(arch) => {
                write!(f, "unsupported architecture \"{arch}\" (need llama-family)")
            }
            Self::UnsupportedQuant { tensor, quant } => {
                write!(f, "tensor \"{tensor}\" uses unsupported quant type {quant}")
            }
            Self::BadTensorShape { tensor } => {
                write!(f, "tensor \"{tensor}\" is not a whole number of quant blocks")
            }
            Self::BadTokenizer(what) => write!(f, "bad tokenizer metadata: {what}"),
            Self::ContextOverflow { needed, cap } => {
                write!(f, "prompt needs {needed} tokens but the context cap is {cap}")
            }
            Self::EmptyPrompt => write!(f, "conversation is empty"),
        }
    }
}

impl std::error::Error for EngineError {}
