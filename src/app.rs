//! Free WASM UI (tpt-appfront-dom) — the only tier.
//!
//! Rendering architecture (mirrors the fea-lite / chat-pdf split):
//! - the **input panel is mounted once** (`tpt_appfront_dom::mount`) — the
//!   text box never loses focus while typing; its value is read straight
//!   from the DOM when Send is pressed (Enter also sends, via a raw
//!   keydown listener — the DOM backend's `on_input` doesn't carry keys);
//! - the **transcript and model-section re-render fine-grained** via
//!   `tpt_appfront_dom::render` into placeholder containers, driven by
//!   signals;
//! - generation runs through the engine's stepwise [`ChatSession`] — one
//!   token per loop iteration with a microtask yield between steps, so the
//!   page stays responsive while the model thinks.

use std::rc::Rc;

use tpt_appfront_core::{Signal, UITree};
use tpt_localchat_engine::{
    ChatEngine, ChatHistory, ChatMessage, GenParams, LocalChatEngine, TokenEvent,
};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Element, HtmlTextAreaElement, Window};

/// The GGUF the app runs: SmolLM2-135M-Instruct, Q4_0 (~92 MB). Overridable
/// for development with `?model=<url>` (must be CORS-readable).
const DEFAULT_MODEL_URL: &str =
    "https://huggingface.co/QuantFactory/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct.Q4_0.gguf";
const CACHE_NAME: &str = "tpt-localchat-model-v1";

/// The generation budget per turn; the runtime context cap is 2048 tokens
/// inside the engine (see the model budget note in todo.md).
fn gen_params() -> GenParams {
    GenParams {
        max_tokens: 256,
        ..GenParams::default()
    }
}

/// One exchange in the transcript. The assistant side streams.
#[derive(Clone)]
struct Turn {
    user: String,
    assistant: String,
}

#[derive(Debug, Clone)]
enum Msg {
    Download,
    Send,
    NewChat,
}

pub fn mount_app(container: &Element) -> Result<(), JsValue> {
    let document = container
        .owner_document()
        .ok_or_else(|| JsValue::from_str("no owner document"))?;

    let status = Signal::new(String::from(
        "Download the model once (about 92 MB) — then everything runs \
         right here in your browser. Nothing is sent anywhere.",
    ));
    let download_pct = Signal::new(0u32);
    let model: Signal<Option<Rc<LocalChatEngine>>> = Signal::new(None);
    let turns: Signal<Rc<Vec<Turn>>> = Signal::new(Rc::new(Vec::new()));
    let busy = Signal::new(false);

    let ui: UITree<Msg> = UITree::container(|c| {
        c.heading(1, "TPT Local Chat");
        c.text(
            "Chat with a small language model running entirely in your \
             browser — no cloud, no API key, nothing uploaded.",
        );
        // Model section placeholder — `render` owns this container.
        c.container(|_| {})
            .attr("id", "tpt-localchat-model")
            .attr("class", "tpt-localchat-model");
        // Transcript placeholder — `render` owns this container.
        c.container(|_| {})
            .attr("id", "tpt-localchat-transcript")
            .attr("class", "tpt-localchat-transcript");
        c.container(|c| {
            c.textarea("")
                .attr("id", "tpt-localchat-input")
                .attr("placeholder", "Ask anything — e.g. What is 15 + 27?")
                .attr("rows", "2");
            c.button("Send").on_click(Msg::Send);
        })
        .attr("class", "tpt-localchat-ask");
        // Cross-promotion: the free app's job is funneling into chat-pdf
        // (apps.md §T2-19) — one clear, non-pushy pointer.
        c.heading(2, "Want to chat with your own documents?");
        c.text(
            "TPT Chat PDF answers questions about your PDFs with quoted, \
             page-cited answers — the same fully on-device approach. This \
             general-purpose chat and Chat PDF are both free to try.",
        );
        c.link(
            "https://tptsolutions.co.nz/tools/chat-pdf",
            "Try TPT Chat PDF — free",
        );
    });

    // ---- dispatch ----
    let status_for_dispatch = status.clone();
    let pct_for_dispatch = download_pct.clone();
    let model_for_dispatch = model.clone();
    let turns_for_dispatch = turns.clone();
    let busy_for_dispatch = busy.clone();
    let container_for_dispatch = container.clone();
    let dispatch: Rc<dyn Fn(Msg)> = Rc::new(move |msg| match msg {
        Msg::Download => {
            let status = status_for_dispatch.clone();
            let pct = pct_for_dispatch.clone();
            let model = model_for_dispatch.clone();
            wasm_bindgen_futures::spawn_local(async move {
                load_model(&status, &pct, &model).await;
            });
        }
        Msg::Send => {
            if busy_for_dispatch.get() {
                return;
            }
            let Some(engine) = model_for_dispatch.get() else {
                status_for_dispatch
                    .set("Download the model first — then chat away.".to_string());
                return;
            };
            let Some(input) = query_textarea(&container_for_dispatch) else {
                return;
            };
            let text = input.value().trim().to_string();
            if text.is_empty() {
                status_for_dispatch.set("Type a message first.".to_string());
                return;
            }
            input.set_value("");

            let turns_signal = turns_for_dispatch.clone();
            let status = status_for_dispatch.clone();
            let busy = busy_for_dispatch.clone();

            // History for the engine: everything so far plus this message,
            // trimmed from the front if the rendered prompt wouldn't fit.
            let mut history = ChatHistory::new();
            for turn in turns_for_dispatch.get().iter() {
                history.push(ChatMessage::user(turn.user.clone()));
                history.push(ChatMessage::assistant(turn.assistant.clone()));
            }
            history.push(ChatMessage::user(text.clone()));
            let budget = gen_params();
            match engine.prompt_token_count(&history) {
                Ok(count) if count + budget.max_tokens as usize > engine.context_cap() => {
                    status.set(
                        "This conversation has outgrown the model's memory. \
                         Start a new chat to keep going."
                            .to_string(),
                    );
                    return;
                }
                Err(_) => {
                    status.set("Something went wrong tokenizing the prompt.".to_string());
                    return;
                }
                Ok(_) => {}
            }

            let mut next: Vec<Turn> = turns_for_dispatch.get().as_ref().clone();
            next.push(Turn {
                user: text,
                assistant: String::new(),
            });
            turns_signal.set(Rc::new(next));
            busy.set(true);
            status.set("Thinking…".to_string());

            log("send: spawning generate");
            wasm_bindgen_futures::spawn_local(async move {
                generate(engine, history, budget, turns_signal, status, busy).await;
            });
        }
        Msg::NewChat => {
            turns_for_dispatch.set(Rc::new(Vec::new()));
            if let Some(engine) = model_for_dispatch.get() {
                engine.reset_cache();
            }
            status_for_dispatch.set("New chat — memory cleared.".to_string());
        }
    });


    tpt_appfront_dom::mount(container, &ui, Rc::clone(&dispatch))?;

    // Status line under the heading (reactive fine-grained text).
    {
        let (text_node, text_handle) = tpt_appfront_dom::reactive_text(&document, status.clone())?;
        container.append_child(&text_node)?;
        // Whole-process root mount: forgetting is an explicit choice here,
        // matching the scaffold and the other hub apps.
        std::mem::forget(text_handle);
    }

    // Model section: re-renders from the model + progress signals.
    {
        let section = container
            .query_selector("#tpt-localchat-model")
            .ok()
            .flatten()
            .ok_or_else(|| JsValue::from_str("model section missing"))?;
        let model_for_view = model.clone();
        let pct_for_view = download_pct.clone();
        let view: Rc<dyn Fn() -> UITree<Msg>> = Rc::new(move || {
            let pct = pct_for_view.get();
            if model_for_view.get().is_some() {
                return UITree::container(|c| {
                    c.text(
                        "Model ready: SmolLM2 135M (91.7 MB, cached for next \
                         visit).",
                    );
                    c.button("New chat").on_click(Msg::NewChat);
                });
            }
            UITree::container(|c| {
                if pct > 0 {
                    c.text(format!("Downloading… {pct}%"));
                } else {
                    c.text(
                        "One-time download: SmolLM2 135M, a small open model \
                         (91.7 MB). It is stored in your browser, so the next \
                         visit needs no download.",
                    );
                }
                c.button("Download model (92 MB)").on_click(Msg::Download);
            })
        });
        // Whole-process root mount: keep the reconciler alive forever, like
        // the other handles in this app.
        std::mem::forget(tpt_appfront_dom::render(&section, view, Rc::clone(&dispatch))?);
    }

    // Transcript: re-renders from the `turns` signal (per streamed token).
    {
        let transcript = container
            .query_selector("#tpt-localchat-transcript")
            .ok()
            .flatten()
            .ok_or_else(|| JsValue::from_str("transcript container missing"))?;
        let turns_for_view = turns.clone();
        let busy_for_view = busy.clone();
        let view: Rc<dyn Fn() -> UITree<Msg>> = Rc::new(move || {
            let turns = turns_for_view.get();
            let streaming = busy_for_view.get();
            UITree::container(|c| {
                if turns.is_empty() {
                    c.text(
                        "The conversation appears here. Try asking for a \
                         calculation, a summary, or an explanation — the \
                         model is small, so keep questions focused.",
                    );
                    return;
                }
                for turn in turns.iter() {
                    c.container(|c| {
                        c.text(format!("You: {}", turn.user));
                    })
                    .attr("class", "tpt-localchat-msg-user");
                    c.container(|c| {
                        c.text(format!("Assistant: {}", turn.assistant));
                    })
                    .attr("class", "tpt-localchat-msg-assistant");
                }
                let _ = streaming;
            })
        });
        // Whole-process root mount: keep the reconciler alive forever.
        std::mem::forget(tpt_appfront_dom::render(&transcript, view, Rc::clone(&dispatch))?);
    }

    attach_enter_key(container, dispatch)?;

    Ok(())
}

/// Runs one streaming assistant turn. `start_session` prefills (blocking
/// briefly), then each `next_token` step yields to the event loop so the
/// partial answer paints as it grows.
async fn generate(
    engine: Rc<LocalChatEngine>,
    history: ChatHistory,
    params: GenParams,
    turns: Signal<Rc<Vec<Turn>>>,
    status: Signal<String>,
    busy: Signal<bool>,
) {
    log("gen: start_session begin");
    let mut session = match engine.start_session(&history, &params) {
        Ok(session) => {
            log("gen: session ready");
            session
        }
        Err(e) => {
            log(&format!("gen: start_session err: {e}"));
            status.set(format!("Could not start: {e}"));
            busy.set(false);
            return;
        }
    };

    loop {
        let turns_for_token = turns.clone();
        let result = session.next_token(&mut |event: TokenEvent| {
            append_to_last_turn(&turns_for_token, &event.token);
            Ok(())
        });
        match result {
            Ok(Some(_)) => {
                // Hand the frame back to the browser so the streamed text
                // paints and the page stays interactive.
                yield_to_ui().await;
            }
            Ok(None) => {
                log("gen: done");
                break;
            }
            Err(e) => {
                log(&format!("gen: token err: {e}"));
                status.set(format!("Generation failed: {e}"));
                busy.set(false);
                return;
            }
        }
    }

    log("gen: loop finished");
    status.set(String::new());
    busy.set(false);
}

fn append_to_last_turn(turns: &Signal<Rc<Vec<Turn>>>, piece: &str) {
    let mut next: Vec<Turn> = turns.get().as_ref().clone();
    if let Some(last) = next.last_mut() {
        last.assistant.push_str(piece);
    }
    turns.set(Rc::new(next));
}

fn log(msg: &str) {
    web_sys::console::log_1(&JsValue::from_str(&format!("[localchat] {msg}")));
}

/// One macrotask/microtask turn — enough for the renderer to flush the
/// updated transcript between model steps.
async fn yield_to_ui() {
    let _ = JsFuture::from(js_sys::Promise::resolve(&JsValue::UNDEFINED)).await;
}

/// Loads the model: cache-first, then a streamed download with progress,
/// then the (blocking, brief) engine build.
async fn load_model(
    status: &Signal<String>,
    pct: &Signal<u32>,
    model: &Signal<Option<Rc<LocalChatEngine>>>,
) {
    if model.get().is_some() {
        return;
    }
    let Some(window) = web_sys::window() else {
        return;
    };
    let url = model_url(&window);

    status.set("Looking for a cached copy of the model…".to_string());
    let cached_bytes = read_from_cache(&window, &url).await;
    let bytes = match cached_bytes {
        Some(bytes) => {
            status.set("Found the cached model — loading…".to_string());
            bytes
        }
        None => match stream_download(&window, &url, status, pct).await {
            Ok(bytes) => bytes,
            Err(_) => {
                status.set(
                    "The download failed — check your connection and try \
                     again."
                        .to_string(),
                );
                pct.set(0);
                return;
            }
        },
    };

    // Let the status paint before the (single-threaded) engine build.
    yield_to_ui().await;
    status.set("Loading the model into memory…".to_string());
    yield_to_ui().await;

    let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
    match LocalChatEngine::load(leaked) {
        Ok(engine) => {
            model.set(Some(Rc::new(engine)));
            status.set("Model ready — ask away. It all stays on this device.".to_string());
        }
        Err(e) => {
            status.set(format!("The model file could not be loaded: {e}"));
            pct.set(0);
        }
    }
}

/// `?model=<url>` override for development; the packaged default otherwise.
fn model_url(window: &Window) -> String {
    window
        .location()
        .search()
        .ok()
        .and_then(|search| {
            let query = search.strip_prefix('?')?;
            for pair in query.split('&') {
                let mut parts = pair.splitn(2, '=');
                if parts.next() == Some("model") {
                    return parts.next().map(|v| {
                        urldecode(v)
                    });
                }
            }
            None
        })
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| DEFAULT_MODEL_URL.to_string())
}

fn urldecode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("");
                match u8::from_str_radix(hex, 16) {
                    Ok(byte) => {
                        out.push(byte);
                        i += 3;
                    }
                    Err(_) => {
                        out.push(bytes[i]);
                        i += 1;
                    }
                }
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            byte => {
                out.push(byte);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).to_string()
}

async fn cache_storage(window: &Window) -> Option<web_sys::Cache> {
    let caches = window.caches().ok()?;
    let promise = caches.open(CACHE_NAME);
    JsFuture::from(promise)
        .await
        .ok()
        .and_then(|v| v.dyn_into::<web_sys::Cache>().ok())
}

async fn read_from_cache(window: &Window, url: &str) -> Option<Vec<u8>> {
    let cache = cache_storage(window).await?;
    let found = cache.match_with_str(url);
    let value = JsFuture::from(found).await.ok()?;
    let response: web_sys::Response = value.dyn_into().ok()?;
    let buffer = JsFuture::from(response.array_buffer().ok()?).await.ok()?;
    Some(js_sys::Uint8Array::new(&buffer).to_vec())
}

async fn stream_download(
    window: &Window,
    url: &str,
    status: &Signal<String>,
    pct: &Signal<u32>,
) -> Result<Vec<u8>, JsValue> {
    let request = web_sys::Request::new_with_str(url)?;
    let response: web_sys::Response =
        JsFuture::from(window.fetch_with_request(&request)).await?.dyn_into()?;
    if !response.ok() {
        return Err(JsValue::from_str("bad status"));
    }
    // Keep a clone for the cache before the body is consumed.
    let for_cache = response.clone();

    let total = response
        .headers()
        .get("content-length")
        .ok()
        .flatten()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0);

    status.set(if total > 0.0 {
        "Downloading the model — one time, then it is cached…".to_string()
    } else {
        "Downloading the model…".to_string()
    });

    let body = response
        .body()
        .ok_or_else(|| JsValue::from_str("no body"))?;
    let reader = body
        .get_reader()
        .dyn_into::<web_sys::ReadableStreamDefaultReader>()?;

    let mut bytes: Vec<u8> = Vec::with_capacity(total as usize);
    let mut last_pct = 0u32;
    loop {
        let chunk = JsFuture::from(reader.read()).await?;
        let done = js_sys::Reflect::get(&chunk, &"done".into())?
            .as_bool()
            .unwrap_or(true);
        if done {
            break;
        }
        let value = js_sys::Reflect::get(&chunk, &"value".into())?;
        let array: js_sys::Uint8Array = value.dyn_into()?;
        bytes.extend_from_slice(&array.to_vec());
        if total > 0.0 {
            let this = (bytes.len() as f64 * 100.0 / total) as u32;
            if this != last_pct && this <= 100 {
                last_pct = this;
                pct.set(this);
                // Paint roughly every 4% so progress is visible without
                // slowing the stream down.
                if this % 4 == 0 {
                    yield_to_ui().await;
                }
            }
        }
    }

    // Persist for future visits (fire and forget — failures are harmless).
    if let Some(cache) = cache_storage(window).await {
        let for_cache = for_cache?;
        let _ = cache.put_with_str(url, &for_cache);
    }

    Ok(bytes)
}

fn query_textarea(container: &Element) -> Option<HtmlTextAreaElement> {
    container
        .query_selector("#tpt-localchat-input")
        .ok()
        .flatten()
        .and_then(|el| el.dyn_into().ok())
}

/// Enter (without Shift) sends — the DOM backend's `on_input` doesn't carry
/// key events, so a raw keydown listener is attached, matching chat-pdf's
/// file-input precedent.
fn attach_enter_key(container: &Element, dispatch: Rc<dyn Fn(Msg)>) -> Result<(), JsValue> {
    let Some(input) = query_textarea(container) else {
        return Err(JsValue::from_str("input missing"));
    };
    let on_key: Closure<dyn FnMut(web_sys::KeyboardEvent)> =
        Closure::wrap(Box::new(move |event: web_sys::KeyboardEvent| {
            if event.key() == "Enter" && !event.shift_key() {
                event.prevent_default();
                dispatch(Msg::Send);
            }
        }));
    input.add_event_listener_with_callback("keydown", on_key.as_ref().unchecked_ref())?;
    on_key.forget();
    Ok(())
}
