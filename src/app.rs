//! Phase 0 pipeline proof (tpt-appfront-dom).
//!
//! Proves the hub-contract plumbing end to end — `mount(container)` renders
//! the shell, a button click runs dispatch → signal → fine-grained re-render
//! — so the Phase 2 chat UI grows into a verified pipeline instead of a
//! blank page. The real chat transcript/model streaming replaces the demo
//! region in Phase 2; the mount/render/reactive_text split (input panel
//! mounted once so the text box never loses focus, transcript re-rendered
//! into a placeholder container) is already the shape Phase 2 uses.

use std::rc::Rc;

use tpt_appfront_core::{Signal, UITree};
use wasm_bindgen::prelude::*;
use web_sys::Element;

#[derive(Debug, Clone)]
enum Msg {
    Ping,
}

/// Mount the whole app into `container` (hub container div or the
/// `#tpt-appfront-root` marker).
pub fn mount_app(container: &Element) -> Result<(), JsValue> {
    let document = container
        .owner_document()
        .ok_or_else(|| JsValue::from_str("no owner document"))?;

    let status = Signal::new(String::from("Pipeline proof — click the button."));
    let pings: Signal<u32> = Signal::new(0);

    let ui: UITree<Msg> = UITree::container(|c| {
        c.heading(1, "TPT Local Chat");
        c.text(
            "Chat with a small language model running entirely in your \
             browser — no cloud, no API key, nothing uploaded.",
        );
        // Transcript placeholder — `render` owns this container's children.
        c.container(|_| {})
            .attr("id", "tpt-localchat-transcript")
            .attr("class", "tpt-localchat-transcript");
        c.button("Ping the pipeline").on_click(Msg::Ping);
    });

    let pings_for_dispatch = pings.clone();
    let dispatch: Rc<dyn Fn(Msg)> = Rc::new(move |msg| match msg {
        Msg::Ping => pings_for_dispatch.set(pings_for_dispatch.get() + 1),
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

    // Transcript region: fine-grained re-render driven by the `pings` signal.
    {
        let transcript = container
            .query_selector("#tpt-localchat-transcript")
            .ok()
            .flatten()
            .ok_or_else(|| JsValue::from_str("transcript container missing"))?;
        let pings_for_view = pings.clone();
        let view: Rc<dyn Fn() -> UITree<Msg>> = Rc::new(move || {
            let n = pings_for_view.get();
            UITree::container(|c| {
                if n == 0 {
                    c.text("No pings yet — the transcript region re-renders here.");
                } else {
                    c.text(format!("Pipeline ping #{n}: dispatch → signal → render OK."));
                }
            })
        });
        // Whole-process root mount: keep the reconciler alive forever, like
        // the other handles in this app.
        std::mem::forget(tpt_appfront_dom::render(&transcript, view, Rc::clone(&dispatch))?);
    }

    let _ = status; // Phase 2 uses this for model-load / streaming status.
    Ok(())
}
