# TPT Local Chat — Build Todo

> Hub listing: `tptsolutions.co.nz/tools/local-chat` (registry slug
> `local-chat`, category `developer`). Spec: `apps.md` §T2-19 in the
> tpt-electrician-nz-2 repo. Base wraps `tpt-spark`. **Price: free** —
> intentionally not a revenue line. Its job is brand visibility and funneling
> into T2-7 Chat With Your PDF (`tpt-app-chat-pdf`) for anyone who needs the
> document-specific paid version. No pro tier, no Gumroad listing. UI on
> `tpt-appfront`.
>
> Hub contract: `wasm-bindgen --target web` glue exposes `mount(container)`
> (called by `WasmAppRunner` after `default()` init). Standalone `trunk serve`
> mounts into the `#tpt-appfront-root` marker div instead — never
> `document.body` — so the two hosts never double-mount.

## Phase 0 — Scaffold & pipeline proof
- [x] `tpt-appfront init tpt-app-local-chat --target dom` (path deps on the tpt-appfront checkout)
      — run into a scratch dir (this dir already held todo.md) and merged in
- [x] Restructure scaffold to hub contract: `#[wasm_bindgen] pub fn mount(el)` + `#[wasm_bindgen(start)]` no-op unless `#tpt-appfront-root` present
- [x] `index.html` carries the `#tpt-appfront-root` marker div
- [x] Build pipeline script mirroring fea-lite/cutlist-optimizer's (`--target web` glue → web repo `public/apps/local-chat/`)
      — `build.ps1` (canonical, UTF-8 BOM) + `build.sh`; both verified (web-repo dest + dist/web fallback);
      hello-world payload ~58 KB gzipped; web repo now has `public/apps/local-chat/` alongside the other apps
- [x] Hello-world build loads through the standalone dev page (`trunk serve`)
      — browser-verified: mount (exactly one `#tpt-appfront-root` child), heading/intro/transcript/
      button/status render, click → dispatch → signal → render → DOM update with the ping counter
      accumulating across renders (the test webview throttles rAF — locator clicks and domSnapshot
      time out; verified via DOM-state reads on a fresh module instance, matching chat-pdf's note —
      not an app bug)
- [x] git init + first commit

## Phase 1 — Core engine (wrap tpt-spark)
- [x] Survey `tpt-spark` for its local-LLM inference support (model
      loading/quantization/inference loop) and what's needed to run
      acceptably in-browser via wasm32
      — cloned from github.com/tpt-solutions/tpt-spark (not previously local).
      Its three engines are all native-only for our purposes: `candle` needs
      rayon + native gemm, the `wgpu` engine mmaps GGUF weights
      (`memmap2::Mmap`), and its `LlmEngine` trait assumes a tokio/Tauri host.
      A wrap cannot compile for wasm32 — same shape of finding as chat-pdf's
      Phase 1 (ort/rusqlite), but here the product IS the model, so instead
      of replacing it with a lexical engine the engine crate re-implements a
      minimal wasm-viable inference path.
- [x] Adapter crate `engine/`: chat message history in → model response out
      as a plain streaming interface — no UI deps, host-unit-testable where
      feasible (model inference itself won't be, but the message-handling
      plumbing should be)
      — `engine/` (tpt-localchat-engine), pure std, zero deps: GGUF v2/v3
      reader → Q4_0/Q8_0/F16/F32/Q4_K/Q6_K dequant (llama.cpp block layouts,
      fused dequant-dot kernels so no f32 weight copy exists) → LLaMA-family
      transformer (RMSNorm, RoPE, GQA, f32 KV cache with cross-turn
      common-prefix reuse, SwiGLU, tied embeddings) → GPT-2-style BPE built
      from the GGUF's own tokenizer metadata → ChatML prompt formatting →
      llama.cpp-style sampler streaming TokenEvents. `ChatEngine` trait +
      `MockChatEngine` cover the plumbing in host tests.
      Validation: 15 unit tests (block layouts vs hand-computed llama.cpp
      references, micro-model forward on a synthetic GGUF, ChatML, mock
      streaming) + 2 ignored-by-default real-model tests
      (`cargo test -p tpt-localchat-engine --release -- --ignored`; fixture =
      SmolLM2-135M-Instruct Q4_0 in `test-data/`, gitignored). The forward
      pass was also checked against an independent NumPy reference
      implementation — exact match through multi-position prefill and greedy
      generation — and the dequantizer against the official `gguf` PyPI
      package (Q8_0 bit-exact; that comparison caught a Q4_0 layout bug:
      nibbles are halves (lows 0..15, highs 16..31), not interleaved).
- [x] Model size/perf budget: this is a single free tier only, so the model
      choice is a straight trade-off between quality and download/load size —
      no "smaller free / bigger pro" split to fall back on
      — SmolLM2-135M-Instruct Q4_0 (QuantFactory GGUF): 91.7 MB download,
      ~92 MB RAM for weights + ~47 MB KV cache at the 2048-token runtime cap,
      ~0.6–1.0 s/token scalar release build native (browser wasm similar
      order; first turn prefills ~100 ChatML tokens, later turns reuse the
      cached prefix). Deliberately the smallest usable instruct model — a
      bigger one multiplies the download visitors must tolerate for free.

## Phase 2 — Free WASM UI (the only tier)
- [ ] Chat UI: message history, input box, streaming response display
- [ ] In-app cross-promotion: a clear, non-pushy callout pointing at
      `chat-pdf` for anyone who wants to chat with their own documents instead
      of general-purpose chat
- [ ] Browser-verified: mount, send a message, get a coherent response,
      cross-promo renders correctly

## Phase 3 — N/A
- [ ] No paid/pro tier for this app — confirm the registry entry has no
      `gumroadUrl` and the hub page doesn't show a "Buy" call to action

## Phase 4 — Ship & measure
- [ ] Registry entry (`developer` category, `wasm: { entry: '/apps/local-chat/local-chat.js' }`,
      no `gumroadUrl`) per the "Adding a finished app to the hub" runbook at
      the bottom of apps.md
- [ ] Verify hub card/detail/sitemap/runner in dev, then prod after deploy —
      confirm it correctly shows as free with no purchase flow
- [ ] WASM size budget check on the shipped build (local-model weights are
      the dominant size cost here — track separately from code size, and
      weigh against how large a free download visitors will tolerate)
