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
- [ ] Survey `tpt-spark` for its local-LLM inference support (model
      loading/quantization/inference loop) and what's needed to run
      acceptably in-browser via wasm32
- [ ] Adapter crate `engine/`: chat message history in → model response out
      as a plain streaming interface — no UI deps, host-unit-testable where
      feasible (model inference itself won't be, but the message-handling
      plumbing should be)
- [ ] Model size/perf budget: this is a single free tier only, so the model
      choice is a straight trade-off between quality and download/load size —
      no "smaller free / bigger pro" split to fall back on

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
