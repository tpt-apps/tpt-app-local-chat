# TPT Local Chat

Free in-browser chat with a small language model running entirely on the
visitor's device — no cloud, no API key, nothing uploaded.

Intentionally not a revenue line (apps.md §T2-19): its job is brand
visibility and funneling into [TPT Chat PDF](https://tptsolutions.co.nz/tools/chat-pdf)
for anyone who wants document-specific chat. No pro tier, no Gumroad listing.

Built on [TPT AppFront](https://github.com/tpt-solutions/tpt-appfront) (DOM
backend); the model/inference plumbing lives in `tpt-spark` behind the
target-independent `engine/` adapter crate.

## Hub contract

`tptsolutions.co.nz/tools` embeds the wasm-bindgen `--target web` glue: the
runner dynamic-imports it, awaits the default export (wasm init), then calls
the named export `mount(container)` with the page's container div. Standalone
dev (`trunk serve`) mounts via the `#tpt-appfront-root` marker div that only
this repo's `index.html` provides — never `document.body` — so the two hosts
never double-mount.

## Develop

```sh
trunk serve            # standalone dev page, mounts into #tpt-appfront-root
```

## Build & ship

```sh
./build.ps1            # Windows (canonical); or ./build.sh on macOS/Linux
```

Copies the wasm-bindgen glue + wasm to the web repo's `public/apps/local-chat/`
(when the `tpt-electrician-nz-2` checkout exists, or set `TPT_WEB_REPO`),
otherwise to `dist/web/`. The build prints the gzipped code-size report;
local-model weights are the dominant size cost and are tracked separately.

## Layout

- `src/lib.rs` — hub contract entry points (`mount` + guarded `start`)
- `src/app.rs` — WASM UI (chat transcript, input, streaming display)
- `engine/` — target-independent chat adapter (history in → streaming
  response out); host-unit-testable where feasible
