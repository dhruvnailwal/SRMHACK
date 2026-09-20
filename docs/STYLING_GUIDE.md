# UI/UX Styling Guide — Unseen-Customer Fraud Detection

### Design language for the analyst dashboard (Dark/Navy theme)

A complete styling reference for the premium financial-analytics look. It is
written as a **Tailwind CSS v4 + React component spec** (the requested
design language: Vite entry point, single `tailwind.css` import) **and — in
the final section — is mapped 1:1 onto the live product**, which today ships
as a **Streamlit** app (`dashboard/app.py`). The visual system is identical;
only the renderer differs.

> **Audience.** Component breakouts, Tailwind v4 utility lists, colour/type
> tokens, and interactive states that a frontend engineer can implement
> directly.

---

## 1. Design principles

1. **Decision-first.** The risk band is the loudest element on the page; a
   stakeholder should grasp "red = act" from peripheral vision.
2. **One page, three acts.** Overview (story) → Upload & Analyse (action) →
   System Evaluation (evidence). The user should never drill into menus.
3. **Data over decoration.** Metrics are numbers with intent: paired with a
   label, a context line, and a delta where meaningful. No 3D, no noise.
4. **Progressive disclosure.** Everything an alert needs lives *inside* an
   expander; the list stays scannable.
5. **Trustworthy numbers.** Any metric that implies quality (PR-AUC, recall)
   is labelled with its source cohort ("this file", "held-out test fold").

---

## 2. Theme tokens

### Colour palette (Dark/Navy)

| token | hex | usage |
|---|---|---|
| `--bg-base` | `#0B1020` | page background (near-navy black) |
| `--bg-surface` | `#141A30` | cards, panels |
| `--bg-elevated` | `#1D2540` | hover, expanders, sticky headers |
| `--bg-input` | `#0F152B` | text inputs, code |
| `--border-subtle` | `#262F4D` | hairline borders |
| `--text-primary` | `#EAF0FF` | headlines |
| `--text-secondary` | `#9AA7C7` | body copy |
| `--text-muted` | `#5D6A8C` | captions, meta |
| `--accent` | `#3B82F6` | primary buttons, links, focus ring |
| `--accent-soft` | `rgba(59,130,246,.12)` | accent tint backgrounds |

### Risk band semantics (the colour language)

| band | token | hex | Tailwind class | meaning |
|---|---|---|---|---|
| normal | `--band-green` | `#22C55E` | `bg-emerald-500` | allow |
| monitor | `--band-amber` | `#F59E0B` | `bg-amber-500` | log only |
| review | `--band-orange` | `#F97316` | `bg-orange-500` | alert — one strong signal |
| high-risk | `--band-red` | `#EF4444` | `bg-red-500` | alert — both strong signals |

Badge pattern: solid chip for the active band, `pill` radius, white text,
small caps label:

```tsx
<span className="inline-flex items-center rounded-full bg-red-500 px-3 py-1
  text-xs font-semibold uppercase tracking-wide text-white">
  high-risk
</span>
```

### Typography

| role | font | size / weight | class |
|---|---|---|---|
| Hero title | Inter | 42 px / 800 | `text-[42px] font-extrabold tracking-tight` |
| Section header | Inter | 24 px / 700 | `text-2xl font-bold` |
| Card heading | Inter | 16 px / 600 | `text-base font-semibold` |
| Numeric metric | Inter / tabular | 28 px / 800 | `text-[28px] font-extrabold tabular-nums` |
| Body | Inter | 14 px / 400 | `text-sm leading-relaxed` |
| Caption | Inter | 12 px / 500 | `text-xs font-medium text-slate-400` |
| Mono (IDs, code) | JetBrains Mono | 13 px | `font-mono text-[13px]` |

Implement via Tailwind v4 `@theme`:

```css
@import "tailwindcss";
@theme {
  --font-sans: "Inter", ui-sans-serif, system-ui, sans-serif;
  --font-mono: "JetBrains Mono", ui-monospace, monospace;
  --color-bg-base: #0B1020;
  --color-bg-surface: #141A30;
  --color-accent: #3B82F6;
  --color-band-normal: #22C55E;
  --color-band-monitor: #F59E0B;
  --color-band-review: #F97316;
  --color-band-high: #EF4444;
}
```

---

## 3. Page anatomy (three acts)

### Layout grid
```
┌────────────────────────────────────────────────────────────┐
│ <Nav />  sticky, blurred, logo + live API status dot        │
├────────────────────────────────────────────────────────────┤
│ <OverviewHero />      act 1 · story + decision matrix       │
│ <ProcessSteps />      4 steps: Load → Score → Decide → R    │
│ ─────────────────────────────────────────────────────────── │
│ <UploadAnalyse />     act 2 · backend choice + upload zone  │
│  ├ <BackendPicker />  in-process / FastAPI toggle           │
│  ├ <UploadDropzone /> dashed target, type="csv" only        │
│  └ <AnalyseButton />  primary, full-width, spinner state    │
│ <ResultsBoard />      metrics row + per-txn expander cards  │
│  ├ <MetricRow />      PR-AUC / recall@2% / alerts / rate    │
│  ├ <TxnCard />        probability bar + novelty bar + badge │
│  └ <SuspectedList />  datatable sorted by risk rank         │
│ ─────────────────────────────────────────────────────────── │
│ <SystemEval />        act 3 · this-file metrics, dynamic    │
│  ├ <EvalMetricGrid /> 4-up metric cards                     │
│  ├ <SeenVsUnseenChart /> bar: PR-AUC seen vs unseen         │
│  └ <CompliancePanel /> leakage checklist                    │
└────────────────────────────────────────────────────────────┘
```
`max-w-[1440px] mx-auto px-6 lg:px-10`, sections separated by
`border-t border-[--border-subtle]` (Tailwind v4: `border-t border-slate-700/40`).

## 4. Component library

### Overview Hero

```tsx
<section className="grid gap-8 lg:grid-cols-[3fr_2fr] items-start
  rounded-2xl border border-slate-700/40 bg-[--bg-surface] p-8">
  <div>
    <h1 className="text-[42px] font-extrabold tracking-tight
      text-[--text-primary]">Unseen-Customer Fraud Detection</h1>
    <p className="mt-2 text-lg text-[--accent] font-semibold">
      Stops fraud even on customers the model has never seen.
    </p>
    <p className="mt-4 text-sm leading-relaxed text-[--text-secondary] max-w-xl">
      Every transaction gets a supervised <b>fraud probability</b>, an
      unsupervised <b>novelty score</b>, and a plain-language explanation —
      computed point-in-time on a leak-safe pipeline.
    </p>
    {processSteps.map(...)}  {/* 4 numbered chips: Load / Score / Decide / Explain */}
  </div>
  <div> {/* decision-matrix heatmap built from a 2×2 grid */}
    <MatrixHeatmap />
  </div>
</section>
```

`ProcessSteps` chip: `inline-flex items-center gap-1 rounded-md px-2 py-1
bg-[--accent-soft] text-xs font-semibold text-[--accent]`.

### Upload & Analyse

**Dropzone** — dashed, tolerant states:
```tsx
<div className="rounded-xl border-2 border-dashed border-slate-600
  bg-[--bg-input] px-6 py-10 text-center transition
  hover:border-[--accent] hover:bg-[--accent-soft] cursor-pointer">
  <p className="text-sm font-medium text-[--text-secondary]">
    Drop your transaction CSV here
  </p>
  <p className="mt-1 text-xs text-[--text-muted]">
    columns: timestamp, amount, customer_id, merchant_id, device_id [, is_fraud]
  </p>
</div>
```
On drag-over: `border-[--accent] ring-2 ring-[--accent]/30`.

**Backend picker** — segmented control: container `inline-flex rounded-lg
border border-slate-700 bg-[--bg-input] p-1`, active tab `rounded-md
bg-[--accent] text-white px-3 py-1 text-sm font-semibold`, inactive
`px-3 py-1 text-sm text-[--text-secondary] hover:text-[--text-primary]`.

**Analyse button** — primary CTA: `w-full rounded-lg bg-[--accent] py-2.5
text-sm font-semibold text-white transition hover:bg-blue-500
disabled:cursor-not-allowed disabled:opacity-50`; spinner state renders
`` inline after the label.

### Results Board — per-transaction card (the signature component)

```tsx
<details className="group rounded-xl border border-slate-700/40
  bg-[--bg-surface]">
  <summary className="flex flex-wrap items-center gap-3 px-4 py-3 cursor-pointer
    hover:bg-[--bg-elevated]">
    <code className="text-[13px] font-mono text-[--text-secondary]">TXN-1192</code>
    <span className="text-sm text-[--text-muted]">2024-05-01 10:31</span>
    <span className="text-sm font-semibold text-[--text-primary]">$2,410.00</span>
    <RiskBadge band="high-risk" />          {/* red pill */}
    <span className="ml-auto text-xs text-[--accent]">risk 0.912</span>
  </summary>
  <div className="grid gap-4 border-t border-slate-700/40 px-4 py-4 md:grid-cols-3">
    <ScoreBar label="Fraud probability" value={0.89} color="text-red-400" />
    <ScoreBar label="Novelty score"     value={0.93} color="text-amber-400" />
    <div>
      <p className="text-xs font-semibold uppercase text-[--text-muted]">Top reasons</p>
      <ul className="mt-1 space-y-1 text-sm text-[--text-secondary]">
        <li>· 9 purchases from this device in 5 minutes</li>
        <li>· amount 3.1σ above this customer's norm</li>
        <li>· merchant first seen 6 minutes ago</li>
      </ul>
    </div>
  </div>
</details>
```

`ScoreBar` fill: min width for readability —
`h-2.5 rounded-full bg-slate-700 overflow-hidden` →
`<div style={{width: pct}} className="h-full bg-gradient-to-r from-amber-400
to-red-500" />`; value as `text-sm font-bold tabular-nums`.

### System Evaluation — dynamic metric cards

```tsx
<figure className="rounded-xl border border-slate-700/40 bg-[--bg-surface] p-5">
  <p className="text-xs font-medium uppercase tracking-wide text-[--text-muted]">
    PR-AUC · unseen customers
  </p>
  <p className="mt-2 text-[28px] font-extrabold tabular-nums
    text-[--text-primary]">0.900</p>
  <p className="mt-1 text-xs text-emerald-400">computed from this file</p>
</figure>
```
Source honesty lives in the footer line — swap `text-emerald-400` for
`text-amber-400` and "static hold-out" text when the source is a trained
report, so the dashboard never overclaims.

**Seen-vs-Unseen chart** (Plotly/React): two bars, `--accent` (#1565C0) for
Seen and band-review (`#F97316`) for Unseen; title "PR-AUC: seen vs unseen
customers".

## 5. Interactive states & motion

| element | idle | hover | active/focus | disabled |
|---|---|---|---|---|
| Primary btn | `bg-[--accent]` | `bg-blue-500` | `translate-y-px` | `opacity-50 cursor-not-allowed` |
| Dropzone | dashed slate | `border-[--accent] bg-[--accent-soft]` | `ring-2 ring-[--accent]/30` | — |
| Risk badge | solid band | `brightness-110` | — | — |
| Txn card | surface | `bg-[--bg-elevated]` (summary row) | border brightens `border-slate-500` | — |

Motion (respect `prefers-reduced-motion`):
- Alert badge: `animate-pulse` (Tailwind built-in) on high-risk only.
- Spinner on Analyse: `animate-spin`.
- Card expand: `transition` + `group-open:bg-[--bg-elevated]`.
- Page section entry: subtle `transition` on mount, 150 ms, fade + 4 px rise
  (`opacity-0 translate-y-1` → `opacity-100 translate-y-0`).

## 6. Light theme variant (optional)

Swap only the surface tokens; risk bands stay identical. Table order tells
the story: light backgrounds (`#F7F9FC` base, white surfaces, `#E2E8F0`
borders, `#0F172A` text); accent stays `#2563EB`-family for contrast on
white.

## 7. Shadow and depth

```css
/* two-step elevation: flat cards, elevated interactions */
--shadow-card: 0 1px 2px rgba(2,6,23,.4);
--shadow-float: 0 8px 24px -6px rgba(2,6,23,.6);
```
Apply `--shadow-card` to all `.rounded-xl` surfaces; `--shadow-float` to
sticky nav, modal, and drag-over dropzone. No borders-only depth, no
bleeding shadows.

---

## 8. Mapping to the live Streamlit app

The shipping dashboard is **Streamlit** (`dashboard/app.py`, Python) — the
exact same three-act page, colours, risk badges and metrics, rendered
natively. React is a possible re-skin; it is not the current runtime. To
port the design language today, in `dashboard/app.py`:

- **Palette** already lives in `RISK_COLORS` (lines 66–67):
  `#2e7d32` normal, `#f9a825` monitor, `#ef6c00` review, `#c62828`
  high-risk. The badge helper `_band_badge()` (line 486) renders the pill
  chips from Section 4 verbatim.
- **Hero** = `_render_hero()` (line 930): title, sub-headline, decision
  matrix heatmap (`px.imshow`, `["#08306b","#7fcdbb","#ffffcc"]`),
  and the four Load/Score/Decide/Explain steps.
- **Upload & Analyse** = `panel_upload()`: `selectbox` (backend),
  `file_uploader` (dropzone), primary `st.button("Analyse")`, plus the
  budget slider (`_render_upload_results`, line 763).
- **Txn cards** = `_render_transaction_cards()`: `st.progress` bars,
  badge chips, and `top_contributing_features` reasons inside
  `st.expander` — the exact composition of the React card above.
- **Dynamic System Evaluation** = `_render_system_eval()`: caption
  "Computed from `<file>`", 4-up `st.metric` grid, seen-vs-unseen
  `px.bar`, and the leakage checklist `_leakage_checklist()`.
- **Global dark look** — add a single `.streamlit/config.toml`:

```toml
[theme]
base = "dark"
primaryColor = "#3B82F6"
backgroundColor = "#0B1020"
secondaryBackgroundColor = "#141A30"
textColor = "#EAF0FF"
font = "sans serif"
```

Follow-up Tailwind/React polish would be a *port*, never a redesign: the
components, tokens and interaction states in this guide are the source of
truth for both renderers.

---

*Design tokens verified against the shipped dashboard (`RISK_COLORS`, the
`_band_badge` chip markup, and the three-section layout).*