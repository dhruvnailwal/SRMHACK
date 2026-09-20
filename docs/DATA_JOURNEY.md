# Inside the Application: The Data Journey & Output Glossary

### Bridging the frontend and backend of *Unseen-Customer Fraud Detection*

This document explains, in plain English, exactly what happens from the
moment an analyst uploads a `.csv` file and clicks **Analyse**, all the way
through to the four outputs that appear on the dashboard cards — and what a
human analyst should *do* with those results.

No prior knowledge is assumed. When an important technical word appears, its
plain meaning is given right next to it.

---

## Table of contents

1. **The Data Journey: How Results are Calculated**
   - 1.1 The one-screen mental model
   - 1.2 Step 1 — Upload & validate the file
   - 1.3 Step 2 — Sort chronologically, open the stream
   - 1.4 Step 3 — Score one transaction at a time (the streaming pipeline)
   - 1.5 The two signals, and why they are "behavioural" not "memorised"
   - 1.6 How the alert budget shapes the final decision
   - 1.7 Where the numbers are computed: dashboard vs FastAPI
2. **Output Glossary: Understanding the Results**
   - 2.1 Fraud Probability (0.00 – 1.00)
   - 2.2 Novelty Score (0.00 – 1.00)
   - 2.3 Risk Band (the four colours)
   - 2.4 Top Reasons (the plain-language explanation)
   - 2.5 Bringing them together: one worked example

---

## 1. The Data Journey: How Results are Calculated

### 1.1 The one-screen mental model

Picture the application as a **conveyor belt with one analyst sitting at the
end**.

You (or your analyst) drop a box of transactions onto the belt. The system
reads each transaction **one at a time, in clock order**, checks it, stamps
it with two numbers, and sorts it into a coloured bin. At the end, the
analyst sees a card for each unusual transaction: two numbers, a colour, and
a list of *why*.

Every transaction passes through the same narrow pipe:

```
read one row  →  validate  →  build features  →  fraud probability
              →  novelty score  →  risk band  →  reasons  →  commit to history
```

That pipe is the same whether you use the web dashboard, the FastAPI
service, or the command-line tool. This is deliberate: **the number on the
screen is the number a live-deployed system would produce**.

### 1.2 Step 1 — Upload & validate the file

When you drop a `.csv` and press **Analyse**:

1. The dashboard saves the upload to a temporary file and opens it with a
   tolerant loader (`load_dataframe`). The loader accepts common column
   aliases — for example `timestamp` / `event_time` / `time`, or
   `customer_id` / `customer_key` / `cid` — and maps them onto one canonical
   schema. If a canonical column is entirely missing, the file is rejected
   with a clear error rather than silently mis-scored.
2. The required columns are: `transaction_id`, `timestamp`, `amount`,
   `customer_id`, `merchant_id`, `device_id` (+ optional `is_fraud` for
   evaluation).
3. Numeric columns are cleaned (`valid_numeric`), and the file is kept in
   memory as a table.

> **What the analyst sees before scoring:** a schema-mapping summary and a
> preview — so a wrong column name is caught *before* the model runs, not
> silently swallowed.

### 1.3 Step 2 — Sort chronologically, open the stream

Fraud detection must never use the future to judge the present. So the rows
are **sorted by timestamp** and then replayed one-by-one through a stateful
engine called the **streaming processor** (`StreamingProcessor`).

The processor holds a running "picture" of the world — how many times each
customer, merchant, and device has appeared, how much money they normally
move, when they usually transact. This picture is called the **streaming
state**, and it grows as transactions pass through.

- When a *later* transaction for the same customer arrives, the engine
  already knows about the *earlier* ones.
- When a *new* customer arrives, the engine has **zero history** for them —
  and this is exactly the situation the whole system is designed for (see
  1.5).

### 1.4 Step 3 — Score one transaction at a time

For each transaction, the pipeline runs five quick sub-steps:

**a) Validate.** The engine checks the row is scoreable: a positive amount,
a timestamp, and a non-empty customer / merchant / device identifier. Bad
rows are reported as invalid instead of crashing the run.

**b) Build features (the "behavioural fingerprint").** A pure function
(`FeatureEngine.prepare_row`) reads the current streaming state and turns
*this one transaction* into ~39 numbers. Examples:

- **Velocity** — how many transactions in the last 5 minutes / 30 minutes /
  24 hours, and how much money moved;
- **Amount pattern** — a z-score (how many standard deviations away from the
  customer's *own* normal amount is this one?);
- **Cold-start shrinkage** — when history is thin, the customer's own
  average is blended toward a **segment prior** (the average for customers
  at the same "history level"), so a first-ever transaction is scored with
  statistical caution, not zero information;
- **Time of day** — is this hour unusual for this customer?
- **Device / merchant history** — has this device or merchant been seen
  before?

Crucially, **these features describe behaviour, not identity**. The raw
`customer_id` / `merchant_id` / `device_id` values are only *keys* used to
look up history — they are **never** fed to the model as features. The model
cannot "memorise that card number X is fraud"; it can only react to
behavioural patterns.

**c) Signal 1 — Fraud probability.** The LightGBM classifier reads the 39
features and outputs a raw score. That raw score is passed through a
**calibrator** (isotonic regression, fitted on *validation* data only) so
that the number becomes a true probability: `0.80` genuinely means "an
80% chance this is fraud," not just "high-ish score."

**d) Signal 2 — Novelty score.** A completely separate, unsupervised model
(IsolationForest) trained only on *legitimate* history — it has never seen a
fraud label. It asks a different question: *"how unlike ordinary traffic is
this behaviour?"* Its raw anomaly output is converted by an ECDF
(rank-averaging against the legit reference) into a 0–1 **novelty_score**:
`0.90` means "more atypical than 90% of normal transactions." This signal
needs **no memory of who the customer is**, which is what makes it
cold-start resilient.

**e) Commit.** After scoring, the row's details are added to the streaming
state (`commit_row`). This ordering — **score first, commit last** — is the
technical guarantee that no transaction ever sees its own future.

### 1.5 The two signals, and why they are behavioural

| signal | trained how? | answers | needs a memory of *this* identity? |
|---|---|---|---|
| **Fraud probability** | supervised, on labelled fraud/legit | "How likely is this *specific* behaviour to be fraud?" | no — works from behaviour features + priors |
| **Novelty score** | unsupervised, on legit traffic only | "How unusual is this *behaviour* vs the normal baseline?" | no — identity-free by construction |

Because both signals are behaviour-based, a never-seen customer's very
first transaction is scored **exactly like a veteran customer's** — from
priors and statistical instincts rather than memorised card numbers. This is
the system's core answer to *cold-start*: it never panics on new identities,
and it never relies on having seen you before.

### 1.6 How the alert budget shapes the final decision

The two signals alone are not enough — they could, in theory, flag half the
world. The system caps the flood with **an analyst budget**:

- The operator picks a budget — default **2%** (slider from 0.5% to 10%) —
  meaning "I can realistically action about 2% of transactions."
- The two signals are combined by a **risk matrix** into a *risk band*,
  and a **rank score** (0.7 × probability + 0.3 × novelty) sorts every
  row from most to least dangerous.
- The dashboard takes **the top N% by risk rank** and re-derives the
  Review / High-Risk *probability thresholds* as the **quantiles** that
  produce exactly that volume:

```
budget = 2%  →  N = 2% of rows (say 40 of 2,000)
             →  the probability of the 98th-percentile row becomes
                "p_review" (the floor for the Review band)
             →  of those 40 slots, at most 70% may be High-Risk
                (high_risk_share_cap_pct); the probability of that
                boundary row becomes "p_high" (the floor for High-Risk)
```

A **streaming budget controller** (`AlertBudgetController`) additionally hard
caps how many alerts can be escalated per hour and per day, so even a
malicious burst cannot blow past the analyst's capacity. The result is a
dashboard where moving the slider **dynamically re-derives** the thresholds
and re-colours the bands — no re-training, ever.

### 1.7 Where the numbers are computed: dashboard vs FastAPI

Two identical pipelines, same semantics:

- **In-process (default):** the dashboard loads the trained
  `model_primary.joblib`, `calibrator.joblib`, `novelty.joblib` and
  thresholds at startup, builds a `StreamingProcessor`, and scores the file
  in-process. Zero infrastructure; a 2,000-row file scores in seconds.
- **Via FastAPI (optional):** the dashboard posts the rows to
  `POST /predict/batch`. The service builds a *fresh* processor per request
  and replays the file chronologically, returning the same decisions. The
  service's `POST /predict` scores a single transaction with the identical
  code path.

Because `StreamingProcessor.score_transaction` is the single production
function used by the UI, the API, and the CLI, every path returns
**bit-for-bit the same outputs** for the same input. There is no
"dashboard-only" or "API-only" logic to drift apart.

---

## 2. Output Glossary: Understanding the Results

Every scored transaction card shows four key pieces of information.

### 2.1 Fraud Probability (0.00 – 1.00)

**What it is.** A **calibrated** likelihood that this transaction is fraud.
Calibrated means the number is trustworthy in a literal way: across many
rows where the model says `0.70`, roughly 70% actually turn out to be
fraud. (Measured calibration quality: Brier 0.0049, expected calibration
error 0.0018 — essentially "what the model says ≈ what happens".)

**How a human should read it.**

| value | read |
|---|---|
| `0.00 – 0.15` | ordinary; baseline risk for most traffic |
| `0.15 – 0.50` | elevated — worth a look |
| `0.50 – 0.90` | strongly suspicious |
| `> 0.90` | near-certain under the model's view |

**Interpretation tip:** treat the *ordering* of rows as more important than
a single number. A transaction ranked #1 by probability is the one the
model considers most fraudulent in this whole file — that is the one to
review first.

### 2.2 Novelty Score (0.00 – 1.00)

**What it is.** How **unusual the behaviour** is compared with ordinary,
legitimate traffic. `0.90` means "more atypical than 90% of normal
transactions." It is computed by an unsupervised anomaly model
(IsolationForest) trained only on non-fraud history and normalised against
a legitimate baseline, so it measures *oddness*, not *guilt*.

**How a human should read it.**

| value | read |
|---|---|
| `0.00 – 0.70` | within the normal range |
| `0.70 – 0.90` | atypical — new device, new merchant, unusual timing |
| `> 0.90` | very atypical — e.g. 5 transactions from a device in 5 minutes |

**The critical nuance:** a **high novelty score alone is NOT evidence of
fraud.** New customers are inherently unusual. The system deliberately
refuses to alert on novelty alone — the novelty channel is gated so it can
only *add* suspicion on top of fraud probability. So when you see a high
novelty score, ask: *"is the fraud probability also elevated?"* If yes, this
is a red flag. If no, it is probably a legitimately new user going about
their business.

### 2.3 Risk Band (the final call)

The risk band is produced by combining fraud probability and novelty in a
decision matrix, then holding the result inside the **alert budget**. The
live system uses **four** bands (three with alert consequences):

| band | colour | fraud p | novelty | analyst action |
|---|---|---|---|---|
| `normal` | **green** | low | low | allow |
| `monitor` | **amber** | low | high (novel but p low) | log only, no alert |
| `review` | **orange** | elevated (≥ `p_review`) | — | alert → review |
| `high-risk` | **red** | high (≥ `p_high`) | typically high | alert → investigate immediately |

> *Why four, not three?* The amber `monitor` band exists so the system still
> *looks* at unusual-but-likely-legitimate behaviour (new customers!) without
> punishing them. Removing it would turn every new customer into an alert —
> exactly the false-alarm trap the project is designed to avoid.

**How the budget controls the bands.** The `p_review` and `p_high`
boundaries are **not fixed constants** — they are re-derived as quantiles so
that the orange+red volume matches the operator's chosen alert budget.
Spending *up* the slider (say 2% → 5%) lowers the bars and more
transactions become orange/red; spending *down* raises the bars and fewer
do. The colour you see is therefore always "the top N% by risk" *and* the
hard per-hour/per-day caps run on top as final backstops.

**Interpretation tip:** "normal + high novelty" is a *watch* case
(amber), not an alert. "Review or high-risk" is a *today* case.

### 2.4 Top Reasons (the plain-language explanation)

**What it is.** Each alert names its **top contributing factors** — the
behaviours that pushed the score up — grouped into human families and
phrased in plain language. The contributions are exact TreeSHAP values
(per-feature contribution to the fraud probability) summed over *feature
families* so that, for example, five velocity columns collapse into one
sentence about velocity.

**Where the phrases come from** (examples seen on real dashboards):

| family | example reason |
|---|---|
| velocity | "Velocity elevated: 4 txns in last 5 min, 6 in 30 min, 9 in 24h (increases risk)." |
| amount | "Amount is 2.4 sd above the historical baseline (z=2.4) (increases risk)." |
| time | "Transaction at 03:00 (night-time) (increases risk)." |
| device | "The device has not been observed previously (increases risk)." or "Device seen 12 time(s) before; behaviour increases risk." |
| merchant | "The merchant has not been observed previously (increases risk)." |
| history | "First transaction for this customer; using segment-level baseline (increases risk)." |

**How a human should read it.** Reasons are **contributions, not causes** —
they say "these factors pushed the probability up," never "this is why it's
fraud." Use them to (1) sanity-check the decision, (2) write the analyst's
own investigation note, and (3) spot *novel-pattern* cases: when velocity,
device-new, and merchant-new all appear together on a brand-new identity, a
card-around-the-clock pattern is underway.

### 2.5 Bringing them together: one worked example

Suppose the dashboard shows:

```
fraud probability  0.87        ← calibrated 87% suspect
novelty score      0.93        ← more atypical than 93% of normal traffic
risk band          high-risk   ← red
top reasons        · Velocity elevated: 4 txns in last 5 min
                   · Amount is 2.4 sd above the historical baseline
                   · The merchant has not been observed previously
```

**The read:** a new merchant, a burst of four rapid transactions, and an
amount wildly above the customer's norm — two independent models agree this
is unusual *and* fraud-like; the budget allocated a "red" slot to it. Action:
review now, contact the customer, and inspect the other three transactions
in the same 5-minute window. The explanations give you the talking points;
the numbers tell you how confidently the system believes it.

---

*Written against the live codebase: `src/streaming_processor.py`
(`score_transaction`, `score_batch`, `AlertBudgetController`),
`src/feature_engineering.py` (`FeatureEngine.prepare_row`),
`src/novelty_model.py` (IsolationForest + ECDF), `src/explainability.py`
(TreeSHAP reason groups), `dashboard/app.py` (upload → analyse →
recompute_bands), and `api/main.py` (`/predict`, `/predict/batch`).*