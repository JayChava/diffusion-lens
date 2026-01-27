# GenAI Session Analyzer — CLAUDE.md

## Persona

You are a **Staff Data Architect** helping me build a production-grade portfolio project in 48 hours. I'm interviewing for a **Lead Data/Strategy role at Luma AI** (high-growth GenAI startup). Due to NDAs, I cannot show actual workplace code—this "Clean Room" project demonstrates my engineering instincts.

**Your communication style:**
- Direct, opinionated, and pragmatic
- Favor speed-to-demo over perfection
- Call out over-engineering immediately
- Suggest the "80/20 solution" first, then mention the "production version" as context
- When I'm stuck, give me the code—don't just describe it

---

## Project Mission

Build the **GenAI Session Analyzer**: a local data platform that ingests real prompt data (DiffusionDB from HuggingFace), enriches it with synthetic telemetry (latency, errors, user cohorts), and exposes a dashboard showing **User Friction** and **Session Costs**.

**Interview Alignment (what Luma evaluators care about):**
1. Clear understanding of requirements and stakeholder needs
2. Strong data modeling and transformation instincts
3. Thoughtful pipeline design and orchestration choices
4. Balance of engineering rigor with analytical usability
5. Clear communication of complex systems through code

---

## Tech Stack (Speed-Optimized for 48hrs)

| Layer | Tool | Why |
|-------|------|-----|
| **Ingestion** | `datasets` (HuggingFace) | One-liner to stream DiffusionDB |
| **Storage** | DuckDB | Local OLAP, zero config, SQL-native, parquet-friendly |
| **Image Storage** | Local filesystem | `/data/images/` with FK references in DuckDB |
| **Orchestration** | Dagster | Modern, asset-based, beautiful local UI, interview-impressive |
| **Transformation** | dbt-duckdb | Industry-standard modeling, star schema, data tests |
| **ML Enrichment** | MLX + sentence-transformers | LLM prompt analysis + semantic search embeddings |
| **Simulation** | Faker + NumPy | Fast synthetic data generation with realistic distributions |
| **Dashboard** | Streamlit | Fast to build, can display images alongside metrics |
| **Environment** | uv + pyproject.toml | Modern Python packaging, fast dependency resolution |

**Not using (and why):**
- Airflow: Overkill for local, slower to set up
- Spark: Unnecessary scale for this data volume
- Postgres: DuckDB is faster for analytics and zero-config
- HuggingFace Transformers: MLX is 3-4x faster on Apple Silicon

---

## Data Architecture

### Source Data
```
DiffusionDB (HuggingFace) — 10K samples
├── prompt (text)
├── image (PIL Image)
├── seed (int)
├── cfg (float)
├── sampler (string)
└── width/height (int)
```

### Storage Layout
```
data/
├── warehouse.duckdb              # All metadata, embeddings, analysis
└── blob/                         # Simulated blob storage (like S3)
    └── images/
        └── generations/          # Images from DiffusionDB
            ├── gen_00001.png
            ├── gen_00002.png
            └── ... (10K images, ~5GB)
```

**Production note:** Local `blob/` directory simulates S3. In production, swap `Path(image_path)` for `s3.get_object()`. Same pattern, different backend.

### Dimensional Model (Star Schema)

```
┌─────────────────┐      ┌─────────────────────────────────────┐
│   dim_users     │      │          dim_prompts                │
├─────────────────┤      ├─────────────────────────────────────┤
│ user_id (PK)    │      │ prompt_id (PK)                      │
│ user_tier       │      │ prompt_text                         │
│ signup_date     │      │ image_path                          │
│ cohort_week     │      │                                     │
│ region          │      │ -- LLM-extracted (pre-computed) --  │
│ device_type     │      │ subject                             │
└────────┬────────┘      │ art_style                           │
         │               │ mood                                │
         │               │ complexity_score (1-5)              │
         │               │ has_nsfw_intent (bool)              │
         │               │ setting                             │
         │               │ lighting                            │
         │               │                                     │
         │               │ -- CLIP (pre-computed) --           │
         │               │ image_embedding (FLOAT[512])        │
         │               │ text_embedding (FLOAT[512])         │
         │               │ alignment_score (prompt vs image)   │
         │               │ style_cluster (k-means cluster ID)  │
         │               │                                     │
         │               │ -- Semantic Search (pre-computed) --│
         │               │ text_embedding_mini (FLOAT[384])    │
         │               └────────────────┬────────────────────┘
         │                                │
         │    ┌───────────────────────────┘
         │    │
         ▼    ▼
┌─────────────────────────────────────────────┐
│              fct_generations                │
├─────────────────────────────────────────────┤
│ generation_id (PK)                          │
│ user_id (FK)                                │
│ prompt_id (FK)                              │
│ session_id                                  │
│ timestamp                                   │
│ latency_ms                                  │
│ status (success/timeout/safety_violation/   │
│         rate_limited/model_error)           │
│ cost_credits                                │
│ model_version                               │
│ retry_count                                 │
└─────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│              fct_sessions (aggregated)      │
├─────────────────────────────────────────────┤
│ session_id (PK)                             │
│ user_id (FK)                                │
│ session_start                               │
│ session_end                                 │
│ total_generations                           │
│ success_rate                                │
│ avg_latency_ms                              │
│ total_cost_credits                          │
│ friction_score (derived)                    │
│ churned_after (boolean)                     │
└─────────────────────────────────────────────┘
```

### Key Business Metrics
- **Friction Score**: Weighted composite of (error_rate × 3) + (avg_latency_normalized × 2) + (retry_rate × 1)
- **Session Cost**: Sum of cost_credits per session (based on prompt tokens + generation time)
- **Churn Indicator**: No activity for 7+ days after session
- **Alignment Score**: CLIP cosine similarity between prompt text and generated image (0-1)
- **Style Clusters**: K-means groupings of image embeddings for visual categorization
- **Semantic Search**: Find similar prompts using sentence-transformer embeddings (RAG pattern)

---

## Simulation Logic Strategy

### 1. User Generation
```python
# Cohort distribution (realistic power law)
user_tiers = {
    'free': 0.70,      # Casual users
    'pro': 0.25,       # Regular users
    'enterprise': 0.05 # Power users
}
```

### 2. Status Determination (from prompt text)
```python
def determine_status(prompt_text: str, user_tier: str) -> str:
    """
    Heuristics based on prompt characteristics:
    """
    prompt_lower = prompt_text.lower()
    token_count = len(prompt_text.split())

    # Safety violation signals (NSFW keywords, violent content)
    nsfw_keywords = ['nude', 'naked', 'nsfw', 'explicit', ...]
    if any(kw in prompt_lower for kw in nsfw_keywords):
        return 'safety_violation' if random.random() < 0.85 else 'success'

    # Timeout signals (very long/complex prompts)
    if token_count > 75:
        return 'timeout' if random.random() < 0.15 else 'success'

    # Rate limiting (free tier + high frequency)
    if user_tier == 'free':
        return 'rate_limited' if random.random() < 0.08 else 'success'

    # Random model errors (baseline noise)
    if random.random() < 0.02:
        return 'model_error'

    return 'success'
```

### 3. Latency Distribution
```python
def generate_latency(prompt_text: str, status: str) -> int:
    """Realistic latency based on prompt complexity and outcome"""
    base_latency = len(prompt_text.split()) * 50  # ~50ms per token

    if status == 'timeout':
        return 30000  # 30s timeout
    elif status == 'safety_violation':
        return random.randint(100, 500)  # Fast rejection
    else:
        # Log-normal distribution (realistic API latency)
        noise = np.random.lognormal(0, 0.5)
        return int(base_latency * noise)
```

### 4. Session Logic
```python
# Session = prompts from same user within 30-min window
SESSION_TIMEOUT_MINUTES = 30

# Churn probability based on friction
def calculate_churn_probability(friction_score: float, user_tier: str) -> float:
    base_churn = {'free': 0.40, 'pro': 0.15, 'enterprise': 0.05}
    return min(0.95, base_churn[user_tier] * (1 + friction_score))
```

---

## ML Enrichment Layer (Pre-computed with MLX)

**Strategy:** Run ML models once on Saturday night, store results as columns. Zero model loading during demo—just fast SQL queries.

### Setup (M5 MacBook Pro)
```bash
# Install MLX ecosystem
pip install mlx mlx-lm mlx-clip

# Models will auto-download on first run (~3GB total)
```

### Time Budget for 10K Items

| Task | Per Item | 10K Total |
|------|----------|-----------|
| Download images from HuggingFace | ~50ms | ~10 min |
| Save images to disk | ~20ms | ~3 min |
| CLIP embeddings (MLX) | ~30ms | ~5 min |
| LLM prompt analysis (MLX) | ~300ms | ~50 min |
| **Total** | | **~70 min** |

### 1. CLIP Embeddings (Image + Text)

**Model:** `mlx-community/clip-vit-base-patch32` (~400MB)

**What it produces:**
- `image_embedding`: 512-dim vector for the generated image
- `text_embedding`: 512-dim vector for the prompt text
- `alignment_score`: Cosine similarity (did model follow the prompt?)

```python
# precompute_clip.py
from mlx_clip import load, image_encoder, text_encoder
from PIL import Image
import numpy as np

model = load("mlx-community/clip-vit-base-patch32")

def compute_clip_features(image_path: str, prompt_text: str) -> dict:
    image = Image.open(image_path)

    image_emb = image_encoder(model, image)
    text_emb = text_encoder(model, prompt_text)

    # Normalize and compute alignment
    image_emb = image_emb / np.linalg.norm(image_emb)
    text_emb = text_emb / np.linalg.norm(text_emb)
    alignment = np.dot(image_emb, text_emb)

    return {
        'image_embedding': image_emb.tolist(),
        'text_embedding': text_emb.tolist(),
        'alignment_score': float(alignment)
    }
```

**What you can query:**
```sql
-- Images with poor prompt adherence
SELECT prompt_text, image_path, alignment_score
FROM dim_prompts
WHERE alignment_score < 0.2
ORDER BY alignment_score;

-- Find similar images (cosine similarity)
SELECT b.prompt_id, b.image_path,
       list_cosine_similarity(a.image_embedding, b.image_embedding) AS similarity
FROM dim_prompts a, dim_prompts b
WHERE a.prompt_id = 42 AND b.prompt_id != 42
ORDER BY similarity DESC
LIMIT 5;
```

### 2. LLM Prompt Analysis

**Model:** `mlx-community/Qwen2.5-1.5B-Instruct-4bit` (~1GB)

**What it extracts:**
- `subject`: Main subject (person, landscape, object, etc.)
- `art_style`: Visual style (photorealistic, anime, oil painting, etc.)
- `mood`: Emotional tone (dark, cheerful, mysterious, etc.)
- `complexity_score`: 1-5 based on prompt detail level
- `has_nsfw_intent`: Boolean safety flag
- `setting`: Location/environment
- `lighting`: Lighting description if present

```python
# precompute_llm.py
from mlx_lm import load, generate
import json

model, tokenizer = load("mlx-community/Qwen2.5-1.5B-Instruct-4bit")

EXTRACTION_PROMPT = """Analyze this image generation prompt. Return ONLY valid JSON.

Prompt: "{prompt}"

{{"subject": "main subject", "style": "art style", "mood": "tone", "complexity": 1-5, "has_nsfw_intent": true/false, "setting": "location", "lighting": "lighting type"}}"""

def analyze_prompt(prompt_text: str) -> dict:
    prompt = EXTRACTION_PROMPT.format(prompt=prompt_text)
    response = generate(model, tokenizer, prompt=prompt, max_tokens=100)

    try:
        json_str = response[response.find('{'):response.rfind('}')+1]
        return json.loads(json_str)
    except:
        return {
            "subject": "unknown", "style": "unknown", "mood": "unknown",
            "complexity": 3, "has_nsfw_intent": False, "setting": "unknown", "lighting": "unknown"
        }
```

**What you can query:**
```sql
-- Do complex prompts timeout more?
SELECT complexity_score,
       AVG(latency_ms) AS avg_latency,
       SUM(CASE WHEN status = 'timeout' THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS timeout_rate
FROM fct_generations g
JOIN dim_prompts p USING (prompt_id)
GROUP BY complexity_score;

-- What styles do power users prefer?
SELECT u.user_tier, p.art_style, COUNT(*) AS count
FROM fct_generations g
JOIN dim_users u USING (user_id)
JOIN dim_prompts p USING (prompt_id)
GROUP BY u.user_tier, p.art_style
ORDER BY u.user_tier, count DESC;

-- NSFW attempt rate by user tier
SELECT user_tier,
       SUM(CASE WHEN has_nsfw_intent THEN 1 ELSE 0 END) AS nsfw_attempts,
       ROUND(100.0 * SUM(CASE WHEN has_nsfw_intent THEN 1 ELSE 0 END) / COUNT(*), 2) AS nsfw_rate_pct
FROM fct_generations g
JOIN dim_users u USING (user_id)
JOIN dim_prompts p USING (prompt_id)
GROUP BY user_tier;
```

### 3. Style Clustering (Post-CLIP)

After CLIP embeddings are computed, run k-means to group images by visual style:

```python
from sklearn.cluster import KMeans
import numpy as np

# Load embeddings from DuckDB
embeddings = duckdb.sql("SELECT prompt_id, image_embedding FROM dim_prompts").df()
X = np.array(embeddings['image_embedding'].tolist())

# Cluster into 8 style groups
kmeans = KMeans(n_clusters=8, random_state=42)
clusters = kmeans.fit_predict(X)

# Update DuckDB
for prompt_id, cluster in zip(embeddings['prompt_id'], clusters):
    duckdb.sql(f"UPDATE dim_prompts SET style_cluster = {cluster} WHERE prompt_id = {prompt_id}")
```

**What you can query:**
```sql
-- Which style clusters have highest friction?
SELECT p.style_cluster,
       COUNT(*) AS generations,
       AVG(s.friction_score) AS avg_friction
FROM fct_generations g
JOIN dim_prompts p USING (prompt_id)
JOIN fct_sessions s USING (session_id)
GROUP BY p.style_cluster
ORDER BY avg_friction DESC;
```

### Production Alternative (Interview Talking Point)

> "For local development, I used MLX—it's optimized for Apple Silicon and let me pre-compute 10K enrichments in about an hour.
>
> In production, I'd swap to **vLLM** or **HuggingFace TGI** on GPU clusters for self-hosted, or use **Claude API / Amazon Bedrock** for managed inference. The enrichment logic stays the same—only the inference backend changes."

---

## SQL Copilot (Live Demo Feature)

**What it does:** User types a natural language question → Local LLM generates SQL → Human reviews → Execute against DuckDB.

**Why it's impressive:** Shows real-time LLM integration, not just pre-computed features.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit UI                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ "Show me error rate for power users"                │   │
│  └─────────────────────────┬───────────────────────────┘   │
│                            │                                │
│                            ▼                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ construct_prompt(question, schema)                  │   │
│  │ Injects table schema so LLM knows column names      │   │
│  └─────────────────────────┬───────────────────────────┘   │
│                            │                                │
│                            ▼                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ MLX + Qwen2.5-1.5B-Instruct                         │   │
│  │ Generates SQL string (~0.5s on M5)                  │   │
│  └─────────────────────────┬───────────────────────────┘   │
│                            │                                │
│                            ▼                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ st.text_area (editable)                             │   │
│  │ SELECT user_tier, AVG(error_rate)...                │   │
│  │                                    [Run Query]      │   │
│  └─────────────────────────┬───────────────────────────┘   │
│                            │                                │
│                            ▼                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ DuckDB executes → Results displayed                 │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 1. Schema Extractor

```python
# src/copilot/schema.py

import duckdb

def get_duckdb_schema(con: duckdb.DuckDBPyConnection, table_name: str = "fct_generations") -> str:
    """
    Extract table schema and format for LLM consumption.
    """
    schema_df = con.execute(f"DESCRIBE {table_name}").fetchdf()

    # Format as CREATE TABLE (LLMs understand this well)
    columns = []
    for _, row in schema_df.iterrows():
        col_name = row['column_name']
        col_type = row['column_type']
        columns.append(f"    {col_name} {col_type}")

    schema_text = f"CREATE TABLE {table_name} (\n"
    schema_text += ",\n".join(columns)
    schema_text += "\n);"

    return schema_text


def get_full_schema(con: duckdb.DuckDBPyConnection) -> str:
    """Get schema for all tables."""
    tables = ['dim_users', 'dim_prompts', 'fct_generations', 'fct_sessions']
    schemas = [get_duckdb_schema(con, t) for t in tables]
    return "\n\n".join(schemas)
```

### 2. Prompt Template

```python
# src/copilot/prompt.py

def construct_prompt(user_question: str, schema_text: str) -> str:
    """
    Construct prompt optimized for SQL generation.
    Format works well with SQLCoder, Llama-3, Qwen2.5.
    """
    prompt = f"""### Task
Generate a SQL query to answer the following question:
{user_question}

### Database Schema
The query will run on a DuckDB database with the following schema:
{schema_text}

### Instructions
- Use DuckDB SQL syntax (LIMIT not TOP, ILIKE for case-insensitive)
- Return only the SQL query, no explanations
- Do not wrap in markdown code blocks

### Answer
SELECT"""

    return prompt
```

### 3. LLM Integration (MLX)

```python
# src/copilot/llm.py

from mlx_lm import load, generate

# Load model once at app startup (cache in st.session_state)
@st.cache_resource
def load_copilot_model():
    model, tokenizer = load("mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    return model, tokenizer


def get_llm_response(prompt: str, model, tokenizer) -> str:
    """Generate SQL from prompt using MLX."""
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=200,
        temp=0.1  # Low temp for deterministic SQL
    )
    return clean_response(response)


def clean_response(response: str) -> str:
    """Clean LLM output to extract SQL."""
    sql = response.strip()

    # Remove markdown if present
    if sql.startswith('```'):
        sql = sql.split('```')[1]
        if sql.startswith('sql'):
            sql = sql[3:]

    # Prompt ends with "SELECT", so prepend if missing
    if not sql.upper().startswith('SELECT'):
        sql = 'SELECT ' + sql

    # Ensure ends with semicolon
    if not sql.strip().endswith(';'):
        sql = sql.strip() + ';'

    return sql
```

### 4. Streamlit Integration

```python
# dashboard/app.py (Copilot Section)

import streamlit as st
import duckdb

# --- SQL Copilot Section ---
st.header("🤖 SQL Copilot")
st.caption("Ask questions in natural language, get SQL queries")

# Load model (cached)
model, tokenizer = load_copilot_model()

# Get schema (cached)
@st.cache_data
def get_schema():
    con = duckdb.connect("data/warehouse.duckdb", read_only=True)
    return get_full_schema(con)

schema_text = get_schema()

# User input
user_question = st.text_input(
    "Ask a question about your data:",
    placeholder="e.g., Show me error rate by user tier"
)

if user_question:
    # Generate SQL
    with st.spinner("Generating SQL..."):
        prompt = construct_prompt(user_question, schema_text)
        generated_sql = get_llm_response(prompt, model, tokenizer)

    # Editable SQL area
    st.subheader("Generated SQL")
    edited_sql = st.text_area(
        "Review and edit if needed:",
        value=generated_sql,
        height=150
    )

    # Run button
    col1, col2 = st.columns([1, 5])
    with col1:
        run_button = st.button("▶️ Run Query", type="primary")

    if run_button:
        try:
            # Validate (block DROP, DELETE, etc.)
            sql_upper = edited_sql.upper()
            if any(kw in sql_upper for kw in ['DROP', 'DELETE', 'UPDATE', 'INSERT']):
                st.error("⚠️ Dangerous operation blocked. SELECT queries only.")
            else:
                # Execute
                con = duckdb.connect("data/warehouse.duckdb", read_only=True)
                result_df = con.execute(edited_sql).fetchdf()

                st.subheader("Results")
                st.dataframe(result_df, use_container_width=True)

                # Show row count
                st.caption(f"Returned {len(result_df)} rows")

        except Exception as e:
            st.error(f"Query error: {str(e)}")
```

### Example Queries to Demo

| Natural Language | Expected SQL |
|------------------|--------------|
| "Show me error rate by user tier" | `SELECT user_tier, AVG(CASE WHEN status != 'success' THEN 1 ELSE 0 END) AS error_rate FROM fct_generations JOIN dim_users USING (user_id) GROUP BY user_tier` |
| "Top 10 users by session cost" | `SELECT user_id, SUM(total_cost_credits) AS total_cost FROM fct_sessions GROUP BY user_id ORDER BY total_cost DESC LIMIT 10` |
| "What art styles have highest latency?" | `SELECT art_style, AVG(latency_ms) AS avg_latency FROM fct_generations JOIN dim_prompts USING (prompt_id) GROUP BY art_style ORDER BY avg_latency DESC` |
| "Count of generations per day" | `SELECT DATE_TRUNC('day', timestamp) AS day, COUNT(*) FROM fct_generations GROUP BY day ORDER BY day` |

### Performance on M5

| Metric | Value |
|--------|-------|
| Model load (first time) | ~3s |
| Model load (cached) | ~0.1s |
| SQL generation | ~0.3-0.5s |
| Total response time | **< 1s** |

### Safety Rails

```python
BLOCKED_KEYWORDS = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE']

def validate_sql(sql: str) -> tuple[bool, str]:
    """Block dangerous operations."""
    sql_upper = sql.upper()
    for kw in BLOCKED_KEYWORDS:
        if kw in sql_upper:
            return False, f"Blocked: {kw} not allowed"
    if not sql_upper.strip().startswith('SELECT'):
        return False, "Only SELECT queries allowed"
    return True, ""
```

### Interview Talking Point

> "The SQL Copilot runs a local LLM in real-time—no API calls, no latency. I inject the database schema into the prompt so the model knows the column names. The human reviews the SQL before execution, and I block any dangerous operations like DROP or DELETE. Generation takes about 300ms on my M5 Mac."

---

## Session Explorer (AI-Powered Search + Filters + Images)

**What it does:** Semantic search + LLM-generated filters + image preview. A full data exploration tool for GenAI sessions.

**The evolution:**
- ~~Basic semantic search~~ → **Full AI-powered explorer**
- Vector similarity + filter by style/mood/status + see actual images

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Session Explorer UI                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Search: "cyberpunk city"                                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐ │
│  │ Style:      │ │ Mood:       │ │ Status:     │ │ Tier:     │ │
│  │ [All     ▼] │ │ [All     ▼] │ │ [All     ▼] │ │ [All   ▼] │ │
│  │ anime       │ │ dark        │ │ success     │ │ free      │ │
│  │ realistic   │ │ cheerful    │ │ timeout     │ │ pro       │ │
│  │ painting    │ │ mysterious  │ │ safety      │ │ enterprise│ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘ │
│                        (LLM-generated labels)                   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Results: 847 similar prompts (showing top 12)            │  │
│  │                                                          │  │
│  │ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐         │  │
│  │ │  IMG    │ │  IMG    │ │  IMG    │ │  IMG    │         │  │
│  │ │         │ │         │ │         │ │         │         │  │
│  │ ├─────────┤ ├─────────┤ ├─────────┤ ├─────────┤         │  │
│  │ │neon     │ │blade    │ │future   │ │cyber    │         │  │
│  │ │tokyo... │ │runner...│ │city...  │ │woman... │         │  │
│  │ │Sim: 0.85│ │Sim: 0.82│ │Sim: 0.81│ │Sim: 0.79│         │  │
│  │ │✅ 2.3s  │ │✅ 1.9s  │ │⏱️ 31s   │ │🛑 safety│         │  │
│  │ └─────────┘ └─────────┘ └─────────┘ └─────────┘         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 📊 Cluster Insights                                      │  │
│  │ Error Rate: 23%  |  Avg Latency: 4.2s  |  Top Style: anime│  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Storage Layout (Simulated Blob Storage)

```
data/
├── warehouse.duckdb              # Metadata + embeddings
└── blob/                         # Simulated blob storage (like S3)
    └── images/
        └── generations/          # Generated images from DiffusionDB
            ├── gen_00001.png
            ├── gen_00002.png
            └── ... (10K images)
```

**Interview note:** "The images live in a local blob directory that simulates S3. In production, you'd swap `Path(image_path)` for `s3.get_object()`. Same pattern, different backend."

### Implementation

```python
# dashboard/app.py — Session Explorer Section

import streamlit as st
import duckdb
from sentence_transformers import SentenceTransformer
from pathlib import Path

st.header("🔍 Session Explorer")
st.caption("Semantic search + LLM-generated filters + image preview")

# --- Setup ---
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_connection():
    con = duckdb.connect("data/warehouse.duckdb", read_only=True)
    con.execute("LOAD vss;")
    return con

model = load_embedding_model()
con = get_connection()

# --- Get filter options from LLM-generated labels ---
@st.cache_data
def get_filter_options():
    styles = con.execute("SELECT DISTINCT art_style FROM dim_prompts WHERE art_style IS NOT NULL").fetchall()
    moods = con.execute("SELECT DISTINCT mood FROM dim_prompts WHERE mood IS NOT NULL").fetchall()
    statuses = con.execute("SELECT DISTINCT status FROM fct_generations").fetchall()
    tiers = con.execute("SELECT DISTINCT user_tier FROM dim_users").fetchall()
    return {
        'styles': ['All'] + [r[0] for r in styles],
        'moods': ['All'] + [r[0] for r in moods],
        'statuses': ['All'] + [r[0] for r in statuses],
        'tiers': ['All'] + [r[0] for r in tiers]
    }

filters = get_filter_options()

# --- Search Input ---
search_query = st.text_input(
    "Search prompts:",
    placeholder="e.g., cyberpunk city, fantasy castle, portrait"
)

# --- Filter Row (powered by LLM-extracted labels) ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    style_filter = st.selectbox("Art Style", filters['styles'])
with col2:
    mood_filter = st.selectbox("Mood", filters['moods'])
with col3:
    status_filter = st.selectbox("Status", filters['statuses'])
with col4:
    tier_filter = st.selectbox("User Tier", filters['tiers'])

# --- Search Logic ---
if search_query:
    # Embed query on-the-fly (~30ms)
    query_embedding = model.encode([search_query])[0].tolist()

    # Build dynamic WHERE clause from filters
    where_clauses = []
    if style_filter != 'All':
        where_clauses.append(f"p.art_style = '{style_filter}'")
    if mood_filter != 'All':
        where_clauses.append(f"p.mood = '{mood_filter}'")
    if status_filter != 'All':
        where_clauses.append(f"g.status = '{status_filter}'")
    if tier_filter != 'All':
        where_clauses.append(f"u.user_tier = '{tier_filter}'")

    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

    # DuckDB VSS search with filters (~2ms)
    results = con.execute(f"""
        SELECT
            p.prompt_id,
            p.prompt_text,
            p.image_path,
            p.art_style,
            p.mood,
            p.complexity_score,
            u.user_tier,
            g.latency_ms,
            g.status,
            array_cosine_similarity(p.text_embedding_mini, $1::FLOAT[384]) AS similarity
        FROM dim_prompts p
        JOIN fct_generations g USING (prompt_id)
        JOIN dim_users u USING (user_id)
        WHERE {where_sql}
        ORDER BY similarity DESC
        LIMIT 12
    """, [query_embedding]).fetchdf()

    # --- Cluster Insights ---
    st.subheader(f"Results for '{search_query}'")

    if len(results) == 0:
        st.warning("No results found. Try adjusting filters.")
    else:
        # Metrics row
        col1, col2, col3 = st.columns(3)
        error_rate = (results['status'] != 'success').mean()
        avg_latency = results['latency_ms'].mean()
        top_style = results['art_style'].mode().values[0] if len(results) > 0 else "N/A"

        col1.metric("Error Rate", f"{error_rate:.1%}")
        col2.metric("Avg Latency", f"{avg_latency:,.0f}ms")
        col3.metric("Top Style", top_style)

        st.divider()

        # --- Image Grid ---
        cols = st.columns(4)
        for idx, row in results.iterrows():
            with cols[idx % 4]:
                # Load image from local blob storage
                image_path = Path(row['image_path'])
                if image_path.exists():
                    st.image(str(image_path), use_container_width=True)
                else:
                    st.image("https://via.placeholder.com/256", use_container_width=True)

                # Status indicator
                status_emoji = {
                    'success': '✅',
                    'timeout': '⏱️',
                    'safety_violation': '🛑',
                    'rate_limited': '⚠️',
                    'model_error': '❌'
                }.get(row['status'], '❓')

                st.caption(f"**{row['prompt_text'][:35]}...**")
                st.caption(f"Sim: {row['similarity']:.2f} | {status_emoji} {row['latency_ms']:,}ms")
                st.caption(f"{row['art_style']} • {row['mood']}")
```

### Why This Is Powerful

| Feature | What It Shows | Business Value |
|---------|---------------|----------------|
| Semantic search | Vector embeddings + DuckDB VSS | "Find prompts like X" |
| LLM-generated filters | Pre-computed labels from Qwen | "Filter by style, mood, complexity" |
| Status filter | Telemetry integration | "Show me what's failing" |
| Image preview | Blob storage pattern | "Actually see the outputs" |
| Cluster metrics | Real-time aggregation | "This cluster has 40% error rate" |

### Example Demo Flow

```
You: "Let me show you the Session Explorer..."

[Types "cyberpunk city"]
[Selects: Status = "timeout"]

Results appear instantly with images:
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│ 🖼️      │ │ 🖼️      │ │ 🖼️      │ │ 🖼️      │
│neon     │ │future   │ │blade    │ │dystopia │
│tokyo... │ │city...  │ │runner...│ │mega...  │
│⏱️ 31s   │ │⏱️ 30s   │ │⏱️ 32s   │ │⏱️ 30s   │
└─────────┘ └─────────┘ └─────────┘ └─────────┘

Error Rate: 100%  |  Avg Latency: 30,750ms  |  Top Style: realistic

You: "All the timeouts for 'cyberpunk city' are realistic style prompts
      with high complexity. The model might be struggling with detailed
      cyberpunk scenes. Let me check if anime style does better..."

[Changes filter: Style = "anime"]

Error Rate: 12%  |  Avg Latency: 2,100ms

You: "Anime cyberpunk prompts have only 12% error rate.
      This tells us the issue is style-specific, not prompt-specific."
```

### Performance

| Step | Time |
|------|------|
| Embed search query | ~30ms |
| DuckDB VSS + filters | ~2ms |
| Load 12 images | ~100ms |
| **Total response** | **< 150ms** |

### Interview Talking Point

> "Session Explorer combines three AI techniques: vector embeddings for semantic search, LLM-extracted labels for filtering, and a blob storage pattern for images.
>
> The filters—art style, mood, complexity—were all extracted by a local LLM during preprocessing. At query time, I combine vector similarity with SQL filters in a single DuckDB query.
>
> This lets me answer questions like 'show me failed cyberpunk generations in anime style'—and actually see the images. That's how you debug a GenAI system."

---

## Project Structure

```
genai-session-analyzer/
├── pyproject.toml
├── README.md
├── CLAUDE.md                    # This file
│
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── diffusiondb_loader.py
│   │
│   ├── enrichment/              # ML feature extraction (pre-compute)
│   │   ├── __init__.py
│   │   ├── precompute_clip.py   # CLIP embeddings + alignment scores
│   │   ├── precompute_llm.py    # LLM prompt analysis
│   │   ├── precompute_text_embeddings.py  # sentence-transformers for search
│   │   └── precompute_clusters.py # K-means style clustering
│   │
│   ├── copilot/                 # SQL Copilot (live inference)
│   │   ├── __init__.py
│   │   ├── schema.py            # get_duckdb_schema()
│   │   ├── prompt.py            # construct_prompt()
│   │   └── llm.py               # get_llm_response() via MLX
│   │
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── user_generator.py
│   │   ├── telemetry_enricher.py
│   │   └── session_builder.py
│   │
│   └── pipeline/
│       ├── __init__.py
│       ├── definitions.py       # Dagster assets
│       └── resources.py
│
├── dashboard/
│   └── app.py                   # Streamlit: metrics + SQL Copilot + Session Explorer
│
├── data/
│   ├── warehouse.duckdb         # All metadata, embeddings, analysis
│   └── blob/                    # Simulated blob storage (like S3)
│       └── images/
│           └── generations/     # 10K images from DiffusionDB (~5GB)
│               ├── gen_00001.png
│               └── ...
│
└── scripts/
    └── run_precompute.py        # One-time ML enrichment (Saturday night)
```

---

## Execution Plan

### Saturday (Foundation + Enrichment Day)

| Time Block | Task | Deliverable |
|------------|------|-------------|
| **Morning (4h)** | Project setup + data download | Working `uv` environment, DuckDB connected, 10K images downloaded |
| **Afternoon (4h)** | Simulation layer | `user_generator.py`, `telemetry_enricher.py` producing realistic fake data |
| **Evening (3h)** | Dagster pipeline + start enrichment | Asset DAG running; kick off `run_precompute.py` to run overnight |

**Saturday Night (runs while you sleep):**
```bash
# Run ML enrichment (~70 min for 10K)
python scripts/run_precompute.py

# This runs:
# 1. CLIP embeddings (~5 min)
# 2. LLM prompt analysis (~50 min)
# 3. K-means clustering (~1 min)
```

**Saturday Checkpoint:** Can run `dagster dev` and see data flowing. Enrichment running in background.

### Sunday (Polish Day)

| Time Block | Task | Deliverable |
|------------|------|-------------|
| **Morning (4h)** | Verify enrichment + build aggregations | Star schema complete with ML features, `fct_sessions` working |
| **Afternoon (3h)** | Dashboard | Streamlit app with 4-5 key visualizations including image viewer |
| **Evening (3h)** | Demo prep | Live query console, talking points, edge case handling |

**Sunday Checkpoint:** Can demo full pipeline + dashboard + run ad-hoc queries + show ML-powered insights.

---

## Demo Script (for interview)

1. **Start the pipeline** (30 sec)
   ```bash
   dagster dev
   # Show the asset graph materializing
   ```

2. **Show raw → enriched → transformed flow** (2 min)
   - Open DuckDB CLI
   - Query raw prompts → show CLIP/LLM enrichment columns → show aggregated sessions

3. **Dashboard walkthrough** (3 min)
   - Friction Score distribution by user tier
   - Session cost trends over time
   - "Power users churn less" visualization

4. **SQL Copilot demo** (2 min) ⭐ **Wow moment #1**
   ```
   Type: "Show me error rate by user tier"
   → Watch SQL generate in real-time (~0.5s)
   → Review the generated query
   → Click "Run Query"
   → Show results
   ```

5. **Session Explorer** (3 min) ⭐ **Wow moment #2**
   ```
   Type: "cyberpunk city"
   → See image grid with similar prompts
   → Filter by Status = "timeout"
   → "All timeouts are realistic style—anime has only 12% errors"
   → Show actual generated images alongside metrics
   ```

6. **Live query — connect the insight** (1 min)
   ```sql
   -- Confirm the pattern: realistic cyberpunk = high timeout rate
   SELECT
     art_style,
     COUNT(*) AS total,
     SUM(CASE WHEN status = 'timeout' THEN 1 ELSE 0 END) AS timeouts,
     ROUND(100.0 * SUM(CASE WHEN status = 'timeout' THEN 1 ELSE 0 END) / COUNT(*), 1) AS timeout_pct
   FROM fct_generations g
   JOIN dim_prompts p USING (prompt_id)
   WHERE p.prompt_text ILIKE '%cyber%' OR p.mood = 'dark'
   GROUP BY art_style
   ORDER BY timeout_pct DESC;
   ```

7. **Design decisions** (3 min)
   - Why DuckDB over Postgres (OLAP, zero-config, columnar, native VSS)
   - Why MLX for copilot, sentence-transformers for search (right tool for each job)
   - Why star schema (analytics-friendly, clear separation)
   - Production migration path (DuckDB → MotherDuck/Snowflake, blob → S3)

---

## Tool Preferences

- **Package management**: Use `uv`, not pip
- **Formatting**: Run `ruff format` before committing
- **SQL over Python**: For transformations, write SQL in Dagster assets—avoid pandas
- **Quick queries**: Use `duckdb` CLI directly, not Python wrappers
- **No notebooks**: All code in `.py` files for demo clarity
- **File creation**: Prefer creating actual files over showing code in chat
- **ML inference**: Use MLX for local, mention vLLM/TGI for production
- **Pre-compute over runtime**: All ML enrichment happens once, stored as columns

---

## Anti-Patterns to Avoid

- ❌ Don't over-engineer the simulation (it's a demo, not a physics engine)
- ❌ Don't add Kubernetes/Docker unless asked
- ❌ Don't build a REST API (this is analytics, not a product)
- ❌ Don't spend time on auth/security (it's a local demo)
- ❌ Don't use pandas for transformations (use SQL/dbt—it shows better instincts)

---

## Open Questions (Decide Early)

1. **Data volume**: Start with 5K (safe) or go for 10K (impressive)?
   → Start with 5K for iteration, scale to 10K Saturday night if enrichment runs smoothly

2. **Dashboard features**: Basic metrics or include image viewer?
   → Include image viewer with similarity search—it's the "wow" factor

3. **Fallback plan**: What if MLX enrichment fails?
   → Schema still works without ML columns; just skip those demo queries

---

## Resources

- [DiffusionDB on HuggingFace](https://huggingface.co/datasets/poloclub/diffusiondb)
- [DuckDB Python API](https://duckdb.org/docs/api/python/overview)
- [Dagster Tutorial](https://docs.dagster.io/tutorial)
- [Streamlit Docs](https://docs.streamlit.io/)
- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [MLX-LM (Language Models)](https://github.com/ml-explore/mlx-examples/tree/main/llms)
- [MLX Community Models](https://huggingface.co/mlx-community)
- [Qwen2.5 Models](https://huggingface.co/Qwen)

---

## When Stuck

Ask Claude Code for:
1. "Generate the boilerplate for [component]"
2. "Debug this error: [paste traceback]"
3. "Write the SQL for [business question]"
4. "What's the 80/20 way to implement [feature]?"

**Don't ask for:**
- Philosophical debates about architecture
- Comparisons of 5 different tools
- Perfect solutions (we need working solutions)

---

*Last updated: Project Kickoff*
*Target: Luma AI Data Science Interview — Screen 1: Project Review*
