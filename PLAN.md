# Scoring Function Lab — Implementation Plan

## What We're Building

A testbed for designing, testing, and comparing agent scoring functions. Given a task and an agent loop, swap in different scoring strategies (rule-based, LLM-graded, test-pass/fail, hybrid) and measure how each affects convergence speed, token cost, and churn. The deliverable is both the framework and a scoring function taxonomy built from real experimental data.

## Why This Matters

The AI Breakdown and NVIDIA Latent Space episodes converge on the same underappreciated insight: **the scoring function is the most important design decision in any agent loop** — and it's the one that gets the least attention.

Most agent frameworks treat scoring as an afterthought ("did the task complete? yes/no"). But the research shows:
- A poorly designed scoring function causes infinite churn (agent explores without converging)
- A well-designed scoring function can cut iterations in half on identical tasks
- The same task with different scoring produces completely different agent behavior

Three things nobody has built yet:
1. A way to **swap scoring functions** on a live agent loop without changing anything else
2. A way to **measure** how scoring function quality affects convergence metrics
3. A **taxonomy** of scoring patterns (which designs work for which task types)

This project builds all three. The Boris Cherny insight about the agent loop primitive is only useful if you can design good scoring functions — this is the missing tool.

## Architecture

```
scorelab/
├── __init__.py
├── task.py          # Task definition: problem, ground truth, evaluation criteria
├── loop.py          # AgentLoop: pluggable scorer, runs iterations, collects metrics
├── scorer.py        # Scorer interface + implementations
├── metrics.py       # Convergence metrics, churn rate, cost per score point
├── runner.py        # ExperimentRunner: run same task against N scorers, compare
├── store.py         # SQLite: experiment history, scorer registry
├── renderer.py      # Rich terminal: live loop view, comparison tables
└── cli.py           # `score run`, `score compare`, `score taxonomy`, `score history`

scorers/
├── rule_based.py    # Pattern matching, regex, threshold rules
├── llm_graded.py    # Ask LLM to grade the hypothesis (0.0-1.0)
├── test_runner.py   # Run tests, score = fraction passing
├── semantic.py      # Embedding cosine similarity to ground truth
├── composite.py     # Weighted combination of multiple scorers
└── human.py         # Interactive: pause and ask user to grade

tasks/
├── code_fix.py      # Fix a bug in P³ codebase
├── extraction.py    # Extract all startups from podcast transcript
├── summarization.py # Summarize episode within 200 tokens
├── query_answer.py  # Answer factual question from transcript
└── refactor.py      # Refactor function to reduce cyclomatic complexity

tests/
├── test_scorer.py
├── test_loop.py
└── test_runner.py

pyproject.toml
README.md
SCORING_TAXONOMY.md   # Output: empirical taxonomy of scoring patterns
```

## Core Concept

```python
# Define a task
task = Task(
    name="extract-startups",
    description="Extract all startup names mentioned in this transcript",
    input={"transcript": "..."},
    ground_truth=["Anthropic", "OpenAI", "Mistral", "Cohere"],
    max_iterations=10,
)

# Define competing scorers
scorers = [
    RuleBasedScorer(rules=["count exact matches"]),
    LLMGradedScorer(prompt="Rate completeness of extraction 0-1"),
    SemanticScorer(embedding_model="all-MiniLM-L6-v2"),
    CompositeScorer([RuleBasedScorer(weight=0.6), SemanticScorer(weight=0.4)]),
]

# Run the same agent loop against each scorer
runner = ExperimentRunner(agent_loop=my_loop, task=task)
results = runner.compare(scorers)

# Output:
# Scorer              Iterations  Tokens   Final Score  Converged?  Cost
# rule_based          8           4,200    0.75         yes         $0.006
# llm_graded          3           1,800    0.95         yes         $0.003
# semantic            12          6,400    0.82         yes         $0.009
# composite_6040      4           2,100    0.92         yes         $0.003
```

## Implementation Phases

### Phase 1: Task definition (task.py)

```python
@dataclass
class Task:
    name: str
    description: str                    # Natural language task description
    input: dict                         # Task inputs (transcript, code, etc.)
    ground_truth: Any                   # Expected answer (for scoring)
    max_iterations: int = 10
    success_threshold: float = 0.85     # Score required to consider "done"
    task_type: str = "extraction"       # extraction | code_fix | summarization | qa

    def to_prompt(self) -> str:
        """Format task as agent prompt."""
```

Five concrete task types, each testing different scoring challenges:
- **extraction**: ground truth is a list of items (easy to score with exact match, hard with partial matches)
- **code_fix**: ground truth is "tests pass" (binary but unambiguous)
- **summarization**: ground truth is subjective (forces LLM grading)
- **query_answer**: ground truth is a fact (can use semantic similarity)
- **refactor**: ground truth is a metric (cyclomatic complexity ≤ N)

### Phase 2: Scorer interface (scorer.py)

```python
class BaseScorer:
    name: str
    description: str

    def score(self, hypothesis: str, task: Task, iteration: int) -> ScorerResult:
        """Score a hypothesis. Returns 0.0-1.0 with explanation."""
        raise NotImplementedError

@dataclass
class ScorerResult:
    score: float            # 0.0 to 1.0
    explanation: str        # Why this score?
    confidence: float       # How confident is the scorer? (0.0-1.0)
    latency_ms: int
    cost_usd: float         # For LLM-graded scorers
```

### Phase 3: Scorer implementations (scorers/)

**RuleBasedScorer**
- Pattern matching: count regex hits, check field presence
- Threshold rules: score ≥ 0.5 if ≥ 3 expected items found
- Zero cost, deterministic, fast
- Best for: structured extraction, factual queries

**LLMGradedScorer**
- Sends hypothesis + task description to LLM
- LLM returns structured JSON: `{"score": 0.7, "rationale": "..."}`
- Cost: ~150 tokens per score call
- Best for: subjective quality, creative tasks, summarization

**TestRunnerScorer**
- Executes a test command: `pytest tests/ -x -q`
- Score = fraction of tests passing
- Binary per test, continuous overall
- Best for: code fix, refactor tasks

**SemanticScorer**
- Embeds hypothesis and ground truth
- Score = cosine similarity
- Zero LLM cost, but misses semantic nuance
- Best for: open-ended answers with known ground truth

**CompositeScorer**
- Weighted combination of any scorers
- `composite = 0.6 * rule_based + 0.4 * semantic`
- Calibrate weights from historical runs

### Phase 4: Agent loop (loop.py)

```python
class AgentLoop:
    def __init__(self, model: str, scorer: BaseScorer, task: Task):
        self.iterations: list[Iteration] = []

    def run(self) -> LoopResult:
        """Execute task with pluggable scorer. Returns full iteration history."""

    def step(self) -> Iteration:
        """One agent iteration: generate hypothesis → score → decide to continue."""

@dataclass
class Iteration:
    number: int
    hypothesis: str          # What the agent tried
    score: float             # Score for this iteration
    score_delta: float       # Improvement over previous iteration
    tokens_used: int
    cost_usd: float
    converged: bool          # Did this iteration meet success_threshold?
    scorer_result: ScorerResult
```

Convergence criteria:
1. Score ≥ `task.success_threshold` → done, success
2. Score hasn't improved in 3 consecutive iterations → done, stuck
3. Iteration count ≥ `task.max_iterations` → done, timeout

### Phase 5: Metrics (metrics.py)

```python
@dataclass
class ConvergenceMetrics:
    iterations_to_converge: int | None    # None if didn't converge
    final_score: float
    total_tokens: int
    total_cost_usd: float
    churn_index: float                    # From agent-churn-visualizer: total_iters / depth_of_win
    score_trajectory: list[float]         # Score per iteration
    stuck_episodes: int                   # Times score didn't improve
    cost_per_score_point: float           # total_cost / final_score
    efficiency_score: float               # sum(positive deltas) / total_iterations
```

Key insight: **cost_per_score_point** is the unified metric for comparing scorers. A scorer that costs $0.001 per call but causes 8 iterations is worse than one costing $0.003 per call with 3 iterations.

### Phase 6: Experiment runner (runner.py)

```python
class ExperimentRunner:
    def compare(self, scorers: list[BaseScorer], runs: int = 3) -> ComparisonReport:
        """Run same task N times against each scorer. Return comparison."""

    def run_matrix(self, tasks: list[Task], scorers: list[BaseScorer]) -> MatrixReport:
        """Full factorial: every task × every scorer. Build taxonomy."""
```

Each run uses the same task with a fresh agent context. Multiple runs (default: 3) average out LLM non-determinism.

### Phase 7: Terminal renderer (renderer.py)

Live view during a run:
```
Task: extract-startups  |  Scorer: composite_6040  |  Iteration 3/10
──────────────────────────────────────────────────────────────────────
Hypothesis: ["Anthropic", "OpenAI", "Mistral"]
Score: 0.75 (+0.25) ████████░░  converging...
Tokens this iter: 420  |  Total: 1,240  |  Cost: $0.002

Score Trajectory: 0.50 → 0.62 → 0.75 → ...
```

Comparison table after experiment:
```
Scorer              Iters  Tokens   Cost     Final    CPS      Converged
────────────────────────────────────────────────────────────────────────
rule_based          8      4,200    $0.006   0.75     $0.008   ✓ (slow)
llm_graded          3      1,800    $0.012   0.95     $0.013   ✓ (best score)
semantic            12     6,400    $0.009   0.82     $0.011   ✓ (overexplored)
composite_6040      4      2,100    $0.004   0.92     $0.004   ✓ best overall
────────────────────────────────────────────────────────────────────────
Winner (cost-per-score-point): composite_6040
```

### Phase 8: Build the Scoring Taxonomy

Run the full task × scorer matrix:
- 5 tasks × 5 scorers × 3 runs each = 75 experiments
- Collect convergence metrics for all 75
- Analyze: which scorer wins for each task type?
- Document patterns in `SCORING_TAXONOMY.md`

Expected findings:
- TestRunnerScorer dominates for code tasks (unambiguous ground truth)
- LLMGradedScorer wins for summarization (subjectivity requires judgment)
- CompositeScorer beats single scorers on extraction (combines precision + recall)
- RuleBasedScorer has best cost-efficiency for factual QA

### Phase 9: CLI

```bash
# Run a task with a specific scorer
score run --task extraction --scorer composite --iterations 10

# Compare all scorers on a task
score compare --task code_fix --runs 3

# Run full matrix and update taxonomy
score matrix --tasks all --scorers all

# View history
score history --task extraction --limit 20

# Print taxonomy
score taxonomy
```

## Key Design Decisions

**Why pluggable scorers over a fixed evaluation framework?**
The whole thesis is that scoring function design drives agent behavior. If the scorer is fixed, you can't test this claim. The pluggable interface makes the hypothesis testable.

**Why 5 task types?**
Different task types expose different scoring failure modes. Code tasks need binary test-pass scoring. Extraction tasks need precision/recall trade-offs. Summarization tasks expose the limits of rule-based scoring. You need all five to build a real taxonomy, not just anecdotes.

**Why cost_per_score_point as the unified metric?**
It integrates three things: scorer cost (per call), agent cost (tokens used), and quality (final score). A cheap scorer that causes 10 iterations may be worse than an expensive one that terminates in 3. CPS collapses this into one number.

**What we're NOT building**
- Automated scorer selection (follow-on: train a meta-scorer)
- Multi-task scoring (scoring N subtasks simultaneously)
- Reward model training (this is eval, not RL)

## Acceptance Criteria

1. All 5 task types run end-to-end with all 5 scorer types
2. `score compare --task code_fix` shows measurable difference between scorers (not all converge in same iterations)
3. LLMGradedScorer correctly grades subjective outputs (verified by hand-checking 10 scores)
4. `SCORING_TAXONOMY.md` committed with real data from 75 experiments — patterns clearly stated with numbers
5. Churn index matches agent-churn-visualizer output on same tasks (cross-validation)

## Learning Outcomes

After building this you will understand:
- Why the scoring function is the most important design decision in an agent loop
- What "convergence" means quantitatively — and how to measure it before you've built the agent
- Why LLM-graded scoring beats rule-based for subjective tasks despite higher cost
- How composite scorers combine the best of multiple strategies
- That the same agent, same task, different scorer = completely different behavior (the main insight)
