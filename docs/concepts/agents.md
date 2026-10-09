# Demystifying Agents: From GPT-2 Next-Token Prediction to the ReAct Loop

> *"An agent is not a thinking being trapped inside silicon; it is an autoregressive token predictor wrapped in a while loop that knows how to read tool results."*

---

## 1. What Is an Agent, Really?

In industry marketing, "AI Agents" are often presented as autonomous digital entities with human-like volition. In software engineering reality, an agent is simply **a formalized tool-calling loop**.

On Day 6 and Day 7 of StockSage, we built the core agentic mechanism from scratch using raw Groq API calls and Python functions (`get_current_price`, `get_pe_ratio`):
1. Send a user prompt and a list of tool definitions to an LLM.
2. The LLM inspects the prompt, compares it against available tools, and outputs a structured tool call.
3. The Python runtime parses the function name and arguments, executes the real function, and captures the result.
4. The runtime appends the function result back into the prompt history and invokes the LLM again.
5. The LLM either requests another tool call or outputs the final natural-language response.

This cyclic pattern is formally known as **ReAct** (Reasoning + Acting, Yao et al., 2022).

---

## 2. The ReAct Architecture: Thought, Action, Observation

The ReAct pattern structures agent execution into three discrete stages executed sequentially inside a bounded loop:

```
                  ┌──────────────────────────────────────────────┐
                  │                USER QUESTION                 │
                  │   "Is NVDA expensive compared to its P/E?"   │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                 ┌────────────────────────────────────────────────┐
                 │ 1. REASONING (LLM Forward Pass)               │
                 │ Attends over history and decides:              │
                 │ "I need NVDA's current P/E ratio before I can │
                 │ assess its valuation."                         │
                 └───────────────────────┬────────────────────────┘
                                         │
                                         ▼
                 ┌────────────────────────────────────────────────┐
                 │ 2. ACTION (External Tool Dispatch)             │
                 │ Emits tool call:                               │
                 │ `get_pe_ratio(ticker="NVDA")`                  │
                 └───────────────────────┬────────────────────────┘
                                         │
                                         ▼
                 ┌────────────────────────────────────────────────┐
                 │ 3. OBSERVATION (Environment Feedback)          │
                 │ Python executes yfinance call:                 │
                 │ Returns `65.4` -> Appended to message context  │
                 └───────────────────────┬────────────────────────┘
                                         │
                      ┌──────────────────┴──────────────────┐
                      ▼                                     ▼
            [Goal Not Satisfied]                  [Goal Satisfied]
        Loop back to 1. REASONING            Emit final answer to user
        (e.g., fetch current price)          ("NVDA's P/E is 65.4, which...")
```

### The Three Primitives in StockSage (`src/core/tools.py`)

- **Reasoning**: The model inspects the chat history and generates internal tokens deciding the next sub-step.
- **Action**: The model emits a special structured token stream matching our function schema (e.g. `{"name": "get_pe_ratio", "arguments": "{\"ticker\": \"NVDA\"}"}`).
- **Observation**: The Python runtime invokes the real API/database and appends the result into the message buffer with `role: "tool"`.

---

## 3. The Theoretical Bridge: Connecting Agents to GPT-2

When you build or train a decoder-only transformer like GPT-2 from scratch, you learn several immutable mathematical truths:

1. **A transformer is stateless:** Given an input sequence of tokens $x_1, x_2, \dots, x_t$, the model computes logits over vocabulary $V$ and samples token $x_{t+1} \sim P(x_{t+1} \mid x_1, \dots, x_t)$. It does not remember past runs. There is no background "thought process" running between inferences.
2. **Causal Multi-Head Self-Attention:** Each token at position $i$ only attends to positions $j \le i$. The Attention matrix is:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$
   where $M$ is the causal mask ($M_{ij} = -\infty$ for $j > i$).
3. **No External Actuators:** A transformer cannot execute Python code, ping an HTTP endpoint, or query ChromaDB. Its only output is a probability vector over its vocabulary.

### How Does Token Prediction Produce "Agentic Behavior"?

If an LLM is just computing $P(x_{t+1} \mid x_{1:t})$, how does it act like an intelligent financial analyst?

#### 1. In-Context Conditioning
During fine-tuning (e.g., RLHF / Instruction Tuning), models are trained on trajectories of:
`User prompt -> Thought -> Action -> Observation -> Final Answer`.
When we prompt the model with a user request and tool specifications, the causal self-attention heads compute attention weights between the user's tokens ("What is Apple's P/E?") and the tool definition tokens (`get_pe_ratio`). Because of statistical associations learned during training, the highest probability next tokens are the opening delimiters of a tool invocation.

#### 2. The Agent Loop Is in Python, NOT the Transformer
The transformer **never runs a loop**. The transformer only runs a single autoregressive forward pass until it emits an End-of-Turn (`<|eot_id|>`) token.
The "loop" is in our Python runtime:
```python
while iteration < max_iterations:
    response = client.chat.completions.create(...) # Forward pass
    if not response.tool_calls:
        return response.content                   # Done
    observation = dispatch_tool(response.tool_calls)
    messages.append({"role": "tool", "content": observation}) # Context expansion
```

#### 3. KV Caching and Expanding Context
In GPT-2, autoregressive generation caches Key and Value projections ($K_{cache}, V_{cache}$) to avoid quadratic recalculation. In an agent loop, each **Observation** is simply an injection of new tokens into the prompt prefix.
When the LLM runs its second reasoning pass, its attention queries attend to the new observation tokens. The model doesn't "remember" it called a tool — it literally sees the tool's answer right in front of it in its context window.

---

## 4. Why Single-Loop Agents Break Down

While a single ReAct loop (`src/core/tools.py`) works for simple queries ("What is AAPL's price?"), it fails as financial queries scale in complexity:
- **Context Bloat:** Dumping 10-Q filing extracts, 5 news articles, and price histories into a single prompt blows through token limits and dilutes attention.
- **Role Confusion:** An LLM asked to calculate technical indicators, summarize news, check SEC filings, and formulate a portfolio recommendation in one prompt suffers from hallucination and drift.
- **No Orchestration Structure:** If a query requires fundamentals *and* news *and* risk metrics before writing an analyst report, a free-form loop may get stuck cycling or prematurely terminate.

This limitation is the exact motivation for **Multi-Agent Architectures (Phase 3)**:
Instead of one generalist agent wandering through a loop, we partition the problem into specialized agents (Data Agent, News Agent, Risk Agent, Analyst Agent) orchestrated deterministically using **LangGraph**.
