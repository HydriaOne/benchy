"""Vendored, deterministic benchmark workload (offline, no runtime downloads).

Provenance / reputability:
- Tool-call scenarios follow the Berkeley Function Calling Leaderboard (BFCL)
  format: predict a function call (name + arguments) from a user turn, graded
  by tool-name match + argument subset match. Categories simple / parallel map
  to BFCL's simple and parallel categories.
- Multi-turn agentic scenarios follow tau-bench / GAIA style: the model calls a
  tool, the harness executes it deterministically, the model produces a final
  answer, which is graded on outcome correctness.
- Throughput prompts follow the LLMPerf synthetic-generation shape: a fixed
  prompt, bounded output tokens, measure tokens/sec under 1x and Nx concurrency.

All tool execution is deterministic canned data; no live network, no wall-clock
dependencies.
"""

import json
import re

TOOLS = {
    "get_weather": {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current temperature for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    },
    "get_stock_price": {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current share price for a stock symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Ticker symbol, e.g. NVDA"},
                },
                "required": ["symbol"],
            },
        },
    },
    "get_flight_status": {
        "type": "function",
        "function": {
            "name": "get_flight_status",
            "description": "Get the current status of a flight.",
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_number": {"type": "string", "description": "Flight number, e.g. AA100"},
                },
                "required": ["flight_number"],
            },
        },
    },
    "search_products": {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search the product catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Maximum number of results"},
                },
                "required": ["query"],
            },
        },
    },
    "send_email": {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    "calculator": {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate an arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Arithmetic expression"},
                },
                "required": ["expression"],
            },
        },
    },
}

WEATHER = {
    "paris": 22.0,
    "tokyo": 31.0,
    "london": 17.0,
    "berlin": 18.0,
    "new york": 26.0,
    "sydney": 15.0,
    "madrid": 30.0,
}
STOCKS = {"NVDA": 120.0, "AAPL": 225.0, "TSLA": 210.0, "MSFT": 415.0}
FLIGHTS = {"AA100": "on time", "UA202": "delayed 45 minutes", "DL301": "on time"}
CATALOG = {
    "laptop": [
        {"name": "Laptop A", "price": 999.0},
        {"name": "Laptop B", "price": 799.0},
        {"name": "Laptop C", "price": 1299.0},
    ],
    "phone": [
        {"name": "Phone X", "price": 699.0},
        {"name": "Phone Y", "price": 899.0},
    ],
}


def execute_tool(name: str, args: dict) -> str:
    args = args or {}
    if name == "get_weather":
        city = str(args.get("city", "")).strip().lower()
        unit = str(args.get("unit", "celsius")).lower()
        temp = WEATHER.get(city)
        if temp is None:
            return f"unknown city: {args.get('city')}"
        if unit == "fahrenheit":
            return f"weather in {city}: {temp * 9 / 5 + 32:.1f}°F"
        return f"weather in {city}: {temp:.1f}°C"
    if name == "get_stock_price":
        sym = str(args.get("symbol", "")).strip().upper()
        price = STOCKS.get(sym)
        if price is None:
            return f"unknown symbol: {args.get('symbol')}"
        return f"{sym} current price: ${price:.2f}"
    if name == "get_flight_status":
        fn = str(args.get("flight_number", "")).strip().upper()
        return f"flight {fn} status: {FLIGHTS.get(fn, 'unknown')}"
    if name == "search_products":
        q = str(args.get("query", "")).strip().lower()
        items = CATALOG.get(q)
        if items is None:
            items = CATALOG.get(q.rstrip("s")) or CATALOG.get(q + "s") or []
        return json.dumps(items) if items else "[]"
    if name == "send_email":
        to = args.get("to", "")
        subj = args.get("subject", "")
        return f"email sent to {to} with subject '{subj}'"
    if name == "calculator":
        expr = str(args.get("expression", ""))
        if not re.fullmatch(r"[0-9+\-*/().\s%]+", expr):
            return "invalid expression"
        try:
            return str(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307 - regex-whitelisted
        except Exception:
            return "invalid expression"
    return f"unknown tool: {name}"


SCENARIOS = [
    # --- simple: single tool call, argument extraction ---
    {"id": "s01", "category": "simple", "tools": ["get_weather"],
     "messages": [{"role": "user", "content": "What is the current temperature in Berlin?"}],
     "expected_calls": [{"name": "get_weather", "arguments": {"city": "Berlin"}}]},
    {"id": "s02", "category": "simple", "tools": ["get_weather"],
     "messages": [{"role": "user", "content": "What is the temperature in Paris, in Fahrenheit?"}],
     "expected_calls": [{"name": "get_weather", "arguments": {"city": "Paris", "unit": "fahrenheit"}}]},
    {"id": "s03", "category": "simple", "tools": ["get_stock_price"],
     "messages": [{"role": "user", "content": "What is the current stock price of NVDA?"}],
     "expected_calls": [{"name": "get_stock_price", "arguments": {"symbol": "NVDA"}}]},
    {"id": "s04", "category": "simple", "tools": ["get_flight_status"],
     "messages": [{"role": "user", "content": "What is the status of flight AA100?"}],
     "expected_calls": [{"name": "get_flight_status", "arguments": {"flight_number": "AA100"}}]},
    {"id": "s05", "category": "simple", "tools": ["search_products"],
     "messages": [{"role": "user", "content": "Search the product catalog for laptops."}],
     "expected_calls": [{"name": "search_products", "arguments": {"query": "laptops"}}]},
    {"id": "s06", "category": "simple", "tools": ["send_email"],
     "messages": [{"role": "user", "content": "Send an email to bob@example.com with subject 'Meeting' to confirm tomorrow's standup."}],
     "expected_calls": [{"name": "send_email", "arguments": {"to": "bob@example.com", "subject": "Meeting"}}]},
    {"id": "s07", "category": "simple", "tools": ["get_weather"],
     "messages": [{"role": "user", "content": "Tell me the current temperature in Sydney."}],
     "expected_calls": [{"name": "get_weather", "arguments": {"city": "Sydney"}}]},
    {"id": "s08", "category": "simple", "tools": ["get_flight_status"],
     "messages": [{"role": "user", "content": "Is flight UA202 delayed?"}],
     "expected_calls": [{"name": "get_flight_status", "arguments": {"flight_number": "UA202"}}]},

    # --- parallel: multiple tool calls in one turn ---
    {"id": "p01", "category": "parallel", "tools": ["get_weather"],
     "messages": [{"role": "user", "content": "Compare the current temperature in Tokyo and London."}],
     "expected_calls": [{"name": "get_weather", "arguments": {"city": "Tokyo"}},
                        {"name": "get_weather", "arguments": {"city": "London"}}]},
    {"id": "p02", "category": "parallel", "tools": ["get_stock_price"],
     "messages": [{"role": "user", "content": "What are the current prices of NVDA and AAPL?"}],
     "expected_calls": [{"name": "get_stock_price", "arguments": {"symbol": "NVDA"}},
                        {"name": "get_stock_price", "arguments": {"symbol": "AAPL"}}]},
    {"id": "p03", "category": "parallel", "tools": ["get_weather"],
     "messages": [{"role": "user", "content": "What is the weather in Paris and New York right now?"}],
     "expected_calls": [{"name": "get_weather", "arguments": {"city": "Paris"}},
                        {"name": "get_weather", "arguments": {"city": "New York"}}]},
    {"id": "p04", "category": "parallel", "tools": ["get_flight_status"],
     "messages": [{"role": "user", "content": "Check the status of flights AA100 and UA202."}],
     "expected_calls": [{"name": "get_flight_status", "arguments": {"flight_number": "AA100"}},
                        {"name": "get_flight_status", "arguments": {"flight_number": "UA202"}}]},
    {"id": "p05", "category": "parallel", "tools": ["get_stock_price"],
     "messages": [{"role": "user", "content": "Get the current prices of MSFT and TSLA."}],
     "expected_calls": [{"name": "get_stock_price", "arguments": {"symbol": "MSFT"}},
                        {"name": "get_stock_price", "arguments": {"symbol": "TSLA"}}]},
    {"id": "p06", "category": "parallel", "tools": ["get_weather"],
     "messages": [{"role": "user", "content": "What are the temperatures in Madrid and Berlin?"}],
     "expected_calls": [{"name": "get_weather", "arguments": {"city": "Madrid"}},
                        {"name": "get_weather", "arguments": {"city": "Berlin"}}]},

    # --- multi-turn agentic: tool result -> final answer ---
    {"id": "m01", "category": "multi_turn", "tools": ["get_stock_price", "calculator"],
     "messages": [{"role": "user", "content": "I want to buy 5 shares of NVDA. What will it cost me in total?"}],
     "expected_answer": "600"},
    {"id": "m02", "category": "multi_turn", "tools": ["get_weather"],
     "messages": [{"role": "user", "content": "Which city is warmer right now: Paris or London?"}],
     "expected_answer": "paris"},
    {"id": "m03", "category": "multi_turn", "tools": ["get_stock_price"],
     "messages": [{"role": "user", "content": "What is the current price of one share of AAPL?"}],
     "expected_answer": "225"},
    {"id": "m04", "category": "multi_turn", "tools": ["search_products"],
     "messages": [{"role": "user", "content": "Search for laptops and tell me which model is the cheapest."}],
     "expected_answer": "laptop b"},
    {"id": "m05", "category": "multi_turn", "tools": ["get_flight_status"],
     "messages": [{"role": "user", "content": "Is flight AA100 currently on time?"}],
     "expected_answer": "on time"},
    {"id": "m06", "category": "multi_turn", "tools": ["send_email"],
     "messages": [{"role": "user", "content": "Email alice@example.com with subject 'Q3 Report' and tell her it is attached."}],
     "expected_answer": "sent"},
]

THROUGHPUT_PROMPTS = [
    "Write a Python function that checks if a string is a palindrome, with a brief explanation.",
    "Write a Python function that computes the nth Fibonacci number with dynamic programming, and briefly explain it.",
    "Write a Python function that merges two sorted lists into one sorted list, with a brief explanation.",
    "Write a Python function that performs binary search on a sorted list, with a brief explanation.",
]
