"""Vendored, deterministic benchmark workload (offline, no runtime downloads).

Provenance / Reputability:
- Tool-calling & Agentic: BFCL (Berkeley Function Calling Leaderboard) format
  (simple, parallel, complex nested args, no-tool restraint) and tau-bench / GAIA
  style multi-turn execution and error recovery.
- Instruction Following: Google IFEval (deterministic constraint verification).
- Math Reasoning: GSM8K (multi-step arithmetic word problems).
- Code Intelligence: HumanEval (Python functional logic with test execution).
- Throughput: LLMPerf-style generation workload with code prompts.
"""

from __future__ import annotations

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
    "book_flight": {
        "type": "function",
        "function": {
            "name": "book_flight",
            "description": "Book flight tickets for one or more passengers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_number": {"type": "string", "description": "Flight number, e.g. AA100"},
                    "passengers": {
                        "type": "array",
                        "description": "List of passenger records",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "seat_class": {"type": "string", "enum": ["economy", "business"]},
                            },
                            "required": ["name"],
                        },
                    },
                    "round_trip": {"type": "boolean", "description": "Whether this is a round trip booking"},
                },
                "required": ["flight_number", "passengers"],
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
            return json.dumps({"error": f"City '{args.get('city')}' not found in weather database."})
        if unit == "fahrenheit":
            return f"weather in {city}: {temp * 9 / 5 + 32:.1f}°F"
        return f"weather in {city}: {temp:.1f}°C"
    if name == "get_stock_price":
        sym = str(args.get("symbol", "")).strip().upper()
        price = STOCKS.get(sym)
        if price is None:
            return json.dumps({"error": f"Symbol '{args.get('symbol')}' not found."})
        return f"{sym} current price: ${price:.2f}"
    if name == "get_flight_status":
        fn = str(args.get("flight_number", "")).strip().upper()
        if fn not in FLIGHTS:
            return json.dumps({"error": f"Flight {fn} not found. Available flights: AA100, UA202, DL301."})
        return f"flight {fn} status: {FLIGHTS[fn]}"
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
            return str(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307
        except Exception:
            return "invalid expression"
    if name == "book_flight":
        fn = args.get("flight_number", "")
        passengers = args.get("passengers", [])
        rt = args.get("round_trip", False)
        return json.dumps({
            "status": "confirmed",
            "booking_id": "BK-98421",
            "flight": fn,
            "passengers_count": len(passengers),
            "round_trip": rt,
        })
    return f"unknown tool: {name}"


# --- Tool-Calling & Agentic Scenarios (BFCL & tau-bench) ---

SCENARIOS = [
    # 1. Simple single tool call (argument extraction)
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

    # 2. Parallel multiple tool calls
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

    # 3. Multi-turn Agentic (tau-bench / GAIA)
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

    # 4. No-Tool Restraint (BFCL Irrelevant Tools / Hallucination Detection)
    {"id": "nt01", "category": "no_tool", "tools": ["get_weather", "get_stock_price"],
     "messages": [{"role": "user", "content": "What is the capital city of France?"}],
     "expected_answer": "paris"},
    {"id": "nt02", "category": "no_tool", "tools": ["get_flight_status", "calculator"],
     "messages": [{"role": "user", "content": "Write a three-line haiku about the blue ocean."}],
     "expected_answer": None},
    {"id": "nt03", "category": "no_tool", "tools": ["search_products", "send_email"],
     "messages": [{"role": "user", "content": "Who wrote the famous tragedy play Hamlet?"}],
     "expected_answer": "shakespeare"},
    {"id": "nt04", "category": "no_tool", "tools": ["get_weather", "calculator"],
     "messages": [{"role": "user", "content": "Explain what gravity is in two sentences."}],
     "expected_answer": None},

    # 5. Error Recovery / Self-Correction (tau-bench)
    {"id": "er01", "category": "error_recovery", "tools": ["get_flight_status"],
     "messages": [{"role": "user", "content": "Check the status of flight AA999."}],
     "expected_answer": "not found"},
    {"id": "er02", "category": "error_recovery", "tools": ["get_weather"],
     "messages": [{"role": "user", "content": "What is the temperature in the fictional city Atlantis?"}],
     "expected_answer": "not found"},

    # 6. Complex Nested Arguments (BFCL Complex)
    {"id": "c01", "category": "complex_args", "tools": ["book_flight"],
     "messages": [{"role": "user", "content": "Book flight AA100 for Alice and Bob in economy class. This is a one-way trip (not round trip)."}],
     "expected_calls": [{"name": "book_flight", "arguments": {
         "flight_number": "AA100",
         "passengers": [{"name": "Alice", "seat_class": "economy"}, {"name": "Bob", "seat_class": "economy"}],
         "round_trip": False,
     }}]},
]


# --- Google IFEval (Instruction Following Evaluation) ---

IFEVAL_SCENARIOS = [
    {
        "id": "ifeval_01",
        "rule_id": "json_schema",
        "prompt": "Generate a profile for a software engineer. Your response MUST be valid JSON with exact keys 'name' (string), 'years_experience' (integer), 'skills' (list of at least 3 strings), and 'remote' (boolean).",
    },
    {
        "id": "ifeval_02",
        "rule_id": "no_comma",
        "prompt": "Write a short paragraph (at least 40 words) describing space exploration. Do NOT use the comma character ',' anywhere in your response.",
    },
    {
        "id": "ifeval_03",
        "rule_id": "keyword_freq",
        "prompt": "Explain what a neural network is. You MUST include the exact word 'neuron' at least 4 times in your explanation.",
    },
    {
        "id": "ifeval_04",
        "rule_id": "exact_paragraphs",
        "prompt": "Explain why clean code is important in software engineering. Your response must contain EXACTLY 3 paragraphs, separated by double newlines. Do NOT use any bullet points or lists.",
    },
    {
        "id": "ifeval_05",
        "rule_id": "tags_and_bold",
        "prompt": "List 3 benefits of physical exercise. Highlight each benefit title in double asterisks like **Benefit Name**, and wrap the entire response inside <response> and </response> tags.",
    },
    {
        "id": "ifeval_06",
        "rule_id": "end_phrase",
        "prompt": "Describe renewable solar energy in about 50 words. Your response MUST end with the exact sentence: 'The future is green.'",
    },
]


# --- GSM8K (Grade School Math Multi-Step Reasoning) ---

GSM8K_SCENARIOS = [
    {
        "id": "gsm8k_01",
        "prompt": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Show your work and end with the final integer.",
        "expected_answer": 72,
    },
    {
        "id": "gsm8k_02",
        "prompt": "Weng earns $12 an hour for babysitting. Yesterday, she did 50 minutes of babysitting. How much money in dollars did she earn? Show your work and end with the final integer.",
        "expected_answer": 10,
    },
    {
        "id": "gsm8k_03",
        "prompt": "Betty is saving money for a new wallet which costs $100. Betty already has half of the money she needs. Her parents gave her $15 and her grandparents gave her twice as much as her parents. How many more dollars does Betty need to buy the wallet? Show your work and end with the final integer.",
        "expected_answer": 5,
    },
    {
        "id": "gsm8k_04",
        "prompt": "A deep-sea creature rises from the waters once every 100 years. Over a span of 300 years, it spends 2/3 of the total time sleeping. How many years was it awake? Show your work and end with the final integer.",
        "expected_answer": 100,
    },
    {
        "id": "gsm8k_05",
        "prompt": "Mark has a garden with flowers. He has 10 rows of flowers with 8 flowers in each row. If he sells 3/4 of all the flowers, how many flowers are left in the garden? Show your work and end with the final integer.",
        "expected_answer": 20,
    },
    {
        "id": "gsm8k_06",
        "prompt": "James writes a 3-page letter to 2 different friends twice a week. How many total pages does he write in one full year (52 weeks)? Show your work and end with the final integer.",
        "expected_answer": 624,
    },
]


# --- HumanEval (Python Code Generation & Execution) ---

HUMANEVAL_SCENARIOS = [
    {
        "id": "he_01",
        "entry_point": "has_close_elements",
        "prompt": 'def has_close_elements(numbers: list[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n',
        "test": 'assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\nassert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\nassert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\nassert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n',
    },
    {
        "id": "he_02",
        "entry_point": "separate_paren_groups",
        "prompt": "def separate_paren_groups(paren_string: str) -> list[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses.\n    Return the list of separated groups.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n",
        "test": "assert separate_paren_groups('(()()) (( )) () ((())()())') == ['(()())', '(())', '()', '((())()())']\nassert separate_paren_groups('() (()) ((())) (((())))') == ['()', '(())', '((()))', '(((())))']\nassert separate_paren_groups('(()(())((())))') == ['(()(())((())))']\n",
    },
    {
        "id": "he_03",
        "entry_point": "truncate_number",
        "prompt": 'def truncate_number(number: float) -> float:\n    """ Given a positive floating point number, return the decimal part of the number.\n    >>> truncate_number(3.5)\n    0.5\n    """\n',
        "test": "assert abs(truncate_number(3.5) - 0.5) < 1e-6\nassert abs(truncate_number(1.25) - 0.25) < 1e-6\nassert abs(truncate_number(123.0) - 0.0) < 1e-6\n",
    },
    {
        "id": "he_04",
        "entry_point": "below_zero",
        "prompt": 'def below_zero(operations: list[int]) -> bool:\n    """ Detect if at any point the bank account balance falls below zero.\n    >>> below_zero([1, 2, 3])\n    False\n    >>> below_zero([1, 2, -4, 5])\n    True\n    """\n',
        "test": "assert below_zero([]) == False\nassert below_zero([1, 2, -3, 1, 2, -3]) == False\nassert below_zero([1, 2, -4, 5, 6]) == True\nassert below_zero([1, -1, 2, -2, 5, -5, 4, -4]) == False\nassert below_zero([1, -1, 2, -2, 5, -5, 4, -5]) == True\n",
    },
    {
        "id": "he_05",
        "entry_point": "mean_absolute_deviation",
        "prompt": 'def mean_absolute_deviation(numbers: list[float]) -> float:\n    """ Calculate Mean Absolute Deviation around the mean of this dataset.\n    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])\n    1.0\n    """\n',
        "test": "assert abs(mean_absolute_deviation([1.0, 2.0, 3.0]) - 2.0/3.0) < 1e-6\nassert abs(mean_absolute_deviation([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-6\nassert abs(mean_absolute_deviation([1.0, 2.0, 3.0, 4.0, 5.0]) - 6.0/5.0) < 1e-6\n",
    },
    {
        "id": "he_06",
        "entry_point": "intersperse",
        "prompt": 'def intersperse(numbers: list[int], delimeter: int) -> list[int]:\n    """ Insert a number \'delimeter\' between every two consecutive elements of input list.\n    >>> intersperse([1, 2, 3], 4)\n    [1, 4, 2, 4, 3]\n    """\n',
        "test": "assert intersperse([], 7) == []\nassert intersperse([5, 6, 3, 2], 8) == [5, 8, 6, 8, 3, 8, 2]\nassert intersperse([2, 2, 2], 2) == [2, 2, 2, 2, 2]\n",
    },
]


# --- Throughput Evaluation Prompts (Code Generation) ---

THROUGHPUT_PROMPTS = [
    "Write a Python function that checks if a string is a palindrome, with a brief explanation.",
    "Write a Python function that computes the nth Fibonacci number with dynamic programming, and briefly explain it.",
    "Write a Python function that merges two sorted lists into one sorted list, with a brief explanation.",
    "Write a Python function that performs binary search on a sorted list, with a brief explanation.",
]
