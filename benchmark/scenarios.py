"""Hardened benchmark workload: multi-step tool chains, distractor tools, IFEval conjunctions, competition math, and HumanEval+ data structures."""

from __future__ import annotations

import json
import re

TOOLS = {
    "get_weather": {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current real-time temperature for a city right now.",
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
    "get_weather_forecast": {
        "type": "function",
        "function": {
            "name": "get_weather_forecast",
            "description": "Get the future weather forecast for a city for upcoming days (not current weather).",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "days_ahead": {"type": "integer", "description": "Number of days in the future (1 to 7)"},
                },
                "required": ["city", "days_ahead"],
            },
        },
    },
    "get_historical_weather": {
        "type": "function",
        "function": {
            "name": "get_historical_weather",
            "description": "Get past historical weather data for a city on a specific past date (YYYY-MM-DD).",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "date": {"type": "string", "description": "Past date in YYYY-MM-DD format"},
                },
                "required": ["city", "date"],
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
    "search_products": {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search the product catalog by query term.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query term, e.g. laptop, phone"},
                    "max_results": {"type": "integer", "description": "Maximum number of results to return"},
                },
                "required": ["query"],
            },
        },
    },
    "check_inventory": {
        "type": "function",
        "function": {
            "name": "check_inventory",
            "description": "Check real-time warehouse stock quantity for a specific product item by exact product name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string", "description": "Exact product name from catalog"},
                },
                "required": ["product_name"],
            },
        },
    },
    "calculate_tax": {
        "type": "function",
        "function": {
            "name": "calculate_tax",
            "description": "Calculate sales tax and total amount based on subtotal amount and state code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subtotal": {"type": "number", "description": "Subtotal purchase amount in USD"},
                    "state_code": {"type": "string", "description": "Two-letter US state code, e.g. CA, NY, TX"},
                },
                "required": ["subtotal", "state_code"],
            },
        },
    },
    "get_flight_status": {
        "type": "function",
        "function": {
            "name": "get_flight_status",
            "description": "Get current status and departure delay for a flight.",
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_number": {"type": "string", "description": "Flight number, e.g. AA100"},
                },
                "required": ["flight_number"],
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
    "cancel_booking": {
        "type": "function",
        "function": {
            "name": "cancel_booking",
            "description": "Cancel an active booking reservation by booking ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "booking_id": {"type": "string", "description": "Booking reference ID, e.g. BK-1002"},
                },
                "required": ["booking_id"],
            },
        },
    },
    "send_email": {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email message.",
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
            "description": "Evaluate an arithmetic math expression.",
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
    "paris": 22.0, "tokyo": 31.0, "london": 17.0, "berlin": 18.0,
    "new york": 26.0, "sydney": 15.0, "madrid": 30.0,
}
STOCKS = {"NVDA": 120.0, "AAPL": 225.0, "TSLA": 210.0, "MSFT": 415.0}
FLIGHTS = {"AA100": "on time", "UA202": "delayed 90 minutes", "DL301": "on time"}
CATALOG = {
    "laptop": [
        {"name": "Laptop Pro 16", "ram_gb": 32, "price": 1499.0},
        {"name": "Laptop Lite 14", "ram_gb": 8, "price": 599.0},
        {"name": "Laptop Air 15", "ram_gb": 16, "price": 899.0},
    ],
    "phone": [
        {"name": "Phone Ultra", "price": 999.0},
        {"name": "Phone Basic", "price": 399.0},
    ],
}
INVENTORY = {
    "laptop air 15": {"in_stock": True, "quantity": 14, "warehouse": "US-East"},
    "laptop pro 16": {"in_stock": True, "quantity": 3, "warehouse": "US-West"},
    "laptop lite 14": {"in_stock": False, "quantity": 0, "warehouse": "US-Central"},
}
TAX_RATES = {"ca": 0.0825, "ny": 0.08875, "tx": 0.0625, "wa": 0.065}


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
    if name == "get_weather_forecast":
        city = str(args.get("city", "")).strip().lower()
        days = int(args.get("days_ahead", 1))
        temp = WEATHER.get(city, 20.0) + (days * 1.5)
        return json.dumps({"city": city, "days_ahead": days, "forecast_temp_c": temp, "condition": "rain expected"})
    if name == "get_historical_weather":
        city = str(args.get("city", "")).strip().lower()
        date = str(args.get("date", ""))
        return json.dumps({"city": city, "date": date, "historical_temp_c": 12.5, "condition": "overcast"})
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
    if name == "check_inventory":
        p = str(args.get("product_name", "")).strip().lower()
        inv = INVENTORY.get(p)
        if inv is None:
            return json.dumps({"error": f"Product '{args.get('product_name')}' not found in inventory."})
        return json.dumps(inv)
    if name == "calculate_tax":
        sub = float(args.get("subtotal", 0.0))
        st = str(args.get("state_code", "")).strip().lower()
        rate = TAX_RATES.get(st, 0.07)
        tax = round(sub * rate, 2)
        total = round(sub + tax, 2)
        return json.dumps({"subtotal": sub, "tax_rate": rate, "tax_amount": tax, "total_with_tax": total})
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
    if name == "cancel_booking":
        bid = args.get("booking_id", "")
        return json.dumps({"status": "cancelled", "booking_id": bid, "refund_issued": True})
    return f"unknown tool: {name}"


# --- Tool-Calling & Agentic Evaluation Scenarios ---

SCENARIOS = [
    # 1. Simple Single-Turn Tool Extraction
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

    # 2. Parallel Multi-Tool Calls
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

    # 3. Multi-Turn Agentic Workflows
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
     "expected_answer": "laptop lite 14"},
    {"id": "m05", "category": "multi_turn", "tools": ["get_flight_status"],
     "messages": [{"role": "user", "content": "Is flight AA100 currently on time?"}],
     "expected_answer": "on time"},
    {"id": "m06", "category": "multi_turn", "tools": ["send_email"],
     "messages": [{"role": "user", "content": "Email alice@example.com with subject 'Q3 Report' and tell her it is attached."}],
     "expected_answer": "sent"},

    # 4. No-Tool Restraint (BFCL Irrelevant Tools / Hallucination Checks)
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

    # 5. Error Recovery & Adaptive Self-Correction (tau-bench)
    {"id": "er01", "category": "error_recovery", "tools": ["get_flight_status"],
     "messages": [{"role": "user", "content": "Check the status of flight AA999."}],
     "expected_answer": "not found"},
    {"id": "er02", "category": "error_recovery", "tools": ["get_weather"],
     "messages": [{"role": "user", "content": "What is the temperature in the fictional city Atlantis?"}],
     "expected_answer": "not found"},

    # 6. Complex Nested Arguments Schema (BFCL Complex)
    {"id": "c01", "category": "complex_args", "tools": ["book_flight"],
     "messages": [{"role": "user", "content": "Book flight AA100 for Alice and Bob in economy class. This is a one-way trip (not round trip)."}],
     "expected_calls": [{"name": "book_flight", "arguments": {
         "flight_number": "AA100",
         "passengers": [{"name": "Alice", "seat_class": "economy"}, {"name": "Bob", "seat_class": "economy"}],
         "round_trip": False,
     }}]},

    # 7. Distractor Tools Ambiguity (BFCL Tool Selection Under Clutter)
    {"id": "dt01", "category": "distractor_tools",
     "tools": ["get_weather", "get_weather_forecast", "get_historical_weather"],
     "messages": [{"role": "user", "content": "What is the weather forecast for Paris in 3 days? Do not give me current weather."}],
     "expected_calls": [{"name": "get_weather_forecast", "arguments": {"city": "Paris", "days_ahead": 3}}]},
    {"id": "dt02", "category": "distractor_tools",
     "tools": ["get_weather", "get_weather_forecast", "get_historical_weather"],
     "messages": [{"role": "user", "content": "What was the temperature in Berlin on 2024-01-15?"}],
     "expected_calls": [{"name": "get_historical_weather", "arguments": {"city": "Berlin", "date": "2024-01-15"}}]},

    # 8. Multi-Step Dependent Tool Chains (GAIA Level 2 & tau-bench Hard)
    {"id": "chain_01", "category": "multi_turn",
     "tools": ["search_products", "check_inventory", "calculate_tax", "send_email"],
     "messages": [{"role": "user", "content": "Find the cheapest laptop with at least 16GB RAM from the product catalog, check if it is in stock at the warehouse, calculate total cost with CA sales tax, and email order confirmation to customer@corp.com with subject 'Order Confirmed'."}],
     "expected_answer": ["973.17", "customer@corp.com"]},
    {"id": "chain_02", "category": "multi_turn",
     "tools": ["get_flight_status", "cancel_booking", "book_flight"],
     "messages": [{"role": "user", "content": "Check the status of flight UA202. If it is delayed by more than 30 minutes, cancel existing reservation BK-1002 and book replacement flight AA100 for passenger 'Alice' in economy class."}],
     "expected_answer": ["cancelled", "aa100"]},
]


# --- Google IFEval Hard Conjunction Scenarios ---

IFEVAL_SCENARIOS = [
    {
        "id": "ifeval_h01",
        "rule_id": "h_paragraph_and_no_comma",
        "prompt": "Write a short essay on serverless cloud computing. You must satisfy 4 strict constraints simultaneously:\n1. Your response must contain EXACTLY 3 paragraphs, separated by double newlines.\n2. Paragraph 2 MUST start with the exact word 'Serverless'.\n3. Paragraph 3 must contain ZERO comma characters ','.\n4. Do NOT use any bullet points or lists anywhere.",
    },
    {
        "id": "ifeval_h02",
        "rule_id": "h_json_schema_ranges",
        "prompt": "Output a system health telemetry report. Your response MUST be valid JSON adhering to this exact schema:\n- 'status': string (must be exactly 'healthy', 'degraded', or 'critical')\n- 'cpu_percent': number between 0.0 and 100.0\n- 'services': array of at least 3 objects, each with 'name' (string) and 'latency_ms' (number)\n- 'alert_triggered': boolean\nProvide only the JSON object.",
    },
    {
        "id": "ifeval_h03",
        "rule_id": "h_word_count_and_keywords",
        "prompt": "Explain distributed Byzantine fault tolerance. You must satisfy these rules simultaneously:\n1. Your response must be between 60 and 90 words long.\n2. Include the word 'consensus' at least 3 times.\n3. Include the word 'byzantine' at least once.\n4. End your response with the exact marker: [END_OF_REPORT]",
    },
    {
        "id": "ifeval_h04",
        "rule_id": "h_table_and_no_letter_e",
        "prompt": "Compare SQL vs NoSQL databases in a markdown table. Your response MUST satisfy:\n1. Include a valid markdown table with at least 3 columns and at least 2 data rows.\n2. Follow the table with a concluding sentence (at least 10 words) that contains ZERO occurrences of the letter 'e' (case-insensitive).",
    },
    {
        "id": "ifeval_h05",
        "rule_id": "h_tags_bold_and_all_caps",
        "prompt": "Create an infrastructure security checklist. You must satisfy:\n1. Wrap your entire response in <audit> and </audit> tags.\n2. Include at least 3 sections starting with tags [SECTION_1], [SECTION_2], [SECTION_3].\n3. Every section title immediately following the tag MUST be written in ALL CAPS.",
    },
    {
        "id": "ifeval_h06",
        "rule_id": "h_forbidden_words",
        "prompt": "Explain how public-key cryptography works in at least 50 words. Strict constraint: you must NOT use any of the following words (or their plurals): 'secure', 'encrypt', 'key', 'cipher', 'protect'.",
    },
]


# --- Competition Math & Multi-Step Reasoning (AIME / Olympiad) ---

GSM8K_SCENARIOS = [
    {
        "id": "aime_01",
        "prompt": "What is the remainder when 7^2026 is divided by 100? Show your step-by-step modular arithmetic reasoning and state the final integer as #### <answer>.",
        "expected_answer": 49,
    },
    {
        "id": "aime_02",
        "prompt": "Find the number of ordered pairs of positive integers (x, y) such that 1/x + 1/y = 1/12. Show your step-by-step factorization and state the final integer count as #### <answer>.",
        "expected_answer": 15,
    },
    {
        "id": "aime_03",
        "prompt": "How many 4-digit positive integers have the property that the sum of their digits is equal to 34? Show your combinatorial reasoning and state the final integer count as #### <answer>.",
        "expected_answer": 10,
    },
    {
        "id": "aime_04",
        "prompt": "A sequence of positive integers a_1, a_2, ..., a_n satisfies a_1 = 1 and a_{k+1} = a_k + 2k + 1 for all k >= 1. What is the value of a_{30}? Show your steps and state the final integer as #### <answer>.",
        "expected_answer": 900,
    },
    {
        "id": "aime_05",
        "prompt": "In right triangle ABC with hypotenuse AB = 65, the altitude to the hypotenuse has length 28. What is the perimeter of triangle ABC? Show your work and state the final integer perimeter as #### <answer>.",
        "expected_answer": 154,
    },
    {
        "id": "aime_06",
        "prompt": "How many four-digit security PIN codes (from 0000 to 9999) contain at least one digit '7'? Show your work and state the final integer count as #### <answer>.",
        "expected_answer": 3439,
    },
]


# --- GPQA Diamond (Google-Proof PhD-Level Science Reasoning) ---

GPQA_SCENARIOS = [
    {
        "id": "gpqa_01",
        "domain": "Physics / Quantum Mechanics",
        "prompt": "Consider a one-dimensional infinite square well of width L extending from x = 0 to x = L. A particle is in the ground state psi(x) = sqrt(2/L) * sin(pi * x / L) for 0 <= x <= L. If the width of the well is suddenly expanded to 2L (from 0 to 2L) symmetrically, what is the probability that the particle is found in the ground state of the expanded well?\n(A) 16 / (9*pi^2)\n(B) 32 / (9*pi^2)\n(C) 8 / (3*pi^2)\n(D) 64 / (9*pi^2)\nState your step-by-step reasoning and end with your final letter choice as #### <Letter>.",
        "expected_answer": "B",
    },
    {
        "id": "gpqa_02",
        "domain": "Organic Chemistry / Reaction Mechanisms",
        "prompt": "When (R)-2-bromobutane is treated with sodium iodide in acetone (Finkelstein reaction conditions), which statement accurately describes the stereochemical outcome of the 2-iodobutane product?\n(A) Complete retention of configuration with optical purity unchanged\n(B) Predominantly (S)-2-iodobutane via SN2 mechanism with inversion of configuration\n(C) A 50:50 racemic mixture via an SN1 carbocation intermediate\n(D) An equal mixture of diastereomers\nState your step-by-step reasoning and end with your final letter choice as #### <Letter>.",
        "expected_answer": "B",
    },
    {
        "id": "gpqa_03",
        "domain": "Molecular Biology & Genetics",
        "prompt": "In E. coli, if the operator region of the lac operon suffers a constitutive mutation (lacO^c) that prevents the LacI repressor protein from binding, but the CAP-cAMP binding site remains intact, what will be the transcription level of beta-galactosidase in the presence of BOTH high glucose and high lactose in the growth medium?\n(A) Maximal high expression rate\n(B) Low/basal transcription rate (due to catabolite repression / low cAMP)\n(C) Complete zero expression\n(D) Oscillating expression dependent on ATP levels\nState your step-by-step reasoning and end with your final letter choice as #### <Letter>.",
        "expected_answer": "B",
    },
    {
        "id": "gpqa_04",
        "domain": "Thermodynamics & Statistical Mechanics",
        "prompt": "For a two-level quantum system with energy states E_1 = 0 and E_2 = epsilon > 0 with degeneracies g_1 = 1 and g_2 = 3, what is the behavior of the heat capacity C_V per particle in the limit of high temperatures (k_B * T >> epsilon)?\n(A) Approaches 3 * k_B / 4\n(B) Approaches 3 * k_B / 2\n(C) Approaches zero proportional to 1/T^2\n(D) Approaches infinity\nState your step-by-step reasoning and end with your final letter choice as #### <Letter>.",
        "expected_answer": "C",
    },
    {
        "id": "gpqa_05",
        "domain": "Optics & Electromagnetism",
        "prompt": "A beam of unpolarized light of initial intensity I_0 passes through three ideal linear polarizers in series. Polarizer 1 has its axis at 0 deg (vertical), Polarizer 2 at 30 deg to the vertical, and Polarizer 3 at 90 deg (horizontal). What is the intensity of the light emerging from Polarizer 3?\n(A) 0\n(B) 3 * I_0 / 32\n(C) I_0 / 8\n(D) 3 * I_0 / 16\nState your step-by-step reasoning and end with your final letter choice as #### <Letter>.",
        "expected_answer": "B",
    },
    {
        "id": "gpqa_06",
        "domain": "Biochemistry & Enzyme Kinetics",
        "prompt": "In the presence of a reversible competitive inhibitor, how do the apparent Michaelis constant (K_m^app) and the maximum reaction velocity (V_max^app) change compared to the uninhibited enzyme kinetics?\n(A) K_m^app increases while V_max^app remains unchanged\n(B) K_m^app remains unchanged while V_max^app decreases\n(C) Both K_m^app and V_max^app decrease proportionally\n(D) K_m^app decreases while V_max^app increases\nState your step-by-step reasoning and end with your final letter choice as #### <Letter>.",
        "expected_answer": "A",
    },
]


# --- HumanEval+ / LeetCode Stateful Data Structures ---

HUMANEVAL_SCENARIOS = [
    {
        "id": "he_01",
        "entry_point": "LRUCache",
        "prompt": '''class LRUCache:
    """ Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.
    Implement the LRUCache class:
    - LRUCache(int capacity) Initialize the LRU cache with positive size capacity.
    - int get(int key) Return the value of the key if the key exists, otherwise return -1.
    - void put(int key, int value) Update the value of the key if the key exists. Otherwise, add the key-value
      pair to the cache. If the number of keys exceeds the capacity from this operation, evict the least recently used key.
    The functions get and put must each run in O(1) average time complexity.
    """
    def __init__(self, capacity: int):
        pass

    def get(self, key: int) -> int:
        pass

    def put(self, key: int, value: int) -> None:
        pass
''',
        "test": '''cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
assert cache.get(1) == 1
cache.put(3, 3)
assert cache.get(2) == -1
cache.put(4, 4)
assert cache.get(1) == -1
assert cache.get(3) == 3
assert cache.get(4) == 4
# Test capacity 1
c1 = LRUCache(1)
c1.put(10, 100)
assert c1.get(10) == 100
c1.put(20, 200)
assert c1.get(10) == -1
assert c1.get(20) == 200
''',
    },
    {
        "id": "he_02",
        "entry_point": "MinStack",
        "prompt": '''class MinStack:
    """ Design a stack that supports push, pop, top, and retrieving the minimum element in constant time O(1).
    Implement the MinStack class:
    - MinStack() initializes the stack object.
    - void push(int val) pushes the element val onto the stack.
    - void pop() removes the element on the top of the stack.
    - int top() gets the top element of the stack.
    - int getMin() retrieves the minimum element in the stack.
    """
    def __init__(self):
        pass

    def push(self, val: int) -> None:
        pass

    def pop(self) -> None:
        pass

    def top(self) -> int:
        pass

    def getMin(self) -> int:
        pass
''',
        "test": '''st = MinStack()
st.push(-2)
st.push(0)
st.push(-3)
assert st.getMin() == -3
st.pop()
assert st.top() == 0
assert st.getMin() == -2
st.push(-5)
assert st.getMin() == -5
assert st.top() == -5
st.pop()
assert st.getMin() == -2
''',
    },
    {
        "id": "he_03",
        "entry_point": "merge_intervals",
        "prompt": '''def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    """ Given an array of intervals where intervals[i] = [start_i, end_i], merge all overlapping intervals,
    and return an array of the non-overlapping intervals that cover all the intervals in the input.
    The intervals must be returned sorted by start time.
    >>> merge_intervals([[1,3],[2,6],[8,10],[15,18]])
    [[1,6],[8,10],[15,18]]
    >>> merge_intervals([[1,4],[4,5]])
    [[1,5]]
    """
''',
        "test": '''assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]
assert merge_intervals([[1,4],[4,5]]) == [[1,5]]
assert merge_intervals([]) == []
assert merge_intervals([[1,4],[0,4]]) == [[0,4]]
assert merge_intervals([[1,4],[2,3]]) == [[1,4]]
assert merge_intervals([[2,3],[4,5],[6,7],[8,9],[1,10]]) == [[1,10]]
''',
    },
    {
        "id": "he_04",
        "entry_point": "longest_valid_parentheses",
        "prompt": '''def longest_valid_parentheses(s: str) -> int:
    """ Given a string containing just the characters '(' and ')', return the length of the longest valid
    (well-formed) parentheses substring.
    >>> longest_valid_parentheses("(()")
    2
    >>> longest_valid_parentheses(")()())")
    4
    >>> longest_valid_parentheses("")
    0
    """
''',
        "test": '''assert longest_valid_parentheses("(()") == 2
assert longest_valid_parentheses(")()())") == 4
assert longest_valid_parentheses("") == 0
assert longest_valid_parentheses("()(()") == 2
assert longest_valid_parentheses("()(())") == 6
assert longest_valid_parentheses("((()))") == 6
assert longest_valid_parentheses(")))(((") == 0
''',
    },
    {
        "id": "he_05",
        "entry_point": "Trie",
        "prompt": '''class Trie:
    """ A trie (pronounced as "try") or prefix tree is a tree data structure used to efficiently store and retrieve keys.
    Implement the Trie class:
    - Trie() Initializes the trie object.
    - void insert(String word) Inserts the string word into the trie.
    - boolean search(String word) Returns true if the string word is in the trie (i.e., was inserted before), and false otherwise.
    - boolean startsWith(String prefix) Returns true if there is a previously inserted string word that has the prefix prefix, and false otherwise.
    """
    def __init__(self):
        pass

    def insert(self, word: str) -> None:
        pass

    def search(self, word: str) -> bool:
        pass

    def startsWith(self, prefix: str) -> bool:
        pass
''',
        "test": '''trie = Trie()
trie.insert("apple")
assert trie.search("apple") == True
assert trie.search("app") == False
assert trie.startsWith("app") == True
trie.insert("app")
assert trie.search("app") == True
assert trie.startsWith("b") == False
trie.insert("banana")
assert trie.search("banana") == True
assert trie.startsWith("ban") == True
''',
    },
    {
        "id": "he_06",
        "entry_point": "max_subarray_sum_circular",
        "prompt": '''def max_subarray_sum_circular(nums: list[int]) -> int:
    """ Given a circular integer array nums of length n, return the maximum possible sum of a non-empty subarray of nums.
    A circular array means the end of the array connects to the beginning of the array.
    >>> max_subarray_sum_circular([1,-2,3,-2])
    3
    >>> max_subarray_sum_circular([5,-3,5])
    10
    >>> max_subarray_sum_circular([-3,-2,-3])
    -2
    """
''',
        "test": '''assert max_subarray_sum_circular([1,-2,3,-2]) == 3
assert max_subarray_sum_circular([5,-3,5]) == 10
assert max_subarray_sum_circular([-3,-2,-3]) == -2
assert max_subarray_sum_circular([3,-1,2,-1]) == 4
assert max_subarray_sum_circular([3,-2,2,-3]) == 3
assert max_subarray_sum_circular([-10]) == -10
''',
    },
]


# --- Throughput Evaluation Prompts (Code Generation) ---

THROUGHPUT_PROMPTS = [
    "Write a Python function that checks if a string is a palindrome, with a brief explanation.",
    "Write a Python function that computes the nth Fibonacci number with dynamic programming, and briefly explain it.",
    "Write a Python function that merges two sorted lists into one sorted list, with a brief explanation.",
    "Write a Python function that performs binary search on a sorted list, with a brief explanation.",
]
