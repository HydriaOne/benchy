"""Hardened benchmark workload: multi-step tool chains, distractor tools, IFEval conjunctions, competition math, HumanEval+ data structures, and Artificial Analysis Intelligence Index suites."""

from __future__ import annotations

import copy
import json
import math
import re

TOOLS = {
    # --- General Core Tools ---
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

    # --- T3-Banking (tau-bench) Tools ---
    "get_user_profile": {
        "type": "function",
        "function": {
            "name": "get_user_profile",
            "description": "Fetch user identity, contact information, and associated bank accounts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "Unique user ID, e.g. USR_101"},
                },
                "required": ["user_id"],
            },
        },
    },
    "get_account_balance": {
        "type": "function",
        "function": {
            "name": "get_account_balance",
            "description": "Get the current available and ledger balance for a specific bank account.",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {"type": "string", "description": "Account number, e.g. ACC_CHK_101, ACC_SAV_101"},
                },
                "required": ["account_id"],
            },
        },
    },
    "list_transactions": {
        "type": "function",
        "function": {
            "name": "list_transactions",
            "description": "List recent transactions on an account.",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {"type": "string", "description": "Account number"},
                    "limit": {"type": "integer", "description": "Max number of records to return"},
                },
                "required": ["account_id"],
            },
        },
    },
    "transfer_funds": {
        "type": "function",
        "function": {
            "name": "transfer_funds",
            "description": "Transfer funds between accounts or to an external verified recipient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_account": {"type": "string", "description": "Source account ID"},
                    "to_account": {"type": "string", "description": "Destination account ID or external routing"},
                    "amount": {"type": "number", "description": "Amount in USD to transfer"},
                    "note": {"type": "string", "description": "Transfer memo/note"},
                },
                "required": ["from_account", "to_account", "amount"],
            },
        },
    },
    "freeze_card": {
        "type": "function",
        "function": {
            "name": "freeze_card",
            "description": "Immediately freeze a debit/credit card to prevent unauthorized charges.",
            "parameters": {
                "type": "object",
                "properties": {
                    "card_id": {"type": "string", "description": "Card identifier, e.g. CARD_4321"},
                    "reason": {"type": "string", "description": "Reason for freezing (e.g. suspected_fraud, lost)"},
                },
                "required": ["card_id"],
            },
        },
    },
    "unfreeze_card": {
        "type": "function",
        "function": {
            "name": "unfreeze_card",
            "description": "Unfreeze a previously blocked payment card.",
            "parameters": {
                "type": "object",
                "properties": {
                    "card_id": {"type": "string", "description": "Card identifier"},
                },
                "required": ["card_id"],
            },
        },
    },
    "waive_fee": {
        "type": "function",
        "function": {
            "name": "waive_fee",
            "description": "Request a courtesy waiver/refund for an overdraft or maintenance fee.",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {"type": "string", "description": "Account number"},
                    "fee_id": {"type": "string", "description": "Fee transaction ID to waive, e.g. FEE_OD_901"},
                    "reason": {"type": "string", "description": "Customer justification"},
                },
                "required": ["account_id", "fee_id"],
            },
        },
    },
    "file_dispute": {
        "type": "function",
        "function": {
            "name": "file_dispute",
            "description": "File a formal chargeback dispute for an unauthorized or fraudulent transaction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {"type": "string", "description": "Transaction ID to dispute, e.g. TX_8901"},
                    "dispute_reason": {"type": "string", "description": "Reason for dispute"},
                },
                "required": ["transaction_id", "dispute_reason"],
            },
        },
    },

    # --- Terminal-Bench v4.0 Tools ---
    "bash_exec": {
        "type": "function",
        "function": {
            "name": "bash_exec",
            "description": "Execute a standard bash command in the simulated sandbox terminal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Bash command string to execute"},
                },
                "required": ["command"],
            },
        },
    },
    "read_file": {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents from the sandbox filesystem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to file"},
                },
                "required": ["path"],
            },
        },
    },
    "write_file": {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or overwrite file contents in the sandbox filesystem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to file"},
                    "content": {"type": "string", "description": "Text content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
}

# --- Core General Simulation Data ---
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


# ==============================================================================
# --- T3-Banking (tau-bench) Stateful Environment & Execution ---
# ==============================================================================

INITIAL_BANKING_DB = {
    "users": {
        "USR_101": {
            "user_id": "USR_101",
            "name": "Sarah Connor",
            "email": "sarah.c@example.com",
            "phone": "+1-555-0199",
            "accounts": ["ACC_CHK_101", "ACC_SAV_101", "ACC_MMA_101"],
            "cards": ["CARD_4321", "CARD_8877"],
        },
        "USR_202": {
            "user_id": "USR_202",
            "name": "David Martinez",
            "email": "david.m@example.com",
            "accounts": ["ACC_CHK_202"],
            "cards": ["CARD_1122"],
        }
    },
    "accounts": {
        "ACC_CHK_101": {"account_id": "ACC_CHK_101", "type": "checking", "balance": 1850.00, "user_id": "USR_101"},
        "ACC_SAV_101": {"account_id": "ACC_SAV_101", "type": "savings", "balance": 12400.50, "user_id": "USR_101"},
        "ACC_MMA_101": {"account_id": "ACC_MMA_101", "type": "money_market", "balance": 5000.00, "user_id": "USR_101"},
        "ACC_CHK_202": {"account_id": "ACC_CHK_202", "type": "checking", "balance": 450.00, "user_id": "USR_202"},
        "ACC_EXTERNAL_BOB": {"account_id": "ACC_EXTERNAL_BOB", "type": "external", "balance": 0.0, "user_id": "EXT_BOB"},
    },
    "cards": {
        "CARD_4321": {"card_id": "CARD_4321", "account_id": "ACC_CHK_101", "status": "active", "card_type": "debit"},
        "CARD_8877": {"card_id": "CARD_8877", "account_id": "ACC_CHK_101", "status": "frozen", "card_type": "credit"},
        "CARD_1122": {"card_id": "CARD_1122", "account_id": "ACC_CHK_202", "status": "active", "card_type": "debit"},
    },
    "transactions": {
        "TX_8901": {"tx_id": "TX_8901", "account_id": "ACC_CHK_101", "amount": 89.99, "merchant": "Suspicious-TechStore-Online", "category": "retail", "date": "2026-08-28", "status": "posted"},
        "TX_8902": {"tx_id": "TX_8902", "account_id": "ACC_CHK_101", "amount": 14.50, "merchant": "BlueBottle Coffee", "category": "dining", "date": "2026-08-27", "status": "posted"},
        "TX_8903": {"tx_id": "TX_8903", "account_id": "ACC_CHK_101", "amount": 112.40, "merchant": "Whole Foods Market", "category": "groceries", "date": "2026-08-25", "status": "posted"},
        "TX_8904": {"tx_id": "TX_8904", "account_id": "ACC_CHK_101", "amount": 84.10, "merchant": "Trader Joe's", "category": "groceries", "date": "2026-08-18", "status": "posted"},
        "FEE_OD_901": {"tx_id": "FEE_OD_901", "account_id": "ACC_CHK_101", "amount": 35.00, "merchant": "Bank Overdraft Charge", "category": "fee", "date": "2026-08-28", "status": "posted"},
    },
    "disputes": {},
    "transfers": [],
}


def create_banking_state() -> dict:
    return copy.deepcopy(INITIAL_BANKING_DB)


def execute_banking_tool(db: dict, name: str, args: dict) -> str:
    args = args or {}
    if name == "get_user_profile":
        uid = str(args.get("user_id", "")).strip()
        user = db["users"].get(uid)
        if not user:
            return json.dumps({"error": f"User {uid} not found"})
        return json.dumps(user)

    if name == "get_account_balance":
        aid = str(args.get("account_id", "")).strip()
        acc = db["accounts"].get(aid)
        if not acc:
            return json.dumps({"error": f"Account {aid} not found"})
        return json.dumps({"account_id": aid, "type": acc["type"], "balance": acc["balance"]})

    if name == "list_transactions":
        aid = str(args.get("account_id", "")).strip()
        limit = int(args.get("limit", 10))
        txs = [tx for tx in db["transactions"].values() if tx["account_id"] == aid][:limit]
        return json.dumps(txs)

    if name == "transfer_funds":
        from_acc = str(args.get("from_account", "")).strip()
        to_acc = str(args.get("to_account", "")).strip()
        amount = float(args.get("amount", 0.0))

        if from_acc not in db["accounts"]:
            return json.dumps({"error": f"Source account {from_acc} not found"})
        if to_acc not in db["accounts"]:
            return json.dumps({"error": f"Destination account {to_acc} not found or invalid routing"})
        if amount <= 0:
            return json.dumps({"error": "Transfer amount must be positive"})
        if db["accounts"][from_acc]["balance"] < amount:
            return json.dumps({
                "error": "Insufficient funds",
                "available_balance": db["accounts"][from_acc]["balance"],
                "requested_amount": amount,
            })

        db["accounts"][from_acc]["balance"] -= amount
        db["accounts"][to_acc]["balance"] += amount
        tid = f"TRF_{len(db['transfers']) + 1001}"
        record = {"transfer_id": tid, "from": from_acc, "to": to_acc, "amount": amount, "status": "completed"}
        db["transfers"].append(record)
        return json.dumps(record)

    if name == "freeze_card":
        cid = str(args.get("card_id", "")).strip()
        if cid not in db["cards"]:
            return json.dumps({"error": f"Card {cid} not found"})
        db["cards"][cid]["status"] = "frozen"
        db["cards"][cid]["freeze_reason"] = args.get("reason", "customer_request")
        return json.dumps({"card_id": cid, "status": "frozen", "message": "Card successfully frozen"})

    if name == "unfreeze_card":
        cid = str(args.get("card_id", "")).strip()
        if cid not in db["cards"]:
            return json.dumps({"error": f"Card {cid} not found"})
        db["cards"][cid]["status"] = "active"
        return json.dumps({"card_id": cid, "status": "active", "message": "Card unblocked and active"})

    if name == "waive_fee":
        aid = str(args.get("account_id", "")).strip()
        fid = str(args.get("fee_id", "")).strip()
        if fid not in db["transactions"]:
            return json.dumps({"error": f"Fee transaction {fid} not found"})
        fee = db["transactions"][fid]
        if fee["account_id"] != aid:
            return json.dumps({"error": "Fee does not match account"})
        # Refund fee amount
        db["accounts"][aid]["balance"] += fee["amount"]
        fee["status"] = "waived_refunded"
        return json.dumps({"status": "fee_waived", "fee_id": fid, "refund_amount": fee["amount"], "new_balance": db["accounts"][aid]["balance"]})

    if name == "file_dispute":
        txid = str(args.get("transaction_id", "")).strip()
        if txid not in db["transactions"]:
            return json.dumps({"error": f"Transaction {txid} not found"})
        dispute_id = f"DSP_{len(db['disputes']) + 501}"
        record = {
            "dispute_id": dispute_id,
            "transaction_id": txid,
            "amount": db["transactions"][txid]["amount"],
            "reason": args.get("dispute_reason", ""),
            "status": "under_investigation",
            "provisional_credit": True,
        }
        db["disputes"][dispute_id] = record
        return json.dumps(record)

    return f"unknown banking tool: {name}"


# ==============================================================================
# --- Terminal-Bench v4.0 Virtual File System (VFS) & Execution ---
# ==============================================================================

INITIAL_VFS = {
    "/var/log/nginx/access.log": """192.168.1.10 - - [29/Aug/2026:08:12:01] "GET /api/v1/health HTTP/1.1" 200 45
10.0.0.52 - - [29/Aug/2026:08:12:05] "POST /api/v1/auth HTTP/1.1" 500 128
192.168.1.10 - - [29/Aug/2026:08:12:10] "GET /api/v1/users HTTP/1.1" 200 512
172.16.4.88 - - [29/Aug/2026:08:12:14] "POST /api/v1/checkout HTTP/1.1" 500 240
10.0.0.52 - - [29/Aug/2026:08:12:19] "POST /api/v1/auth HTTP/1.1" 500 128
192.168.1.15 - - [29/Aug/2026:08:12:22] "GET /static/bundle.js HTTP/1.1" 304 0
10.0.0.99 - - [29/Aug/2026:08:12:30] "DELETE /api/v1/cache HTTP/1.1" 500 64
172.16.4.88 - - [29/Aug/2026:08:12:35] "POST /api/v1/checkout HTTP/1.1" 200 240
""",
    "/etc/nginx/conf.d/api.conf": """upstream backend_cluster {
    server 127.0.0.1:8080
}

server {
    listen 80;
    server_name api.internal.local;

    location / {
        proxy_pass http://backend_cluster;
        proxy_set_header Host $host
    }
}
""",
    "/workspace/config.py": """# Production Database Configuration
DB_TIMEOUT = 30
<<<<<<< HEAD
DATABASE_URL = "postgresql://app_prod:Secr3tP@ss@db-primary.internal.net:5432/analytics_prod"
=======
DATABASE_URL = "postgresql://root:root@localhost:5432/test_db"
>>>>>>> branch-experiment
CACHE_ENABLED = True
""",
    "/proc/simulated_ps": """PID   USER      %CPU  %MEM  COMMAND
1     root      0.1   0.2   /sbin/init
412   postgres  1.2   4.5   postgres: pooler worker
891   app       85.4  94.2  python -m workers.leak_worker --run-infinite
1024  nginx     0.4   0.8   nginx: worker process
""",
    "/app/services/auth.env": """SERVICE_NAME=auth-service
PORT=4001
API_KEY=${OLD_API_KEY}
ENCRYPTION_SALT=s4lt_v1
""",
    "/app/services/billing.env": """SERVICE_NAME=billing-service
PORT=4002
API_KEY=${OLD_API_KEY}
STRIPE_WEBHOOK=live_hook_91
""",
    "/db/migrations/004_users.json": """{
    "version": 4,
    "description": "add oauth tokens column",
    "columns": [
        {"name": "id", "type": "serial"},
        {"name": "email", "type": "varchar(255)"},
        {"name": "oauth_token", "type": "text"},
    ],
    "indices": [
        "idx_users_email",
    ]
}
"""
}


def create_terminal_state() -> dict:
    return copy.deepcopy(INITIAL_VFS)


def execute_terminal_tool(vfs: dict, name: str, args: dict) -> str:
    args = args or {}
    if name == "read_file":
        p = args.get("path", "").strip()
        if p in vfs:
            return vfs[p]
        return f"cat: {p}: No such file or directory"

    if name == "write_file":
        p = args.get("path", "").strip()
        content = args.get("content", "")
        vfs[p] = content
        return f"successfully wrote {len(content)} bytes to {p}"

    if name == "bash_exec":
        cmd = args.get("command", "").strip()
        # Simulated basic shell commands
        if cmd.startswith("cat "):
            p = cmd.split("cat ", 1)[1].strip()
            return vfs.get(p, f"cat: {p}: No such file or directory")
        if "grep 500" in cmd and "/var/log/nginx/access.log" in cmd:
            lines = [l for l in vfs.get("/var/log/nginx/access.log", "").splitlines() if "500" in l]
            return "\n".join(lines)
        if "kill" in cmd and "891" in cmd:
            ps = vfs.get("/proc/simulated_ps", "")
            vfs["/proc/simulated_ps"] = "\n".join([l for l in ps.splitlines() if "891" not in l])
            return "Process 891 terminated (SIGKILL)."
        if "sed" in cmd and "${OLD_API_KEY}" in cmd:
            for path in list(vfs.keys()):
                if path.endswith(".env"):
                    vfs[path] = vfs[path].replace("${OLD_API_KEY}", "${NEW_API_KEY}")
            return "Replaced all occurrences in .env files."
        if "ps" in cmd or "top" in cmd:
            return vfs.get("/proc/simulated_ps", "")
        if "nginx -t" in cmd:
            conf = vfs.get("/etc/nginx/conf.d/api.conf", "")
            if "proxy_set_header Host $host;" not in conf or "8000;" not in conf:
                return "nginx: [emerg] syntax error in /etc/nginx/conf.d/api.conf: missing semicolon or wrong upstream"
            return "nginx: the configuration file /etc/nginx/nginx.conf syntax is ok\nnginx: configuration file /etc/nginx/nginx.conf test is successful"
        return f"[command executed]: {cmd}"

    return f"unknown terminal tool: {name}"


# ==============================================================================
# --- 1. Core Tool-Calling & Agentic Evaluation Scenarios (31) ---
# ==============================================================================

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
     "messages": [{"role": "user", "content": "Compare the current temperatures in Paris and Tokyo."}],
     "expected_calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}, {"name": "get_weather", "arguments": {"city": "Tokyo"}}]},
    {"id": "p02", "category": "parallel", "tools": ["get_stock_price"],
     "messages": [{"role": "user", "content": "Check the share prices for both AAPL and TSLA."}],
     "expected_calls": [{"name": "get_stock_price", "arguments": {"symbol": "AAPL"}}, {"name": "get_stock_price", "arguments": {"symbol": "TSLA"}}]},
    {"id": "p03", "category": "parallel", "tools": ["get_weather", "get_flight_status"],
     "messages": [{"role": "user", "content": "I am traveling to London on flight DL301. Give me the weather in London and the flight status."}],
     "expected_calls": [{"name": "get_weather", "arguments": {"city": "London"}}, {"name": "get_flight_status", "arguments": {"flight_number": "DL301"}}]},
    {"id": "p04", "category": "parallel", "tools": ["get_stock_price"],
     "messages": [{"role": "user", "content": "Get the stock quotes for NVDA, MSFT, and AAPL simultaneously."}],
     "expected_calls": [{"name": "get_stock_price", "arguments": {"symbol": "NVDA"}}, {"name": "get_stock_price", "arguments": {"symbol": "MSFT"}}, {"name": "get_stock_price", "arguments": {"symbol": "AAPL"}}]},
    {"id": "p05", "category": "parallel", "tools": ["get_weather"],
     "messages": [{"role": "user", "content": "What are the temperatures in New York, Madrid, and Berlin?"}],
     "expected_calls": [{"name": "get_weather", "arguments": {"city": "New York"}}, {"name": "get_weather", "arguments": {"city": "Madrid"}}, {"name": "get_weather", "arguments": {"city": "Berlin"}}]},
    {"id": "p06", "category": "parallel", "tools": ["check_inventory"],
     "messages": [{"role": "user", "content": "Check inventory for 'Laptop Pro 16' and 'Laptop Air 15'."}],
     "expected_calls": [{"name": "check_inventory", "arguments": {"product_name": "Laptop Pro 16"}}, {"name": "check_inventory", "arguments": {"product_name": "Laptop Air 15"}}]},

    # 3. Multi-Turn Dependency Chains (tau-bench / GAIA Level 2)
    {"id": "m01", "category": "multi_turn", "tools": ["search_products", "check_inventory"],
     "messages": [{"role": "user", "content": "Find all laptops in the catalog, pick the one with 32GB RAM, and tell me if it is in stock."}],
     "expected_calls": [{"name": "search_products", "arguments": {"query": "laptop"}}],
     "expected_answer": ["3", "yes", "in stock"]},
    {"id": "m02", "category": "multi_turn", "tools": ["search_products", "calculate_tax"],
     "messages": [{"role": "user", "content": "Find the price of Phone Ultra in our catalog and calculate total cost including California (CA) sales tax."}],
     "expected_calls": [{"name": "search_products", "arguments": {"query": "phone"}}],
     "expected_answer": ["1081.42", "1,081.42", "81.42"]},
    {"id": "m03", "category": "multi_turn", "tools": ["get_weather", "calculator"],
     "messages": [{"role": "user", "content": "Get the temperatures of Paris and Berlin in Celsius, compute their sum, and multiply the sum by 1.5."}],
     "expected_calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}],
     "expected_answer": ["60"]},
    {"id": "m04", "category": "multi_turn", "tools": ["search_products", "check_inventory", "calculate_tax"],
     "messages": [{"role": "user", "content": "Find 'Laptop Air 15' price and stock, and compute its total price with New York (NY) tax."}],
     "expected_calls": [{"name": "search_products", "arguments": {"query": "laptop"}}],
     "expected_answer": ["978.79", "978.8", "79.79"]},
    {"id": "m05", "category": "multi_turn", "tools": ["get_stock_price", "calculator"],
     "messages": [{"role": "user", "content": "What is the total value of a portfolio with 10 shares of NVDA and 5 shares of AAPL?"}],
     "expected_calls": [{"name": "get_stock_price", "arguments": {"symbol": "NVDA"}}],
     "expected_answer": ["2325", "2,325"]},
    {"id": "m06", "category": "multi_turn", "tools": ["get_weather", "get_flight_status", "send_email"],
     "messages": [{"role": "user", "content": "Check DL301 status and London weather, then email summary to team@example.com."}],
     "expected_calls": [{"name": "get_flight_status", "arguments": {"flight_number": "DL301"}}],
     "expected_answer": ["email", "sent", "london"]},

    # 4. Distractor Tool Ambiguity
    {"id": "d01", "category": "distractor_tools",
     "tools": ["get_weather", "get_weather_forecast", "get_historical_weather", "get_stock_price", "get_flight_status", "search_products", "check_inventory", "calculate_tax", "book_flight", "cancel_booking", "send_email", "calculator"],
     "messages": [{"role": "user", "content": "What was the weather in Paris on 2024-01-15?"}],
     "expected_calls": [{"name": "get_historical_weather", "arguments": {"city": "Paris", "date": "2024-01-15"}}]},
    {"id": "d02", "category": "distractor_tools",
     "tools": ["get_weather", "get_weather_forecast", "get_historical_weather", "get_stock_price", "get_flight_status", "search_products", "check_inventory", "calculate_tax", "book_flight", "cancel_booking", "send_email", "calculator"],
     "messages": [{"role": "user", "content": "Will it rain in Tokyo in 3 days?"}],
     "expected_calls": [{"name": "get_weather_forecast", "arguments": {"city": "Tokyo", "days_ahead": 3}}]},

    # 5. No-Tool Hallucination Restraint
    {"id": "nt01", "category": "no_tool", "tools": ["get_weather", "get_stock_price", "calculator"],
     "messages": [{"role": "user", "content": "What is the capital of France? Answer in one short sentence."}],
     "expected_calls": [], "expected_answer": "Paris"},
    {"id": "nt02", "category": "no_tool", "tools": ["search_products", "book_flight", "send_email"],
     "messages": [{"role": "user", "content": "Explain the concept of quantum superposition in simple terms."}],
     "expected_calls": [], "expected_answer": "superposition"},
    {"id": "nt03", "category": "no_tool", "tools": ["get_flight_status", "calculate_tax", "get_weather"],
     "messages": [{"role": "user", "content": "Write a four-line haiku about autumn leaves."}],
     "expected_calls": [], "expected_answer": "autumn"},
    {"id": "nt04", "category": "no_tool", "tools": ["get_stock_price", "calculator", "check_inventory"],
     "messages": [{"role": "user", "content": "Who painted the Mona Lisa?"}],
     "expected_calls": [], "expected_answer": "Leonardo da Vinci"},

    # 6. Error Recovery & Dynamic Rollback
    {"id": "err01", "category": "error_recovery", "tools": ["get_flight_status"],
     "messages": [{"role": "user", "content": "Check flight UA999 status. If that flight does not exist, check UA202 instead."}],
     "expected_calls": [{"name": "get_flight_status", "arguments": {"flight_number": "UA999"}}],
     "expected_answer": ["delayed", "90 minutes"]},
    {"id": "err02", "category": "error_recovery", "tools": ["check_inventory"],
     "messages": [{"role": "user", "content": "Check stock for 'Laptop Lite 14'. If it is out of stock, check 'Laptop Pro 16'."}],
     "expected_calls": [{"name": "check_inventory", "arguments": {"product_name": "Laptop Lite 14"}}],
     "expected_answer": ["3", "pro 16"]},

    # 7. Complex Nested Argument Extraction
    {"id": "c01", "category": "complex_args", "tools": ["book_flight"],
     "messages": [{"role": "user", "content": "Book flight AA100 round trip for Alice (business class) and Bob (economy class)."}],
     "expected_calls": [{
         "name": "book_flight",
         "arguments": {
             "flight_number": "AA100",
             "round_trip": True,
             "passengers": [{"name": "Alice", "seat_class": "business"}, {"name": "Bob", "seat_class": "economy"}]
         }
     }]},
]


# ==============================================================================
# --- 2. Google IFEval Hard Conjunction Scenarios (6) ---
# ==============================================================================

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


# ==============================================================================
# --- 3. Competition Math Reasoning (AIME / Olympiad) (6) ---
# ==============================================================================

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


# ==============================================================================
# --- 4. GPQA Diamond (PhD-Level Science Reasoning) (12 Expanded) ---
# ==============================================================================

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
    {
        "id": "gpqa_07",
        "domain": "Astrophysics & General Relativity",
        "prompt": "For a static non-rotating Schwarzschild black hole of mass M, what is the radius r of the circular orbit of photons (the photon sphere)?\n(A) r = 2 * G * M / c^2\n(B) r = 3 * G * M / c^2\n(C) r = 1.5 * G * M / c^2\n(D) r = 6 * G * M / c^2\nState your step-by-step reasoning and end with your final letter choice as #### <Letter>.",
        "expected_answer": "B",
    },
    {
        "id": "gpqa_08",
        "domain": "CRISPR & Genome Engineering",
        "prompt": "What is the canonical Protospacer Adjacent Motif (PAM) sequence recognized by Streptococcus pyogenes Cas9 (SpCas9) on the non-target DNA strand directly adjacent to the 3' end of the target protospacer?\n(A) 5'-TTTV-3'\n(B) 5'-NGG-3'\n(C) 5'-NNNNGATT-3'\n(D) 5'-CC-3'\nState your step-by-step reasoning and end with your final letter choice as #### <Letter>.",
        "expected_answer": "B",
    },
    {
        "id": "gpqa_09",
        "domain": "Solid State Physics",
        "prompt": "Which semiconductor material exhibits a direct band gap, making it highly efficient for optical emission in semiconductor lasers and LEDs without requiring phonon-assisted momentum transfer?\n(A) Silicon (Si)\n(B) Germanium (Ge)\n(C) Gallium Arsenide (GaAs)\n(D) Diamond Carbon (C)\nState your step-by-step reasoning and end with your final letter choice as #### <Letter>.",
        "expected_answer": "C",
    },
    {
        "id": "gpqa_10",
        "domain": "Physical Chemistry & Thermodynamics",
        "prompt": "For the exothermic gas-phase synthesis of ammonia: N_2(g) + 3H_2(g) <=> 2NH_3(g) with Delta H < 0, which operational change will shift the equilibrium position to yield a HIGHER equilibrium mole fraction of NH_3?\n(A) Increasing temperature at constant pressure\n(B) Increasing total system pressure at constant temperature\n(C) Adding an inert gas like Argon at constant total pressure\n(D) Decreasing reactant concentration of N_2\nState your step-by-step reasoning and end with your final letter choice as #### <Letter>.",
        "expected_answer": "B",
    },
    {
        "id": "gpqa_11",
        "domain": "Quantum Information",
        "prompt": "In standard quantum teleportation of an unknown single-qubit state |psi>, how many classical bits of measurement outcome must Alice transmit to Bob for Bob to reconstruct the exact state |psi>?\n(A) 1 classical bit\n(B) 2 classical bits\n(C) 4 classical bits\n(D) 0 classical bits (instantaneous non-local collapse)\nState your step-by-step reasoning and end with your final letter choice as #### <Letter>.",
        "expected_answer": "B",
    },
    {
        "id": "gpqa_12",
        "domain": "Polymer Chemistry",
        "prompt": "A monodisperse synthetic polymer sample has identical molar mass for every single polymer chain. What is the value of its Polydispersity Index (PDI = M_w / M_n)?\n(A) 0\n(B) Exactly 1.0\n(C) Exactly 2.0\n(D) Infinity\nState your step-by-step reasoning and end with your final letter choice as #### <Letter>.",
        "expected_answer": "B",
    },
]


# ==============================================================================
# --- 5. HumanEval+ / LeetCode Stateful Data Structures (6) ---
# ==============================================================================

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

    def getMin() -> int:
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


# ==============================================================================
# --- 5. CritPt (Competition Physics & Advanced Mathematical Reasoning) (8) ---
# ==============================================================================

CRITPT_SCENARIOS = [
    {
        "id": "critpt_01",
        "prompt": "A relativistic light source emitting radiation at frequency f_0 moves directly away from a stationary observer at velocity v = 0.8c. In terms of the relativistic Doppler effect, what is the ratio f_0 / f_obs of emitted frequency to observed frequency? State the exact integer ratio as #### <integer>.",
        "expected_answer": 3,
    },
    {
        "id": "critpt_02",
        "prompt": "In Landau mean-field theory for continuous second-order phase transitions, the order parameter m behaves near the critical temperature T_c as m ~ (T_c - T)^beta for T < T_c. What is the denominator of the mean-field critical exponent beta = 1/2? State the integer denominator as #### <integer>.",
        "expected_answer": 2,
    },
    {
        "id": "critpt_03",
        "prompt": "A reversible Carnot heat engine operates between reservoirs at T_hot = 600 K and T_cold = 300 K. In each cycle, the engine absorbs Q_in = 720 Joules of heat from the hot reservoir. What is the net mechanical work output W (in Joules) delivered by the engine per cycle? State your answer as #### <integer>.",
        "expected_answer": 360,
    },
    {
        "id": "critpt_04",
        "prompt": "An ideal LC circuit contains an inductor L = 4 mH (0.004 H) and a capacitor C = 1000 microfarads (0.001 F). What is the natural resonant angular frequency omega_0 (in radians per second)? State the integer value as #### <integer>.",
        "expected_answer": 500,
    },
    {
        "id": "critpt_05",
        "prompt": "A satellite in a circular orbit of radius r_0 = 10,000 km has an orbital period of T_0 = 100 minutes. If the orbital radius is increased such that the new radius r_1 = 4 * r_0 = 40,000 km, what is the ratio T_1 / T_0 of the new orbital period to the initial orbital period according to Kepler's Third Law? State the integer factor as #### <integer>.",
        "expected_answer": 8,
    },
    {
        "id": "critpt_06",
        "prompt": "In a 3D Maxwell-Boltzmann gas at absolute temperature T, the root-mean-square speed of gas A with molar mass M_A = 4 g/mol is compared to gas B with molar mass M_B = 16 g/mol at the same temperature. What is the ratio v_rms(A) / v_rms(B)? State the integer ratio as #### <integer>.",
        "expected_answer": 2,
    },
    {
        "id": "critpt_07",
        "prompt": "For a 1D quantum harmonic oscillator of mass m and frequency omega in energy state n = 3, what is the energy eigenvalue E_3 in units of (1/2) * hbar * omega? (i.e. E_n = (n + 1/2)*hbar*omega = k * (hbar*omega/2)). State the integer numerator k as #### <integer>.",
        "expected_answer": 7,
    },
    {
        "id": "critpt_08",
        "prompt": "In superconductivity, magnetic flux through a superconducting ring is quantized in discrete integer multiples of the flux quantum Phi_0 = h / (q). What is the integer charge multiple q in terms of the elementary charge e (q = N * e, corresponding to Cooper pairs)? State the integer N as #### <integer>.",
        "expected_answer": 2,
    },
]


# ==============================================================================
# --- 6. Humanity's Last Exam (HLE Curated Frontier Problems) (10) ---
# ==============================================================================

HLE_SCENARIOS = [
    {
        "id": "hle_01",
        "domain": "Cooperative Game Theory / Shapley Value",
        "prompt": "In a 3-player cooperative game N = {1, 2, 3}, the characteristic function values are: v({1}) = 0, v({2}) = 0, v({3}) = 0, v({1,2}) = 30, v({1,3}) = 30, v({2,3}) = 0, and v({1,2,3}) = 60. Using the Shapley value formula phi_i(v), compute the unique equitable payoff phi_1 for player 1. State the final integer value as #### <integer>.",
        "expected_answer": 35,
    },
    {
        "id": "hle_02",
        "domain": "Algebraic Topology",
        "prompt": "What is the Euler characteristic chi(Sigma_g) of a closed, connected, orientable 2-dimensional surface of genus g = 3 (a triple-holed torus), given by the topological formula chi = 2 - 2g?\n(A) 2\n(B) -4\n(C) -6\n(D) 0\nState your step-by-step reasoning and conclude with #### <Letter>.",
        "expected_answer": "B",
    },
    {
        "id": "hle_03",
        "domain": "Computational Complexity Theory",
        "prompt": "According to the classical circuit lower bounds established by Smolensky and Razborov, which complexity class CANNOT compute the PARITY function on n input bits with polynomial-size, constant-depth circuits?\n(A) NC^1\n(B) AC^0\n(C) P/poly\n(D) TC^0\nState your step-by-step reasoning and conclude with #### <Letter>.",
        "expected_answer": "B",
    },
    {
        "id": "hle_04",
        "domain": "Formal Logic & Epistemology",
        "prompt": "In Gödel-Löb provability logic (GL), which modal axiom schema expresses Löb's Theorem characterizing mathematical provability Box(A)?\n(A) Box(A) -> A\n(B) Box(Box(A) -> A) -> Box(A)\n(C) Box(A) -> Box(Box(A))\n(D) Diamond(A) -> Box(Diamond(A))\nState your step-by-step reasoning and conclude with #### <Letter>.",
        "expected_answer": "B",
    },
    {
        "id": "hle_05",
        "domain": "Geometric Group Theory",
        "prompt": "Consider the free group F_2 on two free generators {a, b} with symmetric generating set S = {a, a^-1, b, b^-1}. For length n >= 1 in the word metric, what is the exact spherical growth function sigma(n) counting the number of reduced words of length n?\n(A) 2^n\n(B) 4 * 3^(n-1)\n(C) 4^n\n(D) 3^n\nState your step-by-step reasoning and conclude with #### <Letter>.",
        "expected_answer": "B",
    },
    {
        "id": "hle_06",
        "domain": "Molecular Neuroscience",
        "prompt": "A mammalian neuron at 37 deg C has intracellular [K+] = 140 mM and extracellular [K+] = 5 mM. Using the Nernst equation E_K = (R*T / z*F) * ln([K+]_out / [K+]_in) where 2.303 * R * T / F = 61.5 mV at 37 deg C, calculate the potassium equilibrium potential E_K rounded to the nearest integer mV. State your answer as #### <integer>.",
        "expected_answer": -89,
    },
    {
        "id": "hle_07",
        "domain": "General Relativity & Black Holes",
        "prompt": "For a test particle in a bound circular orbit around a non-rotating Schwarzschild black hole of mass M and gravitational radius r_g = G*M / c^2, what is the radial distance r of the Innermost Stable Circular Orbit (ISCO) in units of r_g? (i.e. r_ISCO = k * r_g). State the integer k as #### <integer>.",
        "expected_answer": 6,
    },
    {
        "id": "hle_08",
        "domain": "Quantum Information Theory",
        "prompt": "According to the Holevo bound in quantum information, what is the maximum accessible mutual information (in classical bits) that Bob can extract from transmitting one single pure quantum qubit without prior entanglement?\n(A) Exactly 1 bit\n(B) Exactly 2 bits\n(C) ln(2) bits\n(D) Infinite bits\nState your step-by-step reasoning and conclude with #### <Letter>.",
        "expected_answer": "A",
    },
    {
        "id": "hle_09",
        "domain": "Population Genetics",
        "prompt": "In a diploid population of effective size N_e, what is the neutral fixation probability of a newly arisen single unique neutral mutation (initial allele count = 1)?\n(A) 1 / (2 * N_e)\n(B) 1 / N_e\n(C) 2 * N_e\n(D) 1 / 2\nState your step-by-step reasoning and conclude with #### <Letter>.",
        "expected_answer": "A",
    },
    {
        "id": "hle_10",
        "domain": "Combinatorial Number Theory",
        "prompt": "By van der Waerden's theorem, W(k, r) is the smallest integer N such that any r-coloring of {1, 2, ..., N} contains a monochromatic arithmetic progression of length k. What is the value of W(3, 2)? State the integer value as #### <integer>.",
        "expected_answer": 9,
    },
]


# ==============================================================================
# --- 7. T3-Banking (tau-bench Multi-Turn Stateful Agentic) (8) ---
# ==============================================================================

BANKING_SCENARIOS = [
    {
        "id": "bank_01",
        "category": "banking",
        "tools": ["get_user_profile", "get_account_balance", "transfer_funds"],
        "user_id": "USR_101",
        "messages": [
            {"role": "user", "content": "Hi, I am Sarah Connor (user ID: USR_101). Please check my checking account (ACC_CHK_101) balance and transfer $250.00 from it to my savings account (ACC_SAV_101)."}
        ],
        "expected_transfers": [{"from": "ACC_CHK_101", "to": "ACC_SAV_101", "amount": 250.0}],
        "expected_answer": ["250", "transferred", "completed"],
    },
    {
        "id": "bank_02",
        "category": "banking",
        "tools": ["list_transactions", "freeze_card", "file_dispute"],
        "user_id": "USR_101",
        "messages": [
            {"role": "user", "content": "I noticed an unauthorized charge of $89.99 (transaction TX_8901) on my debit card CARD_4321. Immediately freeze my card and file a fraud dispute for TX_8901."}
        ],
        "expected_frozen_cards": ["CARD_4321"],
        "expected_disputes": ["TX_8901"],
        "expected_answer": ["freeze", "frozen", "dispute", "filed"],
    },
    {
        "id": "bank_03",
        "category": "banking",
        "tools": ["list_transactions", "waive_fee"],
        "user_id": "USR_101",
        "messages": [
            {"role": "user", "content": "I was charged an overdraft fee of $35 (fee ID FEE_OD_901) on my checking account ACC_CHK_101. As a loyal customer, please waive and refund this fee for me."}
        ],
        "expected_waived_fees": ["FEE_OD_901"],
        "expected_answer": ["waived", "refunded", "35"],
    },
    {
        "id": "bank_04",
        "category": "banking",
        "tools": ["get_account_balance", "transfer_funds"],
        "user_id": "USR_202",
        "messages": [
            {"role": "user", "content": "I am David Martinez (USR_202). Transfer $5000.00 from my checking account ACC_CHK_202 to external account ACC_EXTERNAL_BOB."}
        ],
        "expected_transfers": [],  # Must refuse due to insufficient funds ($450 balance)
        "expected_answer": ["insufficient", "450", "cannot"],
    },
    {
        "id": "bank_05",
        "category": "banking",
        "tools": ["list_transactions"],
        "user_id": "USR_101",
        "messages": [
            {"role": "user", "content": "Review my recent checking transactions (ACC_CHK_101) and calculate the total amount I spent on groceries across Whole Foods and Trader Joe's."}
        ],
        "expected_answer": ["196.50", "196.5"],
    },
    {
        "id": "bank_06",
        "category": "banking",
        "tools": ["unfreeze_card", "get_user_profile"],
        "user_id": "USR_101",
        "messages": [
            {"role": "user", "content": "My credit card CARD_8877 is currently blocked. I confirmed with security that it was a false alarm. Please unfreeze CARD_8877."}
        ],
        "expected_unfrozen_cards": ["CARD_8877"],
        "expected_answer": ["unfrozen", "active", "CARD_8877"],
    },
    {
        "id": "bank_07",
        "category": "banking",
        "tools": ["transfer_funds"],
        "user_id": "USR_101",
        "messages": [
            {"role": "user", "content": "First, attempt to transfer $100.00 from ACC_CHK_101 to invalid account ACC_INVALID_999. When that fails, transfer the $100.00 to ACC_SAV_101 instead."}
        ],
        "expected_transfers": [{"from": "ACC_CHK_101", "to": "ACC_SAV_101", "amount": 100.0}],
        "expected_answer": ["ACC_SAV_101", "100", "completed"],
    },
    {
        "id": "bank_08",
        "category": "banking",
        "tools": ["get_account_balance", "transfer_funds"],
        "user_id": "USR_101",
        "messages": [
            {"role": "user", "content": "I want to rebalance my funds. Transfer $1000.00 from my savings account (ACC_SAV_101) into my checking account (ACC_CHK_101)."}
        ],
        "expected_transfers": [{"from": "ACC_SAV_101", "to": "ACC_CHK_101", "amount": 1000.0}],
        "expected_answer": ["1000", "transferred", "ACC_CHK_101"],
    },
]


# ==============================================================================
# --- 8. GDPval-AA v2 (Economic & White-Collar Structured Workflows) (6) ---
# ==============================================================================

GDPVAL_SCENARIOS = [
    {
        "id": "gdpval_01",
        "rule_id": "gdpval_balance_sheet_reconciliation",
        "prompt": """You are an expert CPA performing a financial balance sheet audit.
Given the company ledger figures:
- Cash & Equivalents: $450,000
- Accounts Receivable: $280,000
- Inventory: $190,000
- Property, Plant & Equipment: $1,200,000
- Accounts Payable: $210,000
- Short-Term Notes Payable: $90,000
- Long-Term Debt: $800,000
- Common Stock: $500,000
- Retained Earnings: $520,000

Perform reconciliation (Assets = Liabilities + Equity).
Output valid JSON adhering strictly to this schema:
{
  "total_assets": number,
  "total_liabilities": number,
  "total_equity": number,
  "is_balanced": boolean,
  "reconciliation_variance": number
}
Provide only the raw JSON object.""",
    },
    {
        "id": "gdpval_02",
        "rule_id": "gdpval_vendor_sla_audit",
        "prompt": """You are an Enterprise Procurement Auditor reviewing vendor cloud SLA uptime.
Contract SLA terms:
- Required Uptime: 99.9% (max allowed monthly downtime: 43.2 minutes).
- Tier 1 penalty (99.0% - 99.89%): 10% monthly service fee credit.
- Tier 2 penalty (< 99.0%): 25% monthly service fee credit.
Monthly fee: $50,000.00.
Actual recorded downtime this month across 3 incidents: 120 minutes (uptime: 99.72%).

Calculate the audit outcome and output JSON adhering to:
{
  "actual_uptime_percent": number,
  "sla_breached": boolean,
  "penalty_tier": string,
  "credit_percentage": number,
  "credit_amount_usd": number
}
Provide only the raw JSON object.""",
    },
    {
        "id": "gdpval_03",
        "rule_id": "gdpval_saas_metrics",
        "prompt": """You are a SaaS Financial Analyst calculating quarterly cohort retention.
Q1 Starting ARR: $1,000,000
Q1 Churned ARR: $50,000
Q1 Contraction ARR: $30,000
Q1 Expansion ARR from existing customers: $120,000
New Logo ARR: $200,000

Calculate Gross Revenue Retention (GRR = (Starting ARR - Churn - Contraction) / Starting ARR) and Net Revenue Retention (NRR = (Starting ARR - Churn - Contraction + Expansion) / Starting ARR).
Output valid JSON:
{
  "grr_percent": number,
  "nrr_percent": number,
  "ending_cohort_arr": number,
  "net_expansion_arr": number
}
Provide only the raw JSON object.""",
    },
    {
        "id": "gdpval_04",
        "rule_id": "gdpval_payroll_withholding",
        "prompt": """Calculate standard monthly US payroll taxes for an employee with gross monthly salary of $10,000.00:
- Social Security tax: 6.2% of gross
- Medicare tax: 1.45% of gross
- Flat federal income tax withholding rate: 22.0%
- State income tax rate: 5.0%

Output JSON with computed values:
{
  "gross_pay": number,
  "social_security_tax": number,
  "medicare_tax": number,
  "federal_tax": number,
  "state_tax": number,
  "total_deductions": number,
  "net_take_home_pay": number
}
Provide only the raw JSON object.""",
    },
    {
        "id": "gdpval_05",
        "rule_id": "gdpval_cloud_optimization",
        "prompt": """You are a FinOps Cloud Architect. An engineering team runs 20 on-demand c6g.2xlarge EC2 instances at $0.272/hour (730 hours/month = $198.56/instance/month; total on-demand: $3,971.20/month).
Switching to 3-Year All-Upfront Reserved Instances reduces the effective hourly rate by 60% (saving 60% of compute cost).
Compute monthly savings and 36-month total contract savings.
Output JSON:
{
  "current_monthly_spend": number,
  "savings_percentage": number,
  "monthly_savings_usd": number,
  "three_year_total_savings_usd": number
}
Provide only the raw JSON object.""",
    },
    {
        "id": "gdpval_06",
        "rule_id": "gdpval_msa_compliance_audit",
        "prompt": """Audit an enterprise contract clause:
'The total aggregate liability of Supplier under this Agreement for all claims shall be uncapped and unlimited regardless of legal theory.'
Corporate Playbook Rule: Supplier liability must be capped at maximum 2x annual contract value ($200,000).

Output JSON audit result:
{
  "clause_compliant": boolean,
  "risk_level": string,
  "flagged_issue": string,
  "recommended_cap_usd": number
}
Provide only the raw JSON object.""",
    },
]


# ==============================================================================
# --- 9. AA-Omniscience (Hallucination Restraint & Traps) (10) ---
# ==============================================================================

OMNISCIENCE_SCENARIOS = [
    {
        "id": "omni_01",
        "type": "trap_false_premise",
        "prompt": "In what calendar year did Alan Turing publish the official Python 3.0 programming language specification?",
        "expected_trap": True,
        "required_denial": ["did not", "never", "1954", "guido", "false premise", "unrelated"],
    },
    {
        "id": "omni_02",
        "type": "trap_nonexistent_event",
        "prompt": "What was the date of the first commercial supersonic passenger flight operated by Concorde connecting New York JFK directly to Tokyo Haneda?",
        "expected_trap": True,
        "required_denial": ["never", "did not fly", "did not operate", "no commercial flight", "unrelated", "fictional"],
    },
    {
        "id": "omni_03",
        "type": "factual_precision",
        "prompt": "What is the exact atomic number and standard atomic weight (to two decimal places) of the chemical element Ytterbium (Yb)? State your concise factual answer.",
        "expected_trap": False,
        "expected_facts": ["70", "173.05"],
    },
    {
        "id": "omni_04",
        "type": "trap_fictional_entity",
        "prompt": "Who served as Prime Minister of the Commonwealth of Atlantis during the 1973 OPEC oil crisis?",
        "expected_trap": True,
        "required_denial": ["atlantis is fictional", "fictional", "mythological", "does not exist", "mythical", "no such"],
    },
    {
        "id": "omni_05",
        "type": "trap_false_attribution",
        "prompt": "Which Nobel Prize in Physics did Albert Einstein win for his 1905 paper on the theory of Special Relativity?",
        "expected_trap": True,
        "required_denial": ["photoelectric", "did not win for special relativity", "photoelectric effect", "1921"],
    },
    {
        "id": "omni_06",
        "type": "factual_precision",
        "prompt": "In the International System of Units (SI), what is the exact defined numerical value of the Boltzmann constant k_B in J/K? State your answer.",
        "expected_trap": False,
        "expected_facts": ["1.380649", "10^-23", "e-23"],
    },
    {
        "id": "omni_07",
        "type": "trap_impossible_chemical",
        "prompt": "What is the boiling point and stable crystal structure of carbon hexachloride (CCl6) at standard atmospheric pressure?",
        "expected_trap": True,
        "required_denial": ["does not exist", "impossible", "valence", "4 bonds", "tetravalent", "cannot form"],
    },
    {
        "id": "omni_08",
        "type": "factual_precision",
        "prompt": "What was the launch year and primary destination target of the European Space Agency's JUICE (Jupiter Icy Moons Explorer) space mission?",
        "expected_trap": False,
        "expected_facts": ["2023", "jupiter", "ganymede"],
    },
    {
        "id": "omni_09",
        "type": "trap_false_count",
        "prompt": "How many natural planetary moons does the planet Mercury possess according to verified astronomical observation?",
        "expected_trap": False,
        "expected_facts": ["0", "zero", "none"],
    },
    {
        "id": "omni_10",
        "type": "trap_nonexistent_mission",
        "prompt": "What was the name of the Apollo 18 astronaut who walked on the Martian surface in July 1985?",
        "expected_trap": True,
        "required_denial": ["never landed", "did not land on mars", "apollo ended", "apollo 17", "fictional", "never walked"],
    },
]


# ==============================================================================
# --- 10. SciCode (Scientific Computing with Sandboxed Subprocess Unit Tests) (6) ---
# ==============================================================================

SCICODE_SCENARIOS = [
    {
        "id": "scicode_01",
        "entry_point": "quantum_density_purity",
        "prompt": '''def quantum_density_purity(rho: list[list[float]]) -> float:
    """ Compute the state purity gamma = Tr(rho^2) for a quantum density matrix rho (given as a square 2D list of floats).
    Purity is 1.0 for pure states, and 1/d <= gamma < 1.0 for mixed states in d dimensions.
    """
''',
        "test": '''import math
# Pure state [[1.0, 0.0], [0.0, 0.0]] -> purity = 1.0
assert math.isclose(quantum_density_purity([[1.0, 0.0], [0.0, 0.0]]), 1.0, rel_tol=1e-5)
# Maximally mixed state [[0.5, 0.0], [0.0, 0.5]] -> purity = 0.5
assert math.isclose(quantum_density_purity([[0.5, 0.0], [0.0, 0.5]]), 0.5, rel_tol=1e-5)
# 3D state [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]] -> purity = 0.5
assert math.isclose(quantum_density_purity([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]]), 0.5, rel_tol=1e-5)
''',
    },
    {
        "id": "scicode_02",
        "entry_point": "lennard_jones_potential",
        "prompt": '''def lennard_jones_potential(r: float, epsilon: float = 1.0, sigma: float = 1.0) -> float:
    """ Compute the Lennard-Jones interatomic potential V(r) = 4 * epsilon * [(sigma/r)^12 - (sigma/r)^6].
    """
''',
        "test": '''import math
# At r = sigma, V(sigma) = 4 * eps * (1 - 1) = 0.0
assert math.isclose(lennard_jones_potential(1.0, 1.0, 1.0), 0.0, abs_tol=1e-5)
# At r = 2^(1/6) * sigma (minimum of potential), V = -epsilon = -1.0
r_min = 2.0 ** (1.0 / 6.0)
assert math.isclose(lennard_jones_potential(r_min, 1.0, 1.0), -1.0, rel_tol=1e-4)
# With eps = 2.5, sigma = 1.2
assert math.isclose(lennard_jones_potential(1.2, 2.5, 1.2), 0.0, abs_tol=1e-5)
''',
    },
    {
        "id": "scicode_03",
        "entry_point": "rk4_step",
        "prompt": '''def rk4_step(dydt_func, y: float, t: float, dt: float) -> float:
    """ Perform a single step of the 4th-Order Classical Runge-Kutta (RK4) numerical integrator.
    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt*k1/2)
    k3 = f(t + dt/2, y + dt*k2/2)
    k4 = f(t + dt, y + dt*k3)
    y_next = y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    """
''',
        "test": '''import math
# dy/dt = -y (analytical solution: y(t) = exp(-t))
f = lambda t, y: -y
y0 = 1.0
dt = 0.1
y1 = rk4_step(f, y0, 0.0, dt)
assert math.isclose(y1, math.exp(-0.1), rel_tol=1e-5)
# Take 10 steps to t = 1.0
curr_y = 1.0
for step in range(10):
    curr_y = rk4_step(f, curr_y, step * 0.1, 0.1)
assert math.isclose(curr_y, math.exp(-1.0), rel_tol=1e-4)
''',
    },
    {
        "id": "scicode_04",
        "entry_point": "trapezoidal_integrate",
        "prompt": '''def trapezoidal_integrate(func, a: float, b: float, n_intervals: int = 1000) -> float:
    """ Compute numerical definite integral of func from a to b using composite Trapezoidal rule with n_intervals.
    """
''',
        "test": '''import math
# integral of sin(x) from 0 to pi = 2.0
val = trapezoidal_integrate(math.sin, 0.0, math.pi, 1000)
assert math.isclose(val, 2.0, rel_tol=1e-4)
# integral of x^2 from 0 to 3 = 9.0
val2 = trapezoidal_integrate(lambda x: x**2, 0.0, 3.0, 1000)
assert math.isclose(val2, 9.0, rel_tol=1e-4)
''',
    },
    {
        "id": "scicode_05",
        "entry_point": "matrix_vector_multiply",
        "prompt": '''def matrix_vector_multiply(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """ Compute matrix-vector product y = A * x for a 2D matrix A and 1D vector x.
    """
''',
        "test": '''import math
A = [[1.0, 2.0], [3.0, 4.0]]
x = [2.0, 3.0]
# y = [1*2 + 2*3, 3*2 + 4*3] = [8.0, 18.0]
y = matrix_vector_multiply(A, x)
assert math.isclose(y[0], 8.0, rel_tol=1e-5)
assert math.isclose(y[1], 18.0, rel_tol=1e-5)
# 3x3 identity
I = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
v = [4.5, -2.1, 7.8]
y_id = matrix_vector_multiply(I, v)
assert all(math.isclose(a, b, rel_tol=1e-5) for a, b in zip(y_id, v))
''',
    },
    {
        "id": "scicode_06",
        "entry_point": "euclidean_distance_matrix",
        "prompt": '''def euclidean_distance_matrix(points: list[list[float]]) -> list[list[float]]:
    """ Compute NxN pairwise Euclidean distance matrix D where D[i][j] = sqrt(sum((points[i][k] - points[j][k])^2)).
    """
''',
        "test": '''import math
pts = [[0.0, 0.0], [3.0, 4.0], [0.0, 4.0]]
D = euclidean_distance_matrix(pts)
assert math.isclose(D[0][0], 0.0, abs_tol=1e-5)
assert math.isclose(D[0][1], 5.0, rel_tol=1e-5)
assert math.isclose(D[0][2], 4.0, rel_tol=1e-5)
assert math.isclose(D[1][2], 3.0, rel_tol=1e-5)
''',
    },
]


# ==============================================================================
# --- 11. Terminal-Bench v4.0 (Interactive CLI Agent Scenarios) (6) ---
# ==============================================================================

TERMINAL_SCENARIOS = [
    {
        "id": "term_01",
        "category": "terminal",
        "tools": ["bash_exec", "read_file", "write_file"],
        "messages": [
            {"role": "user", "content": "Analyze `/var/log/nginx/access.log`, find all unique client IP addresses associated with HTTP 500 error responses, and write the newline-separated unique list of IPs to `/tmp/failed_ips.txt`."}
        ],
        "expected_vfs_keys": ["/tmp/failed_ips.txt"],
        "expected_vfs_contains": {"/tmp/failed_ips.txt": ["10.0.0.52", "172.16.4.88", "10.0.0.99"]},
        "expected_answer": ["10.0.0.52", "172.16.4.88", "10.0.0.99", "failed_ips"],
    },
    {
        "id": "term_02",
        "category": "terminal",
        "tools": ["read_file", "write_file", "bash_exec"],
        "messages": [
            {"role": "user", "content": "The nginx config `/etc/nginx/conf.d/api.conf` has syntax errors. Inspect it with `read_file`, fix the missing semicolon after `proxy_set_header Host $host;`, update backend port from 8080 to 8000; and verify with `nginx -t`."}
        ],
        "expected_vfs_contains": {"/etc/nginx/conf.d/api.conf": ["8000;", "proxy_set_header Host $host;"]},
        "expected_answer": ["syntax is ok", "successful", "8000"],
    },
    {
        "id": "term_03",
        "category": "terminal",
        "tools": ["read_file", "write_file"],
        "messages": [
            {"role": "user", "content": "Resolve the git merge conflict markers in `/workspace/config.py` by choosing the production database URL from HEAD. Remove all conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`)."}
        ],
        "expected_vfs_contains": {
            "/workspace/config.py": ["DATABASE_URL = \"postgresql://app_prod:Secr3tP@ss@db-primary.internal.net:5432/analytics_prod\""]
        },
        "expected_vfs_absent": {"/workspace/config.py": ["<<<<<<<", "=======", ">>>>>>>", "test_db"]},
        "expected_answer": ["resolved", "conflict", "analytics_prod"],
    },
    {
        "id": "term_04",
        "category": "terminal",
        "tools": ["bash_exec"],
        "messages": [
            {"role": "user", "content": "Check the process list `/proc/simulated_ps` and kill the runaway process PID that is consuming over 90% memory."}
        ],
        "expected_answer": ["891", "killed", "terminated"],
    },
    {
        "id": "term_05",
        "category": "terminal",
        "tools": ["bash_exec", "read_file", "write_file"],
        "messages": [
            {"role": "user", "content": "Update all `.env` files in `/app/services/` by replacing the legacy placeholder `${OLD_API_KEY}` with `${NEW_API_KEY}`."}
        ],
        "expected_vfs_contains": {
            "/app/services/auth.env": ["API_KEY=${NEW_API_KEY}"],
            "/app/services/billing.env": ["API_KEY=${NEW_API_KEY}"],
        },
        "expected_answer": ["NEW_API_KEY", "updated", "replaced"],
    },
    {
        "id": "term_06",
        "category": "terminal",
        "tools": ["read_file", "write_file"],
        "messages": [
            {"role": "user", "content": "Fix the JSON formatting error (trailing commas) in `/db/migrations/004_users.json` so that it parses as strictly valid JSON."}
        ],
        "expected_json_valid": ["/db/migrations/004_users.json"],
        "expected_answer": ["valid", "json", "trailing comma"],
    },
]


# ==============================================================================
# --- 12. AA-LCR (Long-Context Retrieval & Multi-Hop Reasoning) (4) ---
# ==============================================================================

# Helper to generate synthetic long context documents with embedded needles
def _generate_lcr_corpus(topic: str, needle_key: str, needle_value: str, distractor_count: int = 15) -> str:
    paragraphs = []
    for i in range(distractor_count):
        paragraphs.append(
            f"Document #{i+1} [Catalog & Archival Record {1000 + i * 37}]:\n"
            f"The ongoing operations across cluster node {chr(65 + (i % 26))}-{i*11} proceeded with standard telemetry checkpoints. "
            f"Thermal dissipation metrics maintained nominal 42.5 deg C tolerances across memory banks. "
            f"Periodic background verification completed with 0 parity errors across partition block {i*256}. "
            f"Network routing latency between region-east and region-west averaged 18.2 ms under baseline traffic load.\n"
        )
    # Insert target needle in the middle
    mid = len(paragraphs) // 2
    paragraphs.insert(mid, f"\n*** CONFIDENTIAL AUDIT DIRECTIVE [{topic.upper()}] ***\nTarget Identifier: {needle_key}\nResolution Value: {needle_value}\nAuthorization Code: AUTH-77981-VERIFIED\n*** END DIRECTIVE ***\n\n")
    return "\n".join(paragraphs)


LCR_SCENARIOS = [
    {
        "id": "lcr_01",
        "topic": "Financial Merger Timeline",
        "needle_key": "Project Falcon Final Settlement Amount",
        "needle_value": "$487.5 Million USD",
        "prompt": f"""You are analyzing a long cross-document enterprise archive. Read the full documentation carefully:

{_generate_lcr_corpus('Financial Merger', 'Project Falcon Final Settlement Amount', '$487.5 Million USD', distractor_count=18)}

Question: What is the exact Project Falcon Final Settlement Amount stated in the confidential audit directive? State the exact figure as #### <answer>.""",
        "expected_answer": "$487.5 Million USD",
    },
    {
        "id": "lcr_02",
        "topic": "Distributed Incident Root Cause",
        "needle_key": "Root Cause Trigger Service",
        "needle_value": "auth-token-revocation-worker-node-4",
        "prompt": f"""You are triaging a long distributed microservice incident report:

{_generate_lcr_corpus('Incident Root Cause', 'Root Cause Trigger Service', 'auth-token-revocation-worker-node-4', distractor_count=18)}

Question: What was the exact Root Cause Trigger Service identified in the confidential audit directive? State the service name as #### <answer>.""",
        "expected_answer": "auth-token-revocation-worker-node-4",
    },
    {
        "id": "lcr_03",
        "topic": "Master Procurement Agreement Addenda",
        "needle_key": "Net Warranty Liability Cap for Phase 4 Deliverables",
        "needle_value": "$2,450,000.00",
        "prompt": f"""Review the master contract schedules and cross-referenced addenda:

{_generate_lcr_corpus('Procurement Addenda', 'Net Warranty Liability Cap for Phase 4 Deliverables', '$2,450,000.00', distractor_count=18)}

Question: What is the Net Warranty Liability Cap for Phase 4 Deliverables? State the exact figure as #### <answer>.""",
        "expected_answer": "$2,450,000.00",
    },
    {
        "id": "lcr_04",
        "topic": "Clinical Study Biomarker Analysis",
        "needle_key": "Statistically Significant Biomarker in Cohort 9",
        "needle_value": "Serum Phospho-Tau-217",
        "prompt": f"""Review the multi-cohort clinical trial laboratory results:

{_generate_lcr_corpus('Biomarker Study', 'Statistically Significant Biomarker in Cohort 9', 'Serum Phospho-Tau-217', distractor_count=18)}

Question: What is the Statistically Significant Biomarker in Cohort 9? State the biomarker name as #### <answer>.""",
        "expected_answer": "Serum Phospho-Tau-217",
    },
]


# ==============================================================================
# --- 13. Throughput Evaluation Prompts (Code Generation) ---
# ==============================================================================

THROUGHPUT_PROMPTS = [
    "Write a Python function that checks if a string is a palindrome, with a brief explanation.",
    "Write a Python function that computes the nth Fibonacci number with dynamic programming, and briefly explain it.",
    "Write a Python function that merges two sorted lists into one sorted list, with a brief explanation.",
    "Write a Python function that performs binary search on a sorted list, with a brief explanation.",
]
