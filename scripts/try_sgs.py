"""Manual check against the real SGS API.

Usage: uv run python scripts/try_sgs.py [series_code] [days]
Example: uv run python scripts/try_sgs.py 1 10   (dollar, last 10 days)
"""

import asyncio
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.tools.bcb import SGSError, fetch_sgs_series


async def main() -> None:
    code = int(sys.argv[1]) if len(sys.argv) > 1 else 433  # default: IPCA
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 180

    end = date.today()
    start = end - timedelta(days=days)
    print(f"Series {code}, {start} to {end}")

    try:
        observations = await fetch_sgs_series(code, start, end)
    except SGSError as e:
        print(f"Error: {e}")
        return

    if not observations:
        print("No data in this period.")
    for obs in observations:
        print(f"{obs.date:%d/%m/%Y}  {obs.value}")


asyncio.run(main())
