# Week 3: Multi-Source Data Pipeline
**Operational question:** Do external conditions (weather) and staffing/holiday patterns help explain
variation in plant throughput (`Flow_Rate_LPM`) recorded by our internal sensors?

This notebook integrates three independent data sources into a single Master DataFrame:

1. **Source 1 (Internal):** Cleaned operational sensor log from Week 2 (`mock_ops.csv`) — pressure,
   temperature, and flow-rate readings taken every 10 minutes across 5 plant zones.
2. **Source 2 (External API):** Daily weather (max temperature, precipitation) fetched live via
   `requests` from the free Open-Meteo API, with a graceful fallback if the API is unreachable.
3. **Source 3 (Database):** A small SQLite database (via SQLAlchemy) containing `Holiday_Calendar`
   and `Employee_Shifts` tables, queried with a SQL `JOIN` + `GROUP BY` to get daily staffing totals.

We align everything on `date`, handle missing values from the merges, and finish with a correlation
analysis addressing the operational question above.


Setup
# Standard imports used throughout the pipeline
import pandas as pd
import numpy as np
import requests                      # Source 2: HTTP calls to the external weather API
from sqlalchemy import create_engine, text   # Source 3: SQLite access
import matplotlib.pyplot as plt

pd.set_option("display.width", 120)

## 1. Source 1 (Internal) — Cleaned Operational Dataset

Loads the Week 2 cleaned sensor log. Each row is one sensor reading (10-minute cadence) for one of
5 zones. Columns:

- `timestamp` — when the reading was taken (already parsed/cleaned in Week 2)
- `Zone` — plant zone name, standardized in Week 2 (e.g. `Zone_North`)
- `Shift` — work shift active at that time (`Morning` / `Afternoon` / `Night`)
- `Pressure_PSI`, `Temperature_C`, `Flow_Rate_LPM` — sensor readings, sentinel fault values and
  outliers already removed/imputed in Week 2's cleaning pass

We aggregate to **daily** granularity here because Sources 2 and 3 are naturally daily-resolution
(one weather reading per day, one staffing total per day), so daily is the common grain the three
sources can be merged on.

# --- Source 1: Internal operational data ---
ops = pd.read_csv("mock_ops.csv", parse_dates=["timestamp"])

# Derive a plain `date` column (no time-of-day) since that's the grain we'll merge on
ops["date"] = ops["timestamp"].dt.date

# Aggregate sensor readings to one row per day:
#   - avg_pressure_psi / avg_internal_temp_c / avg_flow_rate_lpm: daily mean of each sensor
#   - reading_count: how many 10-min readings fed into that day's average (useful sanity check)
daily_ops = (
    ops.groupby("date")
       .agg(
           avg_pressure_psi=("Pressure_PSI", "mean"),
           avg_internal_temp_c=("Temperature_C", "mean"),
           avg_flow_rate_lpm=("Flow_Rate_LPM", "mean"),
           reading_count=("Flow_Rate_LPM", "count"),
       )
       .reset_index()
)

print(f"Daily internal records: {len(daily_ops)}")
daily_ops

## 2. Source 2 (External API) — Daily Weather

We fetch daily max temperature and precipitation for the plant's location from the free,
no-API-key **Open-Meteo** API, for the same date range covered by the internal sensor data.

The `try/except` block below handles the realistic failure modes of any external API call:
network errors, timeouts, and non-200 responses. If the live call fails for any reason
(no internet in this environment, rate limiting, API downtime, etc.), we fall back to a
deterministic synthetic weather series with the **same schema** so the rest of the pipeline
still runs end-to-end. This mirrors what you'd want in a real production pipeline: a data
source outage should degrade gracefully, not crash the whole job.

# --- Source 2: External weather API ---

PLANT_LATITUDE = 40.71     # example plant location (New York) -- replace with your real site
PLANT_LONGITUDE = -74.00
START_DATE = str(daily_ops["date"].min())
END_DATE = str(daily_ops["date"].max())

def fetch_weather(lat, lon, start_date, end_date):
    """Fetch daily max temperature (C) and precipitation (mm) from Open-Meteo.
    Raises on any HTTP/network problem so the caller's except block can catch it."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,precipitation_sum",
        "timezone": "auto",
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()             # raises HTTPError on 4xx/5xx
    payload = resp.json()
    return pd.DataFrame({
        "date": pd.to_datetime(payload["daily"]["time"]).date,
        "ext_temp_max_c": payload["daily"]["temperature_2m_max"],
        "precip_mm": payload["daily"]["precipitation_sum"],
    })

try:
    weather = fetch_weather(PLANT_LATITUDE, PLANT_LONGITUDE, START_DATE, END_DATE)
    print("Live weather API call succeeded.")
except (requests.exceptions.RequestException, KeyError, ValueError) as err:
    # RequestException covers connection errors/timeouts/HTTP errors;
    # KeyError/ValueError guard against an unexpected JSON shape.
    print(f"Weather API call failed ({type(err).__name__}: {err}). Falling back to synthetic data.")
    rng = np.random.default_rng(7)          # seeded -> reproducible fallback for grading/demo
    dates = daily_ops["date"]
    weather = pd.DataFrame({
        "date": dates,
        "ext_temp_max_c": rng.uniform(24, 34, len(dates)).round(1),
        "precip_mm": rng.choice([0, 0, 0, 2.5, 8.0, 15.0], len(dates)),
    })

weather

## 3. Source 3 (Database) — SQLite via SQLAlchemy

We build a small SQLite database with two supplementary tables:

- **`Holiday_Calendar`**: one row per calendar date, flagging whether it's a company holiday
- **`Employee_Shifts`**: one row per (date, zone, shift) with the number of staff scheduled

A single SQL query with a **`JOIN`** (linking shifts to the holiday calendar by date) and a
**`GROUP BY`** (rolling every zone/shift combination up to one row per day) produces a daily
staffing summary, loaded straight into a DataFrame with `pd.read_sql`.

# --- Source 3: Supplementary SQLite database ---
engine = create_engine("sqlite:///supplementary.db")

# Holiday_Calendar: one row per date in our window, flagging any company holiday.
# (In this example, the final day of the observed week is treated as a company holiday
#  to demonstrate the effect on staffing / throughput.)
holiday_calendar = pd.DataFrame({
    "cal_date": daily_ops["date"],
    "is_holiday": [0] * (len(daily_ops) - 1) + [1],
    "holiday_name": [None] * (len(daily_ops) - 1) + ["Company Founding Day"],
})

# Employee_Shifts: staffing headcount per zone per shift per day.
# Synthetic here (real deployment would pull this from an HR system), but deliberately
# staffed *lower* on the holiday to make the JOIN/GROUP BY result meaningful downstream.
rng = np.random.default_rng(42)
zones = ops["Zone"].unique()
shifts_list = ["Morning", "Afternoon", "Night"]
rows = []
last_date = daily_ops["date"].max()
for d in daily_ops["date"]:
    base = 4 if d == last_date else 8   # short-staffed on the holiday
    for z in zones:
        for s in shifts_list:
            rows.append([d, z, s, int(rng.integers(base - 2, base + 3))])
employee_shifts = pd.DataFrame(rows, columns=["shift_date", "zone", "shift", "staff_count"])

# Write both tables into the SQLite database
holiday_calendar.to_sql("Holiday_Calendar", engine, if_exists="replace", index=False)
employee_shifts.to_sql("Employee_Shifts", engine, if_exists="replace", index=False)

# JOIN Employee_Shifts to Holiday_Calendar on date, GROUP BY date to get one summary row/day:
#   - total_staff:  SUM of staff across every zone & shift that day
#   - is_holiday / holiday_name: pulled from the joined calendar table (MAX just selects
#     the single value present per day, since it's constant within a date after the JOIN)
staffing_query = text("""
    SELECT
        e.shift_date            AS date,
        SUM(e.staff_count)      AS total_staff,
        MAX(h.is_holiday)       AS is_holiday,
        MAX(h.holiday_name)     AS holiday_name
    FROM Employee_Shifts e
    JOIN Holiday_Calendar h ON e.shift_date = h.cal_date
    GROUP BY e.shift_date
    ORDER BY e.shift_date
""")

with engine.connect() as conn:
    staffing_daily = pd.read_sql(staffing_query, conn)

staffing_daily

## 4. Integration — Master DataFrame

All three sources are now daily-grain with a `date` column, so we `merge` them left-to-right
starting from the internal ops data (our "spine"), using `how="left"` so we never drop a day of
real operational data even if a supplementary source is missing that date.

**Handling missing values from the merge:**
- `holiday_name` — a missing value here genuinely means "not a holiday", so we fill with the
  explicit label `"None"` rather than leaving `NaN` (avoids confusing "missing data" with
  "not applicable").
- `is_holiday` — missing means false, so we fill with `0`.
- `ext_temp_max_c` / `precip_mm` / `total_staff` — if a date were missing from a source (not the
  case in this dataset, but handled defensively), we forward-fill from the nearest prior day
  since weather/staffing patterns are locally stable day-to-day.

# --- Integration: merge all three sources on `date` ---
master = (
    daily_ops
    .merge(weather, on="date", how="left")
    .merge(staffing_daily, on="date", how="left")
)

# Handle missing values introduced by the merges (see markdown above for rationale)
master["holiday_name"] = master["holiday_name"].fillna("None")
master["is_holiday"] = master["is_holiday"].fillna(0).astype(int)
master[["ext_temp_max_c", "precip_mm", "total_staff"]] = (
    master[["ext_temp_max_c", "precip_mm", "total_staff"]].ffill()
)

print("Missing values remaining after merge + fill:")
print(master.isna().sum())
master

## 5. Analysis — Correlation

**Operational question:** *Does staffing level (and, secondarily, weather) correlate with plant
throughput (`avg_flow_rate_lpm`)?*

We compute the pairwise correlation matrix between throughput, staffing, and the two weather
variables, then visualize the staffing-vs-throughput relationship specifically since that's the
most actionable lever for a Director (staffing is controllable; weather isn't).

# --- Analysis: correlation matrix ---
corr_cols = ["avg_flow_rate_lpm", "total_staff", "precip_mm", "ext_temp_max_c"]
corr_matrix = master[corr_cols].corr()
print(corr_matrix.round(2))

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(master["total_staff"], master["avg_flow_rate_lpm"])
for _, row in master.iterrows():
    label = "Holiday" if row["is_holiday"] else ""
    ax.annotate(label, (row["total_staff"], row["avg_flow_rate_lpm"]),
                textcoords="offset points", xytext=(5, 5), fontsize=8, color="crimson")
ax.set_xlabel("Total staff scheduled that day")
ax.set_ylabel("Average flow rate (LPM)")
ax.set_title("Daily staffing vs. throughput")
plt.tight_layout()
plt.show()

r = corr_matrix.loc["avg_flow_rate_lpm", "total_staff"]
print(f"\nCorrelation(total_staff, avg_flow_rate_lpm) = {r:.2f}")

