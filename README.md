# ReadWater.ai

AI-powered inshore saltwater fishing trip planner. Uses multi-scale satellite imagery analysis to build structured "area knowledge" of coastal fishing zones, then combines it with real-time conditions to generate fishing forecasts and trip plans.

## Architecture

**Area Knowledge Builder** — Recursive pipeline that analyzes satellite imagery at progressively finer zoom levels (30 mi → 0.37 mi) to identify and catalog fishable structure: grass flats, oyster bars, channels, mangrove shorelines, depth transitions, current seams, etc.

**Forecast Engine** — Takes pre-built area knowledge + live tide/weather data to generate 72-hour fishing forecasts.

**Trip Plan Generator** — Produces specific trip plans with spots, timing, and reasoning based on forecasts and user parameters (kayak/boat, target species, time window, launch point).

## Setup

```bash
# Clone and install
git clone https://github.com/Squale88/ReadWater.ai.git
cd ReadWater.ai
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"

# Configure API keys
cp .env.example .env
# Edit .env with your Anthropic and Google Maps API keys

# Run tests
pytest
```

## Project Structure

```
src/readwater/
├── models/          # Data models (Cell, AreaKnowledge)
├── pipeline/        # Recursive cell analyzer
└── api/             # External integrations (mapping, Claude vision, weather, tides)
tests/
```

## Tech Stack

- Python 3.11+
- Anthropic Claude API (vision) for satellite image analysis
- Google Maps Static API for satellite imagery
- Pydantic for data models
