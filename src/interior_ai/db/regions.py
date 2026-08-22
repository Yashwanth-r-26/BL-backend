"""Regional context for pricing and dimension priors.

India only, deliberately. The dimension priors, material SKUs and typical rates
in this system were researched against the Indian market; extending the picker
to other countries would mean asking a model to price a market we have nothing
to check its answer against. A wrong price that looks confident is worse than
an honest "not supported yet".

Two things come out of a location:

* **Currency and market context** for the quotation prompt. A quote for
  Bengaluru and one for a tier-3 town are different numbers for identical work,
  and the model can only reflect that if it is told where the room is.
* **Which dimension prior applies** -- metro flats run smaller than
  independent houses in smaller cities. That was previously a dropdown the
  user had to understand; a city name answers it better than they can.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Country:
    code: str
    name: str
    currency: str
    symbol: str
    supported: bool
    note: str = ""


COUNTRIES: tuple[Country, ...] = (
    Country("IN", "India", "INR", "\u20b9", True),
    # Listed so the picker can say why, rather than silently omitting them.
    Country("US", "United States", "USD", "$", False,
            "Not supported yet -- no verified rate data for this market."),
    Country("AE", "United Arab Emirates", "AED", "AED", False,
            "Not supported yet -- no verified rate data for this market."),
)

DEFAULT_COUNTRY = "IN"


#: Cities whose construction and furnishing costs behave like a metro. The
#: distinction drives both the price context and the room-size prior, so it is
#: about cost of building rather than population alone -- Gurugram is small and
#: expensive, Patna is large and not.
METRO_CITIES = frozenset({
    "mumbai", "navi mumbai", "thane", "delhi", "new delhi", "noida",
    "gurugram", "gurgaon", "faridabad", "ghaziabad", "bengaluru", "bangalore",
    "hyderabad", "secunderabad", "chennai", "kolkata", "pune", "ahmedabad",
    "gandhinagar", "goa", "panaji",
})

#: Cities that sit meaningfully above the national average without being
#: metros -- state capitals and large industrial centres.
TIER_TWO_CITIES = frozenset({
    "jaipur", "lucknow", "chandigarh", "mohali", "panchkula", "kochi",
    "cochin", "ernakulam", "coimbatore", "mysuru", "mysore", "indore",
    "bhopal", "nagpur", "surat", "vadodara", "visakhapatnam", "vijayawada",
    "bhubaneswar", "guwahati", "dehradun", "raipur", "ranchi", "patna",
    "kanpur", "agra", "varanasi", "amritsar", "ludhiana", "jodhpur",
    "udaipur", "madurai", "tiruchirappalli", "salem", "hubballi", "mangaluru",
    "mangalore", "thiruvananthapuram", "trivandrum", "kozhikode", "calicut",
    "nashik", "aurangabad", "rajkot", "jamshedpur", "siliguri", "shimla",
})


#: Coordinates for the cities above, so a device's GPS fix can be resolved to
#: a city without calling a geocoding service. That matters for more than
#: convenience: a reverse-geocode API is a key to manage, a rate limit to hit,
#: a privacy question to answer, and a dependency that fails offline. Nearest
#: known city is coarser than a real geocoder, but the only use for it here is
#: choosing a pricing tier, and tiers do not change between one suburb and the
#: next.
CITY_COORDS: dict[str, tuple[float, float]] = {
    "mumbai": (19.076, 72.877), "navi mumbai": (19.033, 73.030),
    "thane": (19.218, 72.978), "delhi": (28.613, 77.209),
    "noida": (28.535, 77.391), "gurugram": (28.459, 77.027),
    "faridabad": (28.408, 77.317), "ghaziabad": (28.669, 77.454),
    "bengaluru": (12.972, 77.595), "hyderabad": (17.385, 78.487),
    "chennai": (13.083, 80.271), "kolkata": (22.573, 88.364),
    "pune": (18.520, 73.857), "ahmedabad": (23.023, 72.571),
    "gandhinagar": (23.216, 72.684), "panaji": (15.491, 73.828),
    "jaipur": (26.912, 75.787), "lucknow": (26.847, 80.947),
    "chandigarh": (30.733, 76.779), "kochi": (9.932, 76.267),
    "coimbatore": (11.017, 76.956), "mysuru": (12.295, 76.639),
    "indore": (22.720, 75.858), "bhopal": (23.260, 77.413),
    "nagpur": (21.146, 79.088), "surat": (21.170, 72.831),
    "vadodara": (22.307, 73.181), "visakhapatnam": (17.687, 83.219),
    "vijayawada": (16.507, 80.648), "bhubaneswar": (20.296, 85.825),
    "guwahati": (26.145, 91.736), "dehradun": (30.317, 78.032),
    "raipur": (21.251, 81.630), "ranchi": (23.344, 85.310),
    "patna": (25.594, 85.138), "kanpur": (26.450, 80.332),
    "agra": (27.177, 78.008), "varanasi": (25.318, 82.973),
    "amritsar": (31.634, 74.872), "ludhiana": (30.901, 75.857),
    "jodhpur": (26.238, 73.024), "udaipur": (24.586, 73.713),
    "madurai": (9.925, 78.120), "tiruchirappalli": (10.790, 78.704),
    "salem": (11.664, 78.146), "hubballi": (15.364, 75.124),
    "mangaluru": (12.914, 74.856), "thiruvananthapuram": (8.524, 76.936),
    "kozhikode": (11.259, 75.780), "nashik": (19.997, 73.790),
    "aurangabad": (19.876, 75.343), "rajkot": (22.303, 70.802),
    "jamshedpur": (22.804, 86.203), "siliguri": (26.727, 88.395),
    "shimla": (31.105, 77.173), "hosur": (12.740, 77.826),
    "tirupati": (13.629, 79.419), "warangal": (17.978, 79.594),
    "belagavi": (15.850, 74.498), "davangere": (14.464, 75.921),
    "kolhapur": (16.705, 74.243), "jalandhar": (31.326, 75.576),
    "gwalior": (26.218, 78.183), "jabalpur": (23.181, 79.986),
    "cuttack": (20.463, 85.883), "asansol": (23.685, 86.974),
    "ajmer": (26.449, 74.639), "bareilly": (28.367, 79.430),
    "meerut": (28.984, 77.706), "aligarh": (27.897, 78.088),
    "solapur": (17.659, 75.906), "puducherry": (11.914, 79.812),
}

#: Beyond this a fix is probably outside the cities we know. The result is
#: still returned -- a tier beats nothing -- but flagged, because silently
#: pricing a village at a metro's rates would be wrong in a way nobody sees.
NEAREST_CITY_WARN_KM = 120.0


def _haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    from math import asin, cos, radians, sin, sqrt

    lat1, lon1, lat2, lon2 = map(radians, (a[0], a[1], b[0], b[1]))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6371.0 * asin(sqrt(h))


def nearest_city(latitude: float, longitude: float) -> dict:
    """Resolve a GPS fix to the nearest city we hold a pricing tier for.

    No network call, no API key, and no third-party service receiving a
    user's coordinates.
    """
    point = (float(latitude), float(longitude))
    city, distance = min(
        ((name, _haversine_km(point, coords))
         for name, coords in CITY_COORDS.items()),
        key=lambda pair: pair[1],
    )
    return {
        "city": city.title(),
        "distance_km": round(distance, 1),
        "confident": distance <= NEAREST_CITY_WARN_KM,
    }


def normalise_city(city: str) -> str:
    return " ".join(city.strip().lower().split())


def city_tier(city: str) -> str:
    """'metro', 'tier2' or 'tier3' -- the pricing context for a city.

    Unknown cities fall to tier3 rather than a metro default: overstating a
    small-town budget is the more damaging error, because it makes the whole
    quote read as wrong to the person who lives there.
    """
    name = normalise_city(city)
    if name in METRO_CITIES:
        return "metro"
    if name in TIER_TWO_CITIES:
        return "tier2"
    return "tier3"


def prior_region(city: str) -> str:
    """Which :mod:`interior_ai.perception.priors` region a city implies.

    Replaces a dropdown the user had to interpret. A city name is something
    they know for certain; "IN_METRO vs IN_NONMETRO" is a question about our
    data model, not about their home.
    """
    return "IN_METRO" if city_tier(city) == "metro" else "IN_NONMETRO"


def country(code: str) -> Country | None:
    code = (code or "").strip().upper()
    return next((c for c in COUNTRIES if c.code == code), None)


def describe(country_code: str, city: str) -> dict:
    """Everything downstream needs from a location, in one dict."""
    resolved = country(country_code) or country(DEFAULT_COUNTRY)
    tier = city_tier(city)
    return {
        "country": resolved.code,
        "country_name": resolved.name,
        "city": city.strip(),
        "currency": resolved.currency,
        "currency_symbol": resolved.symbol,
        "city_tier": tier,
        "prior_region": prior_region(city),
        "supported": resolved.supported,
        "note": resolved.note,
    }