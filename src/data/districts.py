"""
Study districts: India's major flood-prone regions, grouped by river basin.

WHY THESE DISTRICTS
-------------------
Not a random sample of India, and the report must not claim it is. These are
districts with a documented history of serious flooding, chosen to span the
country's distinct flood-generating systems:

  Brahmaputra   annual monsoon inundation, braided channel, Assam valley
  Ganga         Kosi/Gandak/Bagmati fan floods in north Bihar
  Ghaghara      Rapti and Ghaghara flooding in eastern UP
  Bengal        delta and Teesta/Damodar flooding in West Bengal
  Mahanadi      Odisha deltaic flooding and cyclone-driven surge
  Godavari      Godavari-Krishna deltas in AP/Telangana
  Konkan        west-coast orographic extremes, Mumbai urban flooding
  Kerala        Western Ghats orographic rainfall, the 2018 anchor event
  Gujarat       Tapi/Narmada/Sabarmati basin flooding
  TamilNadu     northeast-monsoon flooding, Chennai 2015
  Himalayan     steep-catchment flash floods, Uttarakhand/Himachal
  Northwest     Sutlej/Ghaggar and Jhelum flooding

That diversity is the point. It gives the DFO archive many more events to
label (a nationwide set is far less likely to leave a training split empty),
and it makes the basin-transferability question askable: train on one basin,
test on another, and report where the model generalises and where it breaks.
Whether flood models transfer across basins is a genuinely open question, and
a transferability matrix is a stronger result than one accuracy number.

A NOTE ON COASTAL DISTRICTS
---------------------------
Chennai and Cuddalore were originally placed at their headquarters, which
sit on the coast. GloFAS has no river grid cell there, so discharge came
back 100% missing and the dataset builder dropped both districts entirely
(17,532 rows). Their points are nudged a few km inland onto the nearest
river reach. Say so in the report: it is a deliberate, documented
adjustment, not a silent fudge.

COORDINATES ARE POINT SAMPLES
-----------------------------
Each entry is a district headquarters, not an area centroid, and weather is
sampled at that single point. Districts are not meteorologically homogeneous
— Chamoli spans thousands of metres of relief — so this smooths real spatial
variation. Say so in the report. The upgrade is zonal statistics over
district boundary polygons (GADM or Survey of India) with geopandas.
"""

from src.config import CFG

# district -> (lat, lon, state, basin)
INDIA_FLOOD_DISTRICTS: dict[str, tuple[float, float, str, str]] = {
    # ---- Brahmaputra valley, Assam -------------------------------------
    "Dhemaji":        (27.4833, 94.5833, "Assam", "Brahmaputra"),
    "Lakhimpur":      (27.2300, 94.1100, "Assam", "Brahmaputra"),
    "Majuli":         (26.9500, 94.1700, "Assam", "Brahmaputra"),
    "Jorhat":         (26.7509, 94.2037, "Assam", "Brahmaputra"),
    "Sivasagar":      (26.9826, 94.6425, "Assam", "Brahmaputra"),
    "Dibrugarh":      (27.4728, 94.9120, "Assam", "Brahmaputra"),
    "Tinsukia":       (27.4922, 95.3468, "Assam", "Brahmaputra"),
    "Barpeta":        (26.3220, 91.0050, "Assam", "Brahmaputra"),
    "Dhubri":         (26.0200, 89.9800, "Assam", "Brahmaputra"),
    "Goalpara":       (26.1700, 90.6200, "Assam", "Brahmaputra"),
    "Nagaon":         (26.3464, 92.6840, "Assam", "Brahmaputra"),
    "Morigaon":       (26.2500, 92.3400, "Assam", "Brahmaputra"),
    "Kamrup":         (26.1445, 91.7362, "Assam", "Brahmaputra"),
    "Cachar":         (24.8333, 92.7789, "Assam", "Brahmaputra"),

    # ---- Ganga plain, north Bihar (Kosi / Gandak / Bagmati) ------------
    "Supaul":         (26.1260, 86.6050, "Bihar", "Ganga"),
    "Saharsa":        (25.8800, 86.6000, "Bihar", "Ganga"),
    "Madhubani":      (26.3500, 86.0700, "Bihar", "Ganga"),
    "Darbhanga":      (26.1700, 85.9000, "Bihar", "Ganga"),
    "Sitamarhi":      (26.5950, 85.4900, "Bihar", "Ganga"),
    "Muzaffarpur":    (26.1225, 85.3906, "Bihar", "Ganga"),
    "Samastipur":     (25.8600, 85.7800, "Bihar", "Ganga"),
    "Khagaria":       (25.5020, 86.4670, "Bihar", "Ganga"),
    "Katihar":        (25.5400, 87.5700, "Bihar", "Ganga"),
    "Bhagalpur":      (25.2425, 86.9842, "Bihar", "Ganga"),
    "Patna":          (25.5941, 85.1376, "Bihar", "Ganga"),
    "Gopalganj":      (26.4700, 84.4400, "Bihar", "Ganga"),
    "WestChamparan":  (26.8000, 84.5000, "Bihar", "Ganga"),

    # ---- Ghaghara / Rapti, eastern Uttar Pradesh -----------------------
    "Gorakhpur":      (26.7606, 83.3732, "UttarPradesh", "Ghaghara"),
    "Balrampur":      (27.4300, 82.1900, "UttarPradesh", "Ghaghara"),
    "Bahraich":       (27.5743, 81.5944, "UttarPradesh", "Ghaghara"),
    "Siddharthnagar": (27.2800, 83.0900, "UttarPradesh", "Ghaghara"),
    "Ballia":         (25.7600, 84.1500, "UttarPradesh", "Ghaghara"),
    "Varanasi":       (25.3176, 82.9739, "UttarPradesh", "Ghaghara"),

    # ---- West Bengal: delta, Damodar, Teesta ---------------------------
    "Murshidabad":    (24.1800, 88.2700, "WestBengal", "Bengal"),
    "Malda":          (25.0100, 88.1400, "WestBengal", "Bengal"),
    "Howrah":         (22.5958, 88.2636, "WestBengal", "Bengal"),
    "Hooghly":        (22.9000, 88.3900, "WestBengal", "Bengal"),
    "Jalpaiguri":     (26.5200, 88.7300, "WestBengal", "Bengal"),
    "CoochBehar":     (26.3200, 89.4500, "WestBengal", "Bengal"),
    "Nadia":          (23.4000, 88.5000, "WestBengal", "Bengal"),

    # ---- Mahanadi / Brahmani / Baitarani, Odisha -----------------------
    "Puri":           (19.8135, 85.8312, "Odisha", "Mahanadi"),
    "Kendrapara":     (20.5000, 86.4200, "Odisha", "Mahanadi"),
    "Jagatsinghpur":  (20.2600, 86.1700, "Odisha", "Mahanadi"),
    "Bhadrak":        (21.0574, 86.5160, "Odisha", "Mahanadi"),
    "Balasore":       (21.4942, 86.9336, "Odisha", "Mahanadi"),
    "Cuttack":        (20.4625, 85.8830, "Odisha", "Mahanadi"),

    # ---- Godavari / Krishna deltas, AP and Telangana -------------------
    "EastGodavari":   (16.9891, 82.2475, "AndhraPradesh", "Godavari"),
    "WestGodavari":   (16.7107, 81.1000, "AndhraPradesh", "Godavari"),
    "Krishna":        (16.1875, 81.1389, "AndhraPradesh", "Godavari"),
    "Khammam":        (17.2473, 80.1514, "Telangana", "Godavari"),

    # ---- Konkan coast and western Maharashtra --------------------------
    "MumbaiSuburban": (19.1136, 72.8697, "Maharashtra", "Konkan"),
    "Thane":          (19.2183, 72.9781, "Maharashtra", "Konkan"),
    "Raigad":         (18.6400, 72.8700, "Maharashtra", "Konkan"),
    "Ratnagiri":      (16.9902, 73.3120, "Maharashtra", "Konkan"),
    "Kolhapur":       (16.7050, 74.2433, "Maharashtra", "Konkan"),
    "Sangli":         (16.8524, 74.5815, "Maharashtra", "Konkan"),

    # ---- Kerala: the anchor event (2018) -------------------------------
    "Alappuzha":      (9.4981, 76.3388, "Kerala", "Kerala"),
    "Ernakulam":      (9.9816, 76.2999, "Kerala", "Kerala"),
    "Idukki":         (9.8497, 76.9681, "Kerala", "Kerala"),
    "Kottayam":       (9.5916, 76.5222, "Kerala", "Kerala"),
    "Pathanamthitta": (9.2648, 76.7870, "Kerala", "Kerala"),
    "Thrissur":      (10.5276, 76.2144, "Kerala", "Kerala"),
    "Palakkad":      (10.7867, 76.6548, "Kerala", "Kerala"),
    "Wayanad":       (11.6854, 76.1320, "Kerala", "Kerala"),

    # ---- Gujarat: Tapi / Narmada / Sabarmati ---------------------------
    "Surat":          (21.1702, 72.8311, "Gujarat", "Gujarat"),
    "Vadodara":       (22.3072, 73.1812, "Gujarat", "Gujarat"),
    "Ahmedabad":      (23.0225, 72.5714, "Gujarat", "Gujarat"),

    # ---- Tamil Nadu: northeast monsoon ---------------------------------
    "Chennai":        (13.0300, 80.1200, "TamilNadu", "TamilNadu"),
    "Cuddalore":      (11.7600, 79.6500, "TamilNadu", "TamilNadu"),
    "Thanjavur":      (10.7870, 79.1378, "TamilNadu", "TamilNadu"),

    # ---- Himalayan flash-flood catchments ------------------------------
    "Dehradun":       (30.3165, 78.0322, "Uttarakhand", "Himalayan"),
    "Chamoli":        (30.4000, 79.3200, "Uttarakhand", "Himalayan"),
    "Mandi":          (31.7080, 76.9318, "HimachalPradesh", "Himalayan"),

    # ---- Northwest: Sutlej / Ghaggar / Jhelum --------------------------
    "Ludhiana":       (30.9010, 75.8573, "Punjab", "Northwest"),
    "Srinagar":       (34.0837, 74.7973, "JammuKashmir", "Northwest"),
}

# Offset in degrees for sampling elevation around each point when deriving
# slope and ruggedness. 0.05 deg is roughly 5.5 km.
TERRAIN_SAMPLE_OFFSET_DEG = 0.05


def all_names() -> list[str]:
    return sorted(INDIA_FLOOD_DISTRICTS)


def basins() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name, (_, _, _, basin) in INDIA_FLOOD_DISTRICTS.items():
        out.setdefault(basin, []).append(name)
    return {k: sorted(v) for k, v in sorted(out.items())}


def selected() -> list[str]:
    """Resolve config's `project.districts` into a concrete list.

    Accepts:
      "all"                     every district above
      "basin:Brahmaputra"       one basin
      "state:Kerala"            one state
      ["Patna", "Dhubri", ...]  an explicit list
    """
    spec = CFG.project.districts

    if isinstance(spec, str):
        token = spec.strip()
        if token.lower() == "all":
            return all_names()
        if ":" in token:
            kind, value = (p.strip() for p in token.split(":", 1))
            idx = {"basin": 3, "state": 2}.get(kind.lower())
            if idx is None:
                raise ValueError(
                    f"Unknown district selector {kind!r}. Use 'all', "
                    f"'basin:<name>' or 'state:<name>'."
                )
            names = sorted(
                n for n, meta in INDIA_FLOOD_DISTRICTS.items()
                if meta[idx].lower() == value.lower()
            )
            if not names:
                field = "basin" if idx == 3 else "state"
                available = sorted({m[idx] for m in INDIA_FLOOD_DISTRICTS.values()})
                raise KeyError(f"No districts with {field}={value!r}. Available: {available}")
            return names
        raise ValueError(f"Unrecognised project.districts value {spec!r}")

    names = list(spec)
    missing = [n for n in names if n not in INDIA_FLOOD_DISTRICTS]
    if missing:
        raise KeyError(
            f"No coordinates for {missing}. Add them to "
            f"src/data/districts.py, or use 'all'."
        )
    return names


def coords(names: list[str] | None = None) -> dict[str, tuple[float, float]]:
    names = names or selected()
    return {n: (INDIA_FLOOD_DISTRICTS[n][0], INDIA_FLOOD_DISTRICTS[n][1]) for n in names}


def meta(names: list[str] | None = None):
    """DataFrame of district, state, basin — joined in as grouping columns."""
    import pandas as pd
    names = names or selected()
    return pd.DataFrame([
        {"district": n,
         "state": INDIA_FLOOD_DISTRICTS[n][2],
         "basin": INDIA_FLOOD_DISTRICTS[n][3]}
        for n in names
    ])


if __name__ == "__main__":
    print(f"{len(INDIA_FLOOD_DISTRICTS)} districts across {len(basins())} basins\n")
    for basin, names in basins().items():
        print(f"  {basin:12s} {len(names):>2}  {', '.join(names)}")
