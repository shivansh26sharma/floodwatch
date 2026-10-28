"""Build the FloodWatch methodology PDF (docs/FloodWatch_methodology.pdf).

    python scripts/build_methodology_pdf.py

Every number in here is traced to a file in the repo: config.yaml for
settings, the source modules for what is computed, and
outputs/reports/summary_h1.json for results. Nothing is from memory.
"""

import os
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, KeepTogether, PageBreak, PageTemplate, Paragraph,
    Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
# Writes into the repo's docs/ by default; FW_PDF_OUT overrides it, which is
# what lets the document be built somewhere else without editing this file.
OUT = os.environ.get("FW_PDF_OUT", str(ROOT / "docs" / "FloodWatch_methodology.pdf"))

INK = colors.HexColor("#0f172a")
BLUE = colors.HexColor("#1d4ed8")
SLATE = colors.HexColor("#475569")
MUTED = colors.HexColor("#64748b")
LINE = colors.HexColor("#cbd5e1")
BAND = colors.HexColor("#f1f5f9")
WARN = colors.HexColor("#b45309")
WARNBG = colors.HexColor("#fffbeb")

ss = getSampleStyleSheet()


def style(name, **kw):
    base = kw.pop("parent", ss["BodyText"])
    return ParagraphStyle(name, parent=base, **kw)


BODY = style("body", fontName="Helvetica", fontSize=9.5, leading=14,
             textColor=INK, alignment=TA_JUSTIFY, spaceAfter=7)
H1 = style("h1", fontName="Helvetica-Bold", fontSize=16, leading=20,
           textColor=INK, spaceBefore=4, spaceAfter=3)
H2 = style("h2", fontName="Helvetica-Bold", fontSize=12, leading=15,
           textColor=BLUE, spaceBefore=15, spaceAfter=6)
H3 = style("h3", fontName="Helvetica-Bold", fontSize=9.8, leading=13,
           textColor=INK, spaceBefore=10, spaceAfter=3)
SUB = style("sub", fontName="Helvetica", fontSize=9.5, leading=13,
            textColor=MUTED, spaceAfter=2)
CAP = style("cap", fontName="Helvetica-Oblique", fontSize=8.2, leading=11,
            textColor=MUTED, spaceBefore=3, spaceAfter=10)
CELL = style("cell", fontName="Helvetica", fontSize=8.3, leading=11,
             textColor=INK, alignment=0, spaceAfter=0)
CELLB = style("cellb", parent=CELL, fontName="Helvetica-Bold")
CELLH = style("cellh", parent=CELL, fontName="Helvetica-Bold",
              textColor=colors.white)
MONO = style("mono", fontName="Courier", fontSize=8.2, leading=11.5,
             textColor=INK, spaceAfter=8)
NOTE = style("note", parent=BODY, fontSize=9, leading=13, textColor=SLATE)


def P(text, s=BODY):
    return Paragraph(text, s)


def rule(space_before=2, space_after=8):
    t = Table([[""]], colWidths=[170 * mm], rowHeights=[0.6])
    t.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 0.6, LINE)]))
    return [Spacer(1, space_before), t, Spacer(1, space_after)]


def table(rows, widths, header=True, align_right=(), zebra=True,
          font_size=8.3):
    data = []
    for i, row in enumerate(rows):
        out = []
        for j, cell in enumerate(row):
            if i == 0 and header:
                st = CELLH
            elif j in align_right:
                st = ParagraphStyle(f"r{i}{j}", parent=CELL, alignment=2)
            else:
                st = CELL
            out.append(Paragraph(str(cell), st))
        data.append(out)

    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    cmds = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, LINE),
    ]
    if header:
        cmds += [("BACKGROUND", (0, 0), (-1, 0), INK),
                 ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK)]
        if zebra:
            for r in range(2, len(data), 2):
                cmds.append(("BACKGROUND", (0, r), (-1, r), BAND))
    t.setStyle(TableStyle(cmds))
    return t


BANDTXT = style("band", parent=CELL, fontName="Helvetica-Bold",
                fontSize=8.4, textColor=colors.HexColor("#14325c"))
NUM = style("num", parent=CELL, textColor=MUTED, fontSize=7.6)


def feature_table(groups, widths):
    """One row per feature, with a banded row per group.

    The grouped counts in section 2.5 say how the feature set is shaped;
    this says exactly what is in it, which is what anyone checking the
    work actually needs.
    """
    data = [[Paragraph(h, CELLH) for h in ("#", "Feature", "Definition")]]
    cmds = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
    ]
    n = 0
    for title, items in groups:
        r = len(data)
        data.append([Paragraph(title, BANDTXT), "", ""])
        cmds += [("SPAN", (0, r), (-1, r)),
                 ("BACKGROUND", (0, r), (-1, r), colors.HexColor("#dde9f6")),
                 ("TOPPADDING", (0, r), (-1, r), 6),
                 ("BOTTOMPADDING", (0, r), (-1, r), 5)]
        for name, desc in items:
            n += 1
            data.append([
                Paragraph(str(n), NUM),
                Paragraph(f"<font face='Courier' size='7.6'>{name}</font>", CELL),
                Paragraph(desc, CELL)])
            cmds.append(("LINEBELOW", (0, len(data) - 1), (-1, len(data) - 1),
                         0.3, colors.HexColor("#e8eef5")))
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle(cmds))
    return t


def callout(title, body, accent=WARN, bg=WARNBG):
    inner = [P(f'<font color="{accent.hexval()}"><b>{title}</b></font>',
               style("ct", parent=BODY, fontSize=9.3, leading=13,
                     spaceAfter=4)),
             P(body, style("cb", parent=BODY, fontSize=9, leading=12.8,
                           textColor=SLATE, spaceAfter=0))]
    t = Table([[inner]], colWidths=[170 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LINEBEFORE", (0, 0), (0, -1), 2.2, accent),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    return [Spacer(1, 4), t, Spacer(1, 12)]


# ----------------------------------------------------------------------
# Page furniture
# ----------------------------------------------------------------------

def decorate(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(20 * mm, 12 * mm,
                      "FloodWatch — data, preprocessing, training and evaluation")
    canvas.drawRightString(w - 20 * mm, 12 * mm, f"{doc.page}")
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.5)
    canvas.line(20 * mm, 15 * mm, w - 20 * mm, 15 * mm)
    canvas.restoreState()


def build():
    doc = BaseDocTemplate(
        OUT, pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=18 * mm, bottomMargin=20 * mm,
        title="FloodWatch — Data, Preprocessing, Training and Evaluation",
        author="FloodWatch project", subject="Methodology reference",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height,
                  id="main", showBoundary=0)
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame],
                                       onPage=decorate)])
    doc.build(story())


# ----------------------------------------------------------------------
# Content
# ----------------------------------------------------------------------

def story():
    s = []

    # ---------------- header ----------------
    s.append(P("FloodWatch", H1))
    s.append(P("District-level flood onset prediction for India — data sources, "
               "preprocessing, model training and evaluation", SUB))
    s += rule(6, 4)
    s.append(P(
        "This document records what the system actually does, so that every "
        "figure quoted about it can be traced back to a file. Settings come "
        "from <font face='Courier'>config.yaml</font>; procedures from the "
        "modules named in each section; results from "
        "<font face='Courier'>outputs/reports/summary_h1.json</font>. The "
        "configuration described throughout is the one that produced the "
        "committed results: 22 districts across two river basins "
        "(Kerala and the Brahmaputra), 2000&#8211;2023, predicting one day "
        "ahead. The district table in "
        "<font face='Courier'>src/data/districts.py</font> carries 75 "
        "districts across 12 basins and the pipeline runs on any subset; the "
        "evaluated slice is smaller because the hosted deployment has about "
        "1&nbsp;GB of memory and the full frame needs roughly 504&nbsp;MB "
        "before training begins.", BODY))

    # ================= 1. DATA =================
    s.append(P("1 &nbsp; Where the data comes from", H2))
    s.append(P(
        "Four layers, from three providers. None requires payment or "
        "registration, and all permit academic use with attribution.", BODY))

    s.append(table([
        ["Layer", "Product", "Access route", "Resolution / form", "Licence"],
        ["Rainfall<br/>(history)", "ERA5 reanalysis, daily "
         "<font face='Courier'>precipitation_sum</font>",
         "Open-Meteo Archive API<br/>"
         "<font face='Courier' size='7.4'>archive-api.open-meteo.com/v1/archive</font>",
         "~25 km grid, daily,<br/>2000&#8211;2023", "Free, non-commercial, "
         "attribution"],
        ["Rainfall<br/>(live)", "Analysis + numerical weather prediction "
         "forecast", "Open-Meteo Forecast API<br/>"
         "<font face='Courier' size='7.4'>api.open-meteo.com/v1/forecast</font>",
         "92 days back,<br/>16 days forward", "Free, non-commercial, "
         "attribution"],
        ["River<br/>discharge", "GloFAS daily "
         "<font face='Courier'>river_discharge</font> (m<super>3</super>/s)",
         "Open-Meteo Flood API<br/>"
         "<font face='Courier' size='7.4'>flood-api.open-meteo.com/v1/flood</font>",
         "Grid cell containing<br/>each district point", "Free, non-commercial, "
         "attribution"],
        ["Terrain", "Elevation, from which slope and ruggedness are derived "
         "here", "Open-Meteo Elevation API<br/>"
         "<font face='Courier' size='7.4'>api.open-meteo.com/v1/elevation</font>",
         "5-point cross per<br/>district, &#177;5.5 km", "Free, attribution"],
        ["Flood<br/>labels", "Dartmouth Flood Observatory event polygons "
         "(v0.9.0)", "Zenodo GeoPackage<br/>"
         "<font face='Courier' size='7.4'>doi 10.5281/zenodo.19288170</font>",
         "Event footprints with<br/>begin and end dates", "CC0 (public "
         "domain)"],
    ], widths=[20 * mm, 38 * mm, 48 * mm, 32 * mm, 32 * mm]))
    s.append(P("Fetching is implemented in <font face='Courier'>"
               "src/data/download.py</font> (history) and <font face='Courier'>"
               "src/live/fetch.py</font> (live). Requests carry a descriptive "
               "User-Agent, pause 1.2 s between calls, and retry four times "
               "with exponential backoff.", CAP))

    s.append(P("1.1 &nbsp; What these products are, stated precisely", H3))
    s.append(P(
        "ERA5 and GloFAS are <b>reanalysis</b>: physics-based models that "
        "assimilate large volumes of real observation &#8212; surface "
        "stations, satellites, radiosondes &#8212; to produce a gridded, "
        "gap-free record. They are real data in the sense that matters. They "
        "are not synthetic, and they are standard inputs in published "
        "hydrology research. They are also <b>not IMD station gauges</b>. At "
        "roughly 25 km, ERA5 smooths local extremes and will systematically "
        "under-represent the peak intensity of a cloudburst. GloFAS discharge "
        "is modelled routing, not a stage reading at a gauge &#8212; which is "
        "why the column is named <font face='Courier'>river_discharge_m3s</font> "
        "rather than anything resembling water level.", BODY))
    s.append(P(
        "This ceiling is real and it is worth stating before any metric is "
        "quoted: a label derived from a coarse event polygon, sampled at a "
        "single point per district, against rainfall that has had its extremes "
        "smoothed, cannot support a very high precision. A model claiming one "
        "on this data would be leaking, not learning.", BODY))

    s.append(P("1.2 &nbsp; How districts are located", H3))
    s.append(P(
        "Each district is one representative latitude/longitude, listed with "
        "its state and river basin in <font face='Courier'>src/data/"
        "districts.py</font>. Two coastal districts (Chennai, Cuddalore) were "
        "moved inland from their administrative headquarters because the "
        "GloFAS grid cell at the coastal point returned no discharge at all. "
        "That is a sampling decision with a consequence &#8212; a single point "
        "stands in for a whole district &#8212; and it belongs in the "
        "limitations, not hidden in a data file.", BODY))

    # ================= 2. PREPROCESSING =================
    s.append(P("2 &nbsp; Preprocessing", H2))

    s.append(P("2.1 &nbsp; Schema quarantine", H3))
    s.append(P(
        "<font face='Courier'>src/data/loaders.py</font> is the only place "
        "raw files are read. Each of the four tables has a declared canonical "
        "schema, and a file that does not match it is rejected by name rather "
        "than flowing downstream to produce a confusing error three modules "
        "later. This is also what lets the same pipeline run against a "
        "synthetic fixture, the raw downloads, or the committed joined "
        "dataset without any other code knowing which.", BODY))

    s.append(P("2.2 &nbsp; Missing values &#8212; a stated policy, not a blanket fill", H3))
    s.append(P(
        "Nothing is filled with zero or with a global mean. The three layers "
        "are treated differently because they fail differently.", BODY))
    s.append(table([
        ["Layer", "Policy", "Why"],
        ["Rainfall", "Linear interpolation within a district, gaps of "
         "&#8804;&nbsp;2 days only. Longer gaps stay NaN and the row is "
         "dropped.",
         "A missing rainfall day is not a zero-rainfall day. Filling it with "
         "zero fabricates dry spells, which is exactly the signal the model "
         "is trying to read."],
        ["River discharge", "Same, limit 3 days.",
         "Gauge and model dropouts are common, and discharge is strongly "
         "autocorrelated, so a short bridge is defensible where a long one "
         "is not."],
        ["Terrain", "Must be complete. A missing value raises an error.",
         "Terrain is static. If it is missing, something is wrong with the "
         "download, and silently continuing would hide that."],
    ], widths=[26 * mm, 66 * mm, 78 * mm]))
    s.append(P("A <font face='Courier'>was_imputed</font> flag is carried per "
               "row so the share of interpolated values can be reported "
               "honestly. The flags are excluded from the feature set. "
               "Implemented in <font face='Courier'>src/data/build_dataset.py"
               "</font>.", CAP))

    s.append(P("2.3 &nbsp; Labelling &#8212; a spatial join, not text matching", H3))
    s.append(P(
        "The DFO release identifies an event by country, so matching on "
        "district names is impossible: every Indian event would reduce to "
        "&#8220;India&#8221;. Labels are therefore produced geometrically. "
        "Event polygons are read from the GeoPackage and reprojected to "
        "EPSG:4326; each district point is tested for containment inside each "
        "event footprint with <font face='Courier'>geopandas.sjoin(predicate="
        "\"within\")</font>; a hit is expanded into one labelled row per day "
        "between the event's begin and end dates, clipped to the study "
        "window. Districts that match no event contribute only negatives, "
        "which is correct and is noted in the console output.", BODY))
    s.append(P(
        "Begin and end column names have changed across DFO releases "
        "(&#8220;Began&#8221;/&#8220;Ended&#8221;, &#8220;Start Date&#8221;/"
        "&#8220;End Date&#8221;, now &#8220;BeginDate&#8221;/&#8220;EndDate"
        "&#8221;), so the matcher normalises case, spaces and underscores "
        "rather than assuming one spelling.", BODY))

    s.append(P("2.4 &nbsp; The prediction target", H3))
    s.append(P(
        "The target is <b>onset within a window</b>: does a flood "
        "<b>begin</b> in this district during the seven days starting at "
        "t&nbsp;+&nbsp;1? Onset is defined per district as a flood day whose "
        "previous day was not a flood day, so a continuing flood does not "
        "re-trigger.", BODY))

    s += callout(
        "Why not the obvious target",
        "The natural choice &#8212; is a flood in progress at t&nbsp;+&nbsp;"
        "horizon &#8212; broke the evaluation, and finding out why was the "
        "most consequential decision in the project. DFO changed how long it "
        "keeps an event open: events in 2000&#8211;2014 average 20.0 days, "
        "events in 2017&#8211;2023 average 8.6 days, with the same number of "
        "events per year and comparable polygons. Only the duration "
        "bookkeeping changed. Training on &#8220;ongoing&#8221; therefore "
        "learns 20-day floods and is tested on 8.6-day floods, which is why "
        "the positive rate collapsed from 7.16% in train to 1.45% in test. "
        "That is a labelling artefact wearing the costume of a hard problem. "
        "A flood's start date is a fact about the world; its recorded "
        "duration is bookkeeping. Switching to onset removed the drift &#8212; "
        "and it is also the operationally useful question, since warning is "
        "wanted before the water arrives, not confirmation that it is still "
        "there on day twelve.")

    s.append(P("2.5 &nbsp; Feature engineering", H3))
    s.append(P(
        "59 features are built from two measured series, the terrain "
        "attributes and the calendar. The count breaks down as follows; "
        "<b>Appendix A lists all 59 by name</b>.", BODY))
    s.append(table([
        ["Group", "n", "Features"],
        ["Raw", "2", "rainfall_mm, river_discharge_m3s"],
        ["Lags", "10", "Each series at 1, 2, 3, 5 and 7 days"],
        ["Rolling", "20", "Windows of 3, 7, 14 and 30 days: mean and max for "
         "both series, plus sum for rainfall"],
        ["Antecedent wetness", "1", "15-day cumulative rainfall &#8212; how "
         "saturated the ground already is. This is the interaction that "
         "decides whether a given rainfall floods or drains"],
        ["Rate of change", "2", "1-day and 3-day discharge difference. A "
         "rising river is more dangerous than a high static one"],
        ["Climatology departure", "3", "Day-of-year rainfall climatology, "
         "plus absolute and ratio departure from it"],
        ["Seasonality", "3", "doy_sin, doy_cos (cyclical, so December and "
         "January are adjacent), is_monsoon"],
        ["Basin context", "7", "Same-day basin mean and max for both series, "
         "each district's ratio to its basin, and basin mean antecedent "
         "rainfall &#8212; a cheap proxy for upstream state"],
        ["Percentile rank", "3", "Where today sits in this district's own "
         "training-period distribution, for rainfall, discharge and "
         "antecedent rainfall"],
        ["Basin dummies", "2", "One-hot river basin (Kerala, Brahmaputra in "
         "this configuration)"],
        ["Terrain", "3", "mean_elevation_m, mean_slope_deg, "
         "terrain_ruggedness_m"],
        ["Terrain interaction", "3", "rain / (slope+1), antecedent &#215; "
         "ruggedness, 1 / (elevation+10)"],
        ["<b>Total</b>", "<b>59</b>", ""],
    ], widths=[34 * mm, 10 * mm, 126 * mm], align_right=(1,)))

    s.append(P(
        "Two of these deserve a sentence. <b>Percentile rank</b> exists "
        "because discharge spans two orders of magnitude between basins: "
        "800&nbsp;m<super>3</super>/s is a drought on the Brahmaputra and a disaster on "
        "a Konkan stream, and a model fed raw magnitudes has to learn that "
        "mapping through terrain, which it does badly. <b>Basin dummies</b> "
        "are deliberately coarse: basin, not district. District identity as a "
        "feature would be memorisation dressed up as signal, and it would "
        "collapse the first time the system was pointed at a district it had "
        "never seen.", BODY))

    s.append(P("2.6 &nbsp; Leakage controls", H3))
    s.append(P(
        "These are enforced by assertions in <font face='Courier'>tests/"
        "test_no_leakage.py</font> and <font face='Courier'>tests/"
        "test_live.py</font> rather than left as good intentions. Eighteen "
        "tests pass.", BODY))
    s.append(table([
        ["Control", "What it prevents"],
        ["Every rolling window uses <font face='Courier'>closed=\"left\"</font>",
         "Today's own value entering the window that is supposed to describe "
         "its past"],
        ["All lag and rolling operations grouped by district",
         "One district's history bleeding into another's features"],
        ["Climatology fitted on the training split only",
         "Test years informing the baseline they are then measured against"],
        ["Percentile reference distributions fitted on the training split only",
         "A test-year value helping to define its own percentile"],
        ["NaN fill uses training-period medians",
         "Test-set statistics leaking in through imputation"],
        ["Splits are strictly chronological, asserted disjoint and ordered",
         "The future being used to predict the past &#8212; what a random "
         "shuffle would do to a time series"],
        ["Operating threshold chosen on validation, then frozen",
         "Tuning the decision rule on the data used to report performance"],
        ["Train-time references saved to <font face='Courier'>"
         "feature_reference.pkl</font> and reloaded at serving time",
         "Train/serve skew: live scoring silently re-deriving its own scaling "
         "and feeding the model inputs it never saw in training"],
    ], widths=[72 * mm, 98 * mm]))

    # ================= 3. TRAINING =================
    s.append(P("3 &nbsp; Training", H2))

    s.append(P("3.1 &nbsp; The split", H3))
    s.append(P(
        "Strictly time-ordered, never shuffled. The boundaries were chosen so "
        "that the anchor event (Kerala, August 2018) falls in test while train "
        "still spans fifteen years &#8212; enough for sparse DFO events to "
        "appear in every window.", BODY))
    s.append(table([
        ["Split", "Period", "Rows", "Positive rate"],
        ["Train", "2000-01-06 &#8594; 2014-12-31", "120,428", "1.67%"],
        ["Validation", "2015-01-01 &#8594; 2016-12-31", "16,082", "1.44%"],
        ["Test", "2017-01-01 &#8594; 2023-12-30", "56,210", "1.62%"],
        ["<b>Total</b>", "", "<b>192,720</b>", "<b>1.63%</b>"],
    ], widths=[26 * mm, 62 * mm, 32 * mm, 50 * mm], align_right=(2, 3)))
    s.append(P("132 rows were dropped for insufficient lag history at the "
               "start of each district's series. Validation is used for one "
               "purpose only: choosing the alert threshold.", CAP))

    s.append(P(
        "A guard in <font face='Courier'>src/models/train_risk.py</font> "
        "refuses to train if any split contains zero positives, printing the "
        "class counts and the three ways to fix it. This is the normal "
        "failure mode with real flood archives &#8212; large events are "
        "sparse and can miss a window entirely &#8212; and the guard exists "
        "because the alternative is a cryptic scikit-learn error and the "
        "temptation to shuffle the split, which would leak the future.", BODY))

    s.append(P("3.2 &nbsp; Models", H3))
    s.append(P(
        "All models see the same features and the same split, so the "
        "comparison is fair. Logistic regression is included as the baseline "
        "that has to be beaten &#8212; if the ensembles do not beat it, that "
        "is a finding to report, not a failure to hide.", BODY))
    s.append(table([
        ["Model", "Configuration", "Imbalance handling"],
        ["Logistic regression",
         "StandardScaler &#8594; LogisticRegression, "
         "<font face='Courier'>max_iter=2000</font>",
         "<font face='Courier'>class_weight=\"balanced\"</font>"],
        ["Random forest",
         "<font face='Courier'>n_estimators=400, max_depth=12, "
         "min_samples_leaf=5</font>",
         "<font face='Courier'>class_weight=\"balanced\"</font>"],
        ["XGBoost <font color='#64748b'>(optional)</font>",
         "<font face='Courier'>n_estimators=500, max_depth=6, lr=0.05, "
         "subsample=0.8, colsample_bytree=0.8</font>",
         "<font face='Courier'>scale_pos_weight</font>, computed from the "
         "training split only"],
    ], widths=[32 * mm, 83 * mm, 55 * mm]))
    s.append(P("<font face='Courier'>random_state=42</font> throughout. "
               "XGBoost is skipped with a printed message if the library is "
               "unavailable, so the pipeline never half-fails; the results in "
               "section 4 are from a run without it.", CAP))

    s.append(P(
        "Class imbalance is handled by reweighting rather than by resampling. "
        "At a 1.63% positive rate, oversampling floods or undersampling quiet "
        "days would distort the base rate the probabilities are supposed to "
        "express, and the calibration reported in section 4.4 would become "
        "meaningless. Reweighting changes what the loss function cares about "
        "while leaving the data intact.", BODY))

    # ================= 4. EVALUATION =================
    s.append(P("4 &nbsp; Evaluation metrics", H2))
    s.append(P(
        "Metrics are listed in order of how much they matter for a warning "
        "system, which is not the order a classification tutorial would use. "
        "Implemented in <font face='Courier'>src/models/evaluate.py</font>.",
        BODY))

    s.append(P("4.1 &nbsp; Lead time and alarm duty cycle &#8212; the headline pair", H3))
    s.append(P(
        "For each recorded flood event, lead time is the number of hours "
        "between the first threshold crossing in the unbroken run of alarm "
        "days leading into onset, and the onset itself. An event with no "
        "alarm on the day before onset counts as a miss even if an isolated "
        "spike occurred a week earlier, because that is what an operator "
        "would actually experience.", BODY))
    s.append(P(
        "Lead time alone is gameable: a model that leaves the alarm on all "
        "monsoon shows enormous lead time and is operationally useless. It is "
        "therefore always reported beside the <b>alarm duty cycle</b> &#8212; "
        "the share of district-days on which the alarm is raised. A good "
        "system has long lead time <i>and</i> a low duty cycle. Neither number "
        "means anything on its own.", BODY))

    s.append(P("4.2 &nbsp; Cost-sensitive threshold", H3))
    s.append(P(
        "The operating threshold is not 0.5. It is chosen on the validation "
        "split to minimise expected cost at a configured ratio of "
        "<b>20:1</b>, missing a flood against raising a false alarm "
        "(<font face='Courier'>cost_false_negative: 20</font>, "
        "<font face='Courier'>cost_false_positive: 1</font> in config.yaml). "
        "The curve is swept across 99 thresholds and the minimum is taken.",
        BODY))
    s.append(P(
        "This produces <b>low precision by design</b>. Saying that twenty "
        "false alarms are worth one additional flood caught is the same as "
        "saying most alarms will come to nothing. Because that reads as a "
        "defect when quoted alone, a second operating point &#8212; the "
        "F1-optimal threshold, weighting precision and recall equally &#8212; "
        "is reported from the same model. The pair is the honest answer; "
        "either number alone misleads in a different direction.", BODY))

    s.append(P("4.3 &nbsp; Discrimination", H3))
    s.append(P(
        "PR-AUC is reported ahead of ROC-AUC. At a 1.63% positive rate, ROC "
        "curves look flattering because the true-negative pool is enormous, "
        "while precision-recall curves stay honest about how rare the "
        "positives are. The confusion matrix at the chosen threshold is given "
        "in full, and recall is quoted before precision because recall is the "
        "number that corresponds to people not being warned.", BODY))

    s.append(P("4.4 &nbsp; Calibration &#8212; and the fact that these models have none", H3))
    s.append(P(
        "Reported as a Brier score plus a ten-bin reliability curve. If a "
        "model says 30%, floods should begin on roughly 30% of such days. "
        "These models do not clear that bar, and the report says so rather "
        "than quoting the Brier score and moving on.", BODY))
    s.append(table([
        ["", "Logistic", "Random forest", "Always predict zero"],
        ["Mean predicted risk", "0.269", "0.112", "0.000"],
        ["Actual onset rate", "0.016", "0.016", "0.016"],
        ["Over-statement", "16.6&#215;", "6.9&#215;", "&#8212;"],
        ["Brier score", "0.162", "0.060", "<b>0.016</b>"],
    ], widths=[46 * mm, 38 * mm, 42 * mm, 44 * mm], align_right=(1, 2, 3)))
    s.append(P("Measured on the test split, 56,210 district-days.", CAP))
    s += callout(
        "What this means, and what it does not",
        "Both models over-state risk, and both score worse on Brier than a "
        "model that always predicts zero. So the output is a <b>ranking "
        "score, not a probability</b>: a reading of 0.30 does not mean a 30% "
        "chance. It also explains an oddity in the results &#8212; the "
        "cost-optimal threshold lands at 0.71 rather than near the 1.6% base "
        "rate, because the threshold is absorbing the inflation. Ranking is "
        "unaffected, which is why lead time and event detection remain "
        "meaningful. The fix is to wrap each estimator in "
        "<font face='Courier'>CalibratedClassifierCV</font> (isotonic, fitted "
        "on the validation split) before the thresholds are chosen; it is "
        "recorded as future work rather than half-applied, because it moves "
        "every number in this document.",
        accent=colors.HexColor("#1d4ed8"), bg=colors.HexColor("#eff6ff"))

    s.append(P("4.5 &nbsp; Results", H3))
    s.append(P("Held-out test split, 2017&#8211;2023, 130 recorded flood "
               "onsets. Threshold fixed on validation and applied unchanged.",
               BODY))
    s.append(table([
        ["Metric", "Logistic regression", "Random forest"],
        ["Events warned", "84 / 130 &nbsp;(64.6%)", "66 / 130 &nbsp;(50.8%)"],
        ["Median lead time", "312 h &nbsp;(13.0 days)", "120 h &nbsp;(5.0 days)"],
        ["Mean lead time", "236.6 h &nbsp;(9.9 days)", "139.6 h &nbsp;(5.8 days)"],
        ["Alarm duty cycle", "14.65%", "10.46%"],
        ["Recall", "0.545", "0.393"],
        ["Precision", "0.060", "0.061"],
        ["F1", "0.108", "0.105"],
        ["PR-AUC", "0.095", "0.061"],
        ["ROC-AUC", "0.780", "0.787"],
        ["Brier score", "0.162", "0.060"],
        ["Operating threshold", "0.71", "0.41"],
        ["Confusion (tp / fp / fn / tn)", "496 / 7,740 / 414 / 47,560",
         "358 / 5,521 / 552 / 49,779"],
        ["Expected cost at 20:1", "16,020", "16,561"],
    ], widths=[52 * mm, 59 * mm, 59 * mm], align_right=(1, 2)))
    s.append(P("Source: outputs/reports/summary_h1.json. Lead times converted "
               "from hours for readability; the stored values are hours.", CAP))

    s.append(P(
        "Read operationally: <b>logistic regression is the better warning "
        "system here</b>, catching roughly two-thirds of real onsets with a "
        "median of thirteen days' notice, for an alarm on about one "
        "district-day in seven. The random forest has the smaller Brier score "
        "(0.060 against 0.162) and is marginally better at ranking (ROC-AUC "
        "0.787), but section 4.4 shows neither model beats the trivial "
        "baseline on Brier, and the forest warns half the events at a third "
        "of the notice. "
        "That the simple baseline wins on the metric that matters is a "
        "result worth reporting, not an embarrassment to bury.", BODY))

    s += callout(
        "Why accuracy does not appear in that table",
        "Predicting &#8220;no flood&#8221; every single day scores "
        "<b>98.38%</b> accuracy on this test set, because flood onsets begin "
        "on 1.62% of district-days. Both models score <i>below</i> that "
        "do-nothing baseline at their F1-optimal thresholds &#8212; 85.5% and "
        "90.1% &#8212; and both are nonetheless useful, because they warn "
        "most real events many days early. Accuracy is not a weak metric at "
        "this base rate; it is an actively misleading one, and the system "
        "prints the all-negative baseline beside every accuracy figure so "
        "that the comparison cannot be avoided.",
        accent=colors.HexColor("#b91c1c"), bg=colors.HexColor("#fef2f2"))

    s.append(P("4.6 &nbsp; On the precision figure", H3))
    s.append(P(
        "A precision of roughly 0.06 deserves an explanation rather than a "
        "footnote, and it has two causes in known proportions. The first is "
        "deliberate: the 20:1 cost ratio instructs the model to accept about "
        "twenty false alarms rather than miss one onset, and precision is the "
        "price of that instruction. Moving to the F1-optimal threshold "
        "changes it very little (0.061 &#8594; 0.064 for the forest), which "
        "shows the second cause is the larger one: irreducible label noise. "
        "Coarse DFO polygons, one sample point per district, and ERA5's "
        "smoothing of extremes together put a ceiling on precision that no "
        "amount of model tuning will lift. The honest conclusion is that this "
        "system is useful for deciding <i>where to look</i>, and is not a "
        "basis for acting alone.", BODY))

    s.append(P("4.7 &nbsp; A case where the model beat the archive", H3))
    s.append(P(
        "Kerala, August 2018. The flood peaked on 15&#8211;17 August and "
        "Thrissur was badly hit. Scoring that week gives:", BODY))
    s.append(table([
        ["Thrissur, 2018", "13 Aug", "14 Aug", "15 Aug", "16 Aug", "17 Aug"],
        ["Rainfall (mm)", "34.7", "80.7", "86.8", "70.3", "33.6"],
        ["Discharge (m<super>3</super>/s)", "22.3", "45.6", "84.9", "84.1", "70.8"],
        ["Risk (logistic, alert at 0.71)", "0.556",
         "<b>0.749</b>", "<b>0.910</b>", "<b>0.802</b>", "0.540"],
        ["DFO label", "0", "0", "0", "0", "0"],
    ], widths=[46 * mm, 24 * mm, 25 * mm, 25 * mm, 25 * mm, 25 * mm],
        align_right=(1, 2, 3, 4, 5)))
    s.append(P(
        "The model raised the alarm on 14 August &#8212; two days early "
        "&#8212; and held it through the peak. Every one of those days was "
        "scored as a false positive, because the label is zero. Two separate "
        "failures stack up. <b>First, the event is not in the archive.</b> "
        "DFO's only Kerala-area event overlapping August 2018 runs "
        "7&#8211;10 August with 20 fatalities, which is the first wave; the "
        "catastrophe of 15&#8211;17 August, with a death toll reported in "
        "the hundreds, has no record. The nearest candidate, an India event "
        "dated 15 August to 7 September, has its polygon at "
        "93.7&#8211;95.8&#176;E / 24.9&#8211;27.2&#176;N &#8212; Nagaland "
        "and Manipur, some 23 degrees away. <b>Second, Thrissur's sample "
        "point misses the polygon that does exist</b>, falling 0.029&#176; "
        "&#8212; roughly 3 km &#8212; outside the 7&#8211;10 August "
        "footprint. Palakkad and Wayanad miss it too.", BODY))
    s.append(P(
        "Across the Kerala basin from 11 to 20 August 2018 the model raised "
        "21 district-day alarms in 7 districts, all scored false. That single "
        "episode is 0.3% of the logistic model's 7,740 reported false "
        "positives. The reported precision of 0.06 is therefore "
        "<b>pessimistic by an unknown amount</b>: some share of what is "
        "counted against the model is the archive being incomplete rather "
        "than the model being wrong. The claim is checkable &#8212; the "
        "labelling is in <font face='Courier'>src/data/loaders.py</font> and "
        "the GeoPackage is a CC0 download.", BODY))
    s.append(P(
        "The random forest peaked at 0.017 that week against its own 0.46 "
        "threshold, missing the largest flood in the record completely. That "
        "is why the console defaults to logistic regression.", BODY))

    # ================= 5. LIMITATIONS =================
    # Keep the heading with its table: it was orphaning at a page foot.
    s.append(KeepTogether([
        P("5 &nbsp; Limitations to state in the report", H2),
        table([
        ["Limitation", "Consequence"],
        ["ERA5 is ~25 km and smooths extremes; GloFAS is modelled routing, "
         "not gauge stage",
         "Peak intensities are under-represented, which caps achievable "
         "precision"],
        ["One sample point stands in for an entire district",
         "A flood affecting part of a district may be labelled wrongly in "
         "either direction"],
        ["DFO records only large events",
         "Smaller, locally significant floods are absent from the labels and "
         "count as negatives"],
        ["The archive misses events outright &#8212; the Kerala flood of "
         "15&#8211;17 August 2018 is not in it (section 4.7)",
         "Reported precision is pessimistic by an unknown amount, because "
         "some false alarms are real floods with no record"],
        ["Model outputs are not calibrated probabilities (section 4.4)",
         "A score of 0.30 is not a 30% chance. Ranking, lead time and event "
         "detection hold; the numeric values do not"],
        ["The live feed is analysis and forecast; the model was trained on "
         "reanalysis",
         "Train/serve skew. Live output carries more uncertainty than the "
         "backtest implies, and the gap widens with lead time. The rigorous "
         "fix needs an archive of past forecasts, which is not available"],
        ["Forecast skill decays with lead time",
         "A six-day-ahead score is built largely from forecast rainfall. Each "
         "output row carries a <font face='Courier'>forecast_share</font> "
         "column so this is visible rather than hidden"],
        ["Evaluated on 22 districts in two basins",
         "Results should not be assumed to transfer to basins not in the "
         "training set"],
        ], widths=[74 * mm, 96 * mm]),
    ]))


    # ================= APPENDIX A =================
    s.append(PageBreak())
    s.append(P("Appendix A &nbsp; The 59 features in full", H2))
    s.append(P(
        "Read straight out of <font face='Courier'>features_h1.parquet</font> "
        "by <font face='Courier'>feature_columns()</font>, in the order the "
        "model receives them. Every rolling window is <b>left-closed</b>: it "
        "covers the days <i>before</i> t, never t itself, so the current day "
        "cannot enter the window that is meant to describe its own past. "
        "Everything is grouped per district, so one district's history never "
        "reaches another's row.", BODY))

    rolls = "Rolling {stat} of {what} over the preceding {n} days."
    s.append(feature_table([
        ("Measured series", [
            ("rainfall_mm",
             "ERA5 daily precipitation total at the district point, in mm."),
            ("river_discharge_m3s",
             "GloFAS daily mean river discharge for the grid cell containing "
             "the point, in m<super>3</super>/s. Modelled routing, not a gauge "
             "stage reading."),
        ]),
        ("Terrain (static per district)", [
            ("mean_elevation_m",
             "Elevation at the district point, in metres. Note the column "
             "name: the stored value is the centre sample of the five-point "
             "cross, not the mean of the five."),
            ("mean_slope_deg",
             "Slope in degrees, from the finite-difference gradient of the "
             "five elevation samples taken at &#177;0.05&#176; (~5.5 km)."),
            ("terrain_ruggedness_m",
             "Standard deviation of those five elevation samples &#8212; a "
             "proxy for local relief."),
        ]),
        ("Rainfall lags", [
            ("rainfall_mm_lag1", "Rainfall one day earlier."),
            ("rainfall_mm_lag2", "Rainfall two days earlier."),
            ("rainfall_mm_lag3", "Rainfall three days earlier."),
            ("rainfall_mm_lag5", "Rainfall five days earlier."),
            ("rainfall_mm_lag7", "Rainfall seven days earlier."),
        ]),
        ("Rainfall rolling windows", [
            ("rainfall_mm_rollmean3", rolls.format(stat="mean", what="rainfall", n=3)),
            ("rainfall_mm_rollmax3", rolls.format(stat="maximum", what="rainfall", n=3)),
            ("rainfall_mm_rollsum3", rolls.format(stat="total", what="rainfall", n=3)),
            ("rainfall_mm_rollmean7", rolls.format(stat="mean", what="rainfall", n=7)),
            ("rainfall_mm_rollmax7", rolls.format(stat="maximum", what="rainfall", n=7)),
            ("rainfall_mm_rollsum7", rolls.format(stat="total", what="rainfall", n=7)),
            ("rainfall_mm_rollmean14", rolls.format(stat="mean", what="rainfall", n=14)),
            ("rainfall_mm_rollmax14", rolls.format(stat="maximum", what="rainfall", n=14)),
            ("rainfall_mm_rollsum14", rolls.format(stat="total", what="rainfall", n=14)),
            ("rainfall_mm_rollmean30", rolls.format(stat="mean", what="rainfall", n=30)),
            ("rainfall_mm_rollmax30", rolls.format(stat="maximum", what="rainfall", n=30)),
            ("rainfall_mm_rollsum30", rolls.format(stat="total", what="rainfall", n=30)),
        ]),
        ("Discharge lags", [
            ("river_discharge_m3s_lag1", "Discharge one day earlier."),
            ("river_discharge_m3s_lag2", "Discharge two days earlier."),
            ("river_discharge_m3s_lag3", "Discharge three days earlier."),
            ("river_discharge_m3s_lag5", "Discharge five days earlier."),
            ("river_discharge_m3s_lag7", "Discharge seven days earlier."),
        ]),
        ("Discharge rolling windows &#8212; mean and max only", [
            ("river_discharge_m3s_rollmean3", rolls.format(stat="mean", what="discharge", n=3)),
            ("river_discharge_m3s_rollmax3", rolls.format(stat="maximum", what="discharge", n=3)),
            ("river_discharge_m3s_rollmean7", rolls.format(stat="mean", what="discharge", n=7)),
            ("river_discharge_m3s_rollmax7", rolls.format(stat="maximum", what="discharge", n=7)),
            ("river_discharge_m3s_rollmean14", rolls.format(stat="mean", what="discharge", n=14)),
            ("river_discharge_m3s_rollmax14", rolls.format(stat="maximum", what="discharge", n=14)),
            ("river_discharge_m3s_rollmean30", rolls.format(stat="mean", what="discharge", n=30)),
            ("river_discharge_m3s_rollmax30", rolls.format(stat="maximum", what="discharge", n=30)),
        ]),
        ("Accumulation and rate of change", [
            ("antecedent_rain_mm",
             "Total rainfall over the preceding 15 days &#8212; how saturated "
             "the ground already is. This is the term that decides whether a "
             "given rainfall floods or drains."),
            ("river_discharge_delta_1d",
             "Change in discharge since the previous day. A rising river is "
             "more dangerous than a high static one."),
            ("river_discharge_delta_3d", "Change in discharge over three days."),
        ]),
        ("Climatology departure", [
            ("rainfall_climatology_mm",
             "Mean rainfall for this district on this day-of-year, fitted on "
             "the <b>training split only</b> and smoothed over a 15-day "
             "centred window."),
            ("rainfall_departure_mm",
             "Today's rainfall minus that climatology &#8212; how unusual "
             "today is for the time of year."),
            ("rainfall_departure_ratio",
             "Today's rainfall divided by that climatology."),
        ]),
        ("Seasonality", [
            ("doy_sin",
             "sin(2&#960; &#215; day-of-year / 365.25). Cyclical, so 31 "
             "December sits next to 1 January instead of 364 days away."),
            ("doy_cos", "cos(2&#960; &#215; day-of-year / 365.25)."),
            ("is_monsoon", "1 for June to September, 0 otherwise."),
        ]),
        ("Basin context &#8212; same day, other districts", [
            ("basin_rainfall_mm_mean",
             "Mean rainfall across every district in the same river basin on "
             "the same day."),
            ("basin_rainfall_mm_max",
             "Maximum rainfall anywhere in the basin that day."),
            ("rainfall_mm_vs_basin",
             "This district's rainfall divided by its basin mean &#8212; how "
             "extreme it is relative to its neighbours."),
            ("basin_river_discharge_m3s_mean", "Basin mean discharge that day."),
            ("basin_river_discharge_m3s_max", "Basin maximum discharge that day."),
            ("river_discharge_m3s_vs_basin",
             "This district's discharge divided by its basin mean."),
            ("basin_antecedent_mean",
             "Basin mean of the 15-day antecedent rainfall &#8212; a cheap "
             "proxy for upstream saturation, without building a routing model."),
        ]),
        ("Percentile rank within the district's own history", [
            ("rainfall_mm_pctile",
             "Where today's rainfall sits in this district's "
             "<b>training-period</b> rainfall distribution, 0 to 1."),
            ("river_discharge_m3s_pctile",
             "The same for discharge. This is what makes 800&nbsp;m"
             "<super>3</super>/s readable as a drought on the Brahmaputra and "
             "a disaster on a Konkan stream."),
            ("antecedent_rain_mm_pctile",
             "The same for 15-day antecedent rainfall."),
        ]),
        ("River basin, one-hot", [
            ("basin_Brahmaputra", "1 if the district is in the Brahmaputra basin."),
            ("basin_Kerala", "1 if the district is in the Kerala basin."),
        ]),
        ("Terrain interactions", [
            ("rain_x_inv_slope",
             "rainfall / (slope + 1). The same rainfall is more dangerous on "
             "flat ground that drains slowly."),
            ("antecedent_x_ruggedness",
             "15-day antecedent rainfall &#215; ruggedness."),
            ("elevation_inv",
             "1 / (elevation + 10). Large for low-lying districts."),
        ]),
    ], widths=[8 * mm, 52 * mm, 110 * mm]))

    s.append(P(
        "The basin one-hot has two columns because this configuration covers "
        "two basins. Running the pipeline over all 12 basins in "
        "<font face='Courier'>districts.py</font> produces ten more columns "
        "and a feature count of 69; nothing else in the list changes.", CAP))

    s.append(P("A.1 &nbsp; What is deliberately excluded", H3))
    s.append(P(
        "Columns that exist in the table and are kept out of the model on "
        "purpose. Listing them matters as much as listing the features: each "
        "one is a way the model could have cheated.", BODY))
    s.append(table([
        ["Column", "Why it is not a feature"],
        ["<font face='Courier'>lat</font>, <font face='Courier'>lon</font>",
         "Coordinates would let a tree carve out individual districts and "
         "memorise their flood history. Location enters only through terrain "
         "and the basin one-hot."],
        ["<font face='Courier'>district</font>",
         "District identity is memorisation dressed up as signal. A model "
         "using it would collapse the first time it was pointed at a district "
         "it had never seen."],
        ["<font face='Courier'>basin</font>, <font face='Courier'>state</font>",
         "Kept as columns for grouping and reporting. Basin reaches the model "
         "only as the coarse one-hot above; state never does."],
        ["<font face='Courier'>date</font>",
         "A raw timestamp lets a model learn the calendar of the training "
         "years. Time enters only through the cyclical day-of-year terms."],
        ["<font face='Courier'>flood_occurred</font>",
         "Today's label. Feeding it alongside a target built from the same "
         "series is the most direct leak available."],
        ["<font face='Courier'>rainfall_mm_was_missing</font><br/>"
         "<font face='Courier'>river_discharge_m3s_was_missing</font>",
         "Imputation flags, kept so the share of interpolated rows can be "
         "reported. They describe the data-collection process, not the "
         "weather."],
    ], widths=[52 * mm, 118 * mm]))

    s += rule(10, 6)
    s.append(P(
        "FloodWatch is a student project and decision support. It is not an "
        "operational warning service, and nothing it produces is an "
        "evacuation order. India's official flood forecasts come from the "
        "Central Water Commission and the India Meteorological Department.",
        NOTE))

    return s


if __name__ == "__main__":
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    build()
    print("wrote", OUT)
