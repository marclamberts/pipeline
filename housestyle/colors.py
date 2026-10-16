"""
Meridian house-style palette.

Every categorical set below passes the dataviz-skill validator
(scripts/validate_palette.js) against its own surface:

  Light — adjacent pairs:  node validate_palette.js "<CATEGORICAL_LIGHT>" \\
            --mode light --surface "#F7F1E3"
  Dark  — adjacent pairs:  node validate_palette.js "<CATEGORICAL_DARK>" \\
            --mode dark --surface "#14202B"
  First three slots additionally clear --pairs all in both modes, so they
  are the only slots safe for scatter/bubble/choropleth use (see STYLEGUIDE.md).

Do not reorder the categorical lists — the order IS the CVD-safety mechanism
(adjacent-pair separation depends on which hues sit next to which).
"""

# ---------------------------------------------------------------------------
# Chrome & ink — the paper-and-ink surface that anchors the identity
# ---------------------------------------------------------------------------

LIGHT = {
    "surface": "#F7F1E3",       # warm paper — the signature background
    "page": "#FBF7EC",          # surrounding page plane, one step lighter
    "ink_primary": "#1B1B18",   # headline / value text
    "ink_secondary": "#6B6558", # dek, axis titles
    "ink_muted": "#8C8570",     # tick labels, footnotes
    "grid": "#E4DCC8",          # hairline gridlines
    "axis": "#C9BFA5",          # baseline / axis rule
    "border": "rgba(27,27,24,0.10)",
    "accent": "#C1512F",        # signature terracotta — the "flag" color
}

DARK = {
    "surface": "#14202B",
    "page": "#0C1219",
    "ink_primary": "#F2EFE6",
    "ink_secondary": "#B7B0A0",
    "ink_muted": "#8A8778",
    "grid": "#263440",
    "axis": "#3C4A57",
    "border": "rgba(242,239,230,0.12)",
    "accent": "#E8794C",        # terracotta, stepped for the dark surface
}

# ---------------------------------------------------------------------------
# Categorical — fixed order, never cycled or reassigned by rank
# ---------------------------------------------------------------------------

CATEGORICAL_LIGHT = [
    "#2A5C99",  # 1 ink blue      (primary series / "us")
    "#C1512F",  # 2 terracotta    (signature accent — reserve for emphasis)
    "#0C8A76",  # 3 teal
    "#C6932B",  # 4 ochre  (sub-3:1 on paper — pair with a direct label)
    "#9C3E82",  # 5 plum
    "#74881F",  # 6 moss
    "#6A4C9C",  # 7 violet
    "#B23A55",  # 8 crimson
]

CATEGORICAL_DARK = [
    "#5B8AC9",  # 1 ink blue
    "#D9663A",  # 2 terracotta
    "#1FA88C",  # 3 teal
    "#B37F1E",  # 4 ochre
    "#C168A8",  # 5 plum
    "#7C9A2E",  # 6 moss
    "#9A82CE",  # 7 violet
    "#D9637F",  # 8 crimson
]

# First 3 slots only: safe for scatter / bubble / small-multiples (all-pairs CVD)
CATEGORICAL_SCATTER_SAFE_LIGHT = CATEGORICAL_LIGHT[:3]
CATEGORICAL_SCATTER_SAFE_DARK = CATEGORICAL_DARK[:3]

# ---------------------------------------------------------------------------
# Sequential — single hue (blue), light -> dark, for magnitude
# ---------------------------------------------------------------------------

SEQUENTIAL_BLUE_LIGHT = {
    100: "#D9E5F4",
    200: "#B9CEEA",
    300: "#93B2DD",
    400: "#6C95CE",
    500: "#2A5C99",  # matches categorical slot 1
    600: "#1F4676",
    700: "#142E4E",
}

SEQUENTIAL_BLUE_DARK = {
    100: "#1B2C40",
    200: "#233D5C",
    300: "#2E5079",
    400: "#3E6B9E",
    500: "#5B8AC9",  # matches categorical slot 1 (dark)
    600: "#88ACDA",
    700: "#B8CDE8",
}

# ---------------------------------------------------------------------------
# Diverging — blue (cool) <-> terracotta (warm), neutral paper-gray midpoint
# ---------------------------------------------------------------------------

DIVERGING_LIGHT = {
    "cool": ["#93B2DD", "#6C95CE", "#2A5C99", "#1F4676"],
    "neutral": "#E3DDCC",
    "warm": ["#E3A47F", "#D97A4C", "#C1512F", "#8F3A1E"],
}

DIVERGING_DARK = {
    "cool": ["#2E5079", "#3E6B9E", "#5B8AC9", "#88ACDA"],
    "neutral": "#33404A",
    "warm": ["#8F3A1E", "#D9663A", "#E8794C", "#EFA582"],
}

# ---------------------------------------------------------------------------
# Status — fixed, never themed, never reused for a categorical series
# ---------------------------------------------------------------------------

STATUS_LIGHT = {
    "good": "#2E7D46",
    "warning": "#D9A31B",
    "serious": "#D9682E",
    "critical": "#B2242E",
}

STATUS_DARK = {
    "good": "#3FAE64",
    "warning": "#E6B94A",
    "serious": "#E67D3E",
    "critical": "#D9434B",
}

# ---------------------------------------------------------------------------
# Texture (accessibility channel — CVD / print / forced-colors only)
# ---------------------------------------------------------------------------

TEXTURE_ANGLES = (45, 135)  # "Lines" fill, inked tone-on-tone of the mark's own hue
