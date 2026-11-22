# WA Fog theme

Cool, light colour theme in the Waltzing Analytics 2x3 grid (Cream / Clay / Espresso, Fog / Dusk / Navy).

| Token | Hex | Use |
|---|---|---|
| CANVAS | #E7ECEE | figure background |
| INK | #1A2226 | titles, labels, values |
| MUTED | #78888D | footer, axis ticks |
| MUTED2 | #A9B7BB | subtitle, byline |
| HAIR | #CBD6D9 | gridlines, dividers |
| CORAL | #E8794C | brand accent, primary series, FWD |
| MID | #4C7A93 | secondary series (steel blue) |
| DEF | #5E8C6A | third series (green) |

Type: Georgia (serif headline), Helvetica Neue (everything else).

## Files
- `wa_style_fog.py`: importable module, same API as the other theme modules (`setup_figure`, `draw_masthead`, `draw_footer` plus colour constants). Master copy lives in `7 Models/WAR/war_analysis/`; keep both in sync if colours change.
- `build_fog_preview.py`: renders `wa_style_fog_preview.png`, a reference chart with the palette swatches.

## Usage
```python
import sys
sys.path.insert(0, "/Users/marclamberts/Downloads/01 Football & Work/11 Content & Business/Fog Style")
from wa_style_fog import setup_figure, draw_masthead, draw_footer, CORAL, INK, HAIR
```
