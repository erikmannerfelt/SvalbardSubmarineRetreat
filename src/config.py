from pathlib import Path
from typing import Final, Literal, TypeAlias

#: Cache-invalidation mode shared by every derived cache in the project.
#: ``False`` reuses an existing cache. ``"self"`` recomputes the current
#: artifact while its dependencies keep theirs. ``"all"`` propagates the
#: recomputation through the derived-cache chain. Source downloads are not
#: covered by this and take a separate ``refresh_download`` flag.
ForceRedo: TypeAlias = Literal[False, "self", "all"]

PROJECT_DIR: Final = Path(__file__).resolve().parent.parent
INPUT_DIR: Final = PROJECT_DIR / "input"
SHAPES_DIR: Final = PROJECT_DIR / "shapes"
CACHE_DIR: Final = PROJECT_DIR / "cache"
PRODUCTS_DIR: Final = PROJECT_DIR / "products"

CACHE_DIR.mkdir(exist_ok=True)

CRS_EPSG: Final = 32633

RGI7_ZIP: Final = INPUT_DIR / "RGI7" / "RGI2000-v7.0-G-07_svalbard_jan_mayen.zip"
OUTLINE_CORRECTIONS_PATH: Final = SHAPES_DIR / "outline_corrections.geojson"
COASTLINE_DIR: Final = INPUT_DIR / "coastlines"

#: The three NPI annual-coastline years the change polygons are built from.
COASTLINE_YEARS: Final = (2012, 2018, 2023)

#: Years `bed.marine_domain` checks for "was this ever open water", a
#: broader set than `COASTLINE_YEARS`: a cell classified "land" throughout
#: 2012-2023 can still be true marine bed if it was drowned by a pre-2012
#: glacier advance/surge and never retreated back into view since --
#: confirmed for the Nathorstbreen Glacier System, which surged ~2008.
#: 2008 is the only pre-2012 NPI coastline year used; older ones in NPI's
#: archive are left out until independently vetted.
MARINE_DOMAIN_YEARS: Final = (2008, *COASTLINE_YEARS)

SVIFT_TAR: Final = INPUT_DIR / "svift" / "svift_v11.tar.gz"
SVIFT_DIR: Final = INPUT_DIR / "svift" / "svift_v11"
#: The marine bathymetry source for every `bed.py` model (IBCAO v5.2).
IBCAO_DIR: Final = INPUT_DIR / "ibcao"
MAFFEZZOLI_DIR: Final = INPUT_DIR / "maffezzoli"
MILLAN_DIR: Final = INPUT_DIR / "millan"
FRANK_DIR: Final = INPUT_DIR / "frank"
VANPELT_DIR: Final = INPUT_DIR / "vanpelt"
DEM_ZIP: Final = INPUT_DIR / "dem" / "NP_S0_DTM50.zip"
DEM_DIR: Final = INPUT_DIR / "dem" / "NP_S0_DTM50"
