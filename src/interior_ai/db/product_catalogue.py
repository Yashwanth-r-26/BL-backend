"""Bulk product catalogue -- 10+ items per class, Indian market dimensions.

Every dimension here is drawn from published Indian furniture sizing rather
than invented, because the fit engine vetoes products against real room
geometry: a sofa listed 300 mm too wide is not a cosmetic error, it changes
which products a customer is offered.

Reference figures used:

* Sofas -- 3-seater 1830-2285 mm wide, 2-seater 1370-1675 mm, 1-seater
  760-1060 mm; depth 850-1000 mm, height 800-900 mm.
* Beds (India) -- Single 900x1900, Double 1350x1900, Queen 1500x1900,
  King 1800x2000 mattress; frames run 100-200 mm larger per side.
* Dining -- 4-seater 1200x750, 6-seater 1500-1800x900, 8-seater
  2000-2400x1000; height fixed 750-760 mm across the market.
* Wardrobes -- depth 580-620 mm (a hanging rail plus hanger clearance).
* Circulation -- 900 mm clear walking space around furniture (BIS guidance),
  which the solver already enforces.

Prices are indicative Bengaluru retail in INR and exist so quotes are
complete on day one; real vendor prices overwrite them through /prices.

**Images are deliberately absent.** Product photographs on retail sites are
copyrighted by those retailers, so this file ships specifications only and the
loader attaches photos *you* supply -- your own product shots, or images you
have licensed. See :mod:`interior_ai.db.seed_products`.
"""

from __future__ import annotations

# (sku, name, object_class, width_mm, depth_mm, height_mm, price_inr, description)
# Descriptions matter: they are fed verbatim into the replacement prompt, so
# colour, material and silhouette here directly improve swap fidelity.
PRODUCTS: list[tuple[str, str, str, int, int, int, str, str]] = [
    # ---------------------------------------------------------- sofas (10)
    ("SOFA-MILANO-3S", "Milano 3-Seater Sofa", "sofa", 2100, 880, 820, "52000",
     "Charcoal grey fabric three-seater, tapered oak legs, boxy low-back silhouette"),
    ("SOFA-OSLO-2S", "Oslo Compact 2-Seater", "sofa", 1650, 850, 800, "38000",
     "Sage green woven two-seater, slim rolled arms, light beech legs"),
    ("SOFA-JAIPUR-L", "Jaipur L-Shape Sofa", "sofa", 2600, 1600, 850, "78000",
     "Warm sand fabric left-facing L-shape with chaise, deep seat, walnut plinth"),
    ("SOFA-UDAIPUR-3S", "Udaipur Velvet 3-Seater", "sofa", 2050, 900, 850, "58000",
     "Deep teal velvet three-seater, channel-tufted back, brass-capped legs"),
    ("SOFA-SHIMLA-CH", "Shimla Chesterfield", "sofa", 2130, 950, 880, "71000",
     "Tan leather chesterfield, deep button tufting, rolled arms, dark wood feet"),
    ("SOFA-KOCHI-2S", "Kochi Rattan 2-Seater", "sofa", 1520, 820, 790, "34000",
     "Natural rattan frame two-seater with ivory cushions, airy open sides"),
    ("SOFA-GOA-BED", "Goa Sofa-Cum-Bed", "sofa", 1980, 950, 840, "45000",
     "Slate grey convertible sofa bed, square arms, chrome legs"),
    ("SOFA-PUNE-2S", "Pune Slim 2-Seater", "sofa", 1450, 800, 780, "29500",
     "Mustard fabric compact two-seater, narrow arms, splayed wooden legs"),
    ("SOFA-MYSURU-3S", "Mysuru Teak 3-Seater", "sofa", 1980, 870, 810, "62000",
     "Solid teak frame three-seater, cream cotton cushions, exposed joinery"),
    ("SOFA-DELHI-MOD", "Delhi Modular 3-Seater", "sofa", 2200, 900, 830, "64000",
     "Stone grey modular three-seater, chunky seat cushions, hidden low base"),

    # ------------------------------------------------------ armchairs (10)
    ("AC-PUNE-1", "Pune Lounge Armchair", "armchair", 780, 800, 900, "16500",
     "Rust-orange accent chair, high curved back, tapered walnut legs"),
    ("AC-WING-CL", "Wingback Classic", "armchair", 820, 860, 1080, "22000",
     "Charcoal linen wingback, buttoned back, turned oak front legs"),
    ("AC-SWIVEL-1", "Swivel Accent Chair", "armchair", 760, 780, 850, "18500",
     "Cream boucle swivel chair on a matte black disc base"),
    ("AC-CANE-BAR", "Cane Barrel Chair", "armchair", 800, 780, 760, "14500",
     "Curved cane barrel chair, natural finish, ivory seat pad"),
    ("AC-RECLINE-1", "Comfort Recliner", "armchair", 900, 950, 1020, "34000",
     "Brown leatherette manual recliner, padded arms, chunky footrest"),
    ("AC-NORDIC-1", "Nordic Accent Chair", "armchair", 740, 760, 820, "12500",
     "Light grey fabric chair, slim profile, angled birch legs"),
    ("AC-TUB-VEL", "Velvet Tub Chair", "armchair", 810, 790, 740, "15800",
     "Emerald velvet tub chair, low rounded back, gold-tipped legs"),
    ("AC-ROCK-LN", "Rocking Lounge Chair", "armchair", 760, 900, 1000, "19500",
     "Beige upholstered rocker, bentwood runners, high headrest"),
    ("AC-PAPASAN", "Papasan Round Chair", "armchair", 1000, 1000, 900, "11500",
     "Rattan papasan bowl chair with thick ivory cushion"),
    ("AC-EXEC-WING", "Executive Wing Chair", "armchair", 840, 880, 1100, "26000",
     "Oxblood leather wing chair, nailhead trim, dark mahogany legs"),

    # -------------------------------------------------- coffee tables (10)
    ("CT-RIVA", "Riva Coffee Table", "coffee_table", 1000, 550, 420, "11000",
     "Sheesham wood rectangular table with slatted lower shelf"),
    ("CT-EDGE", "Edge Marble-Top Table", "coffee_table", 1200, 600, 400, "18500",
     "White Carrara marble top on a slim matte black steel frame"),
    ("CT-NEST-2", "Nesting Table Set of 2", "coffee_table", 900, 500, 450, "9500",
     "Pair of nesting tables, oak veneer tops, gold hairpin legs"),
    ("CT-ROUND-TK", "Round Teak Table", "coffee_table", 900, 900, 420, "13500",
     "Circular solid teak top, three tapered legs, honey finish"),
    ("CT-GLASS-ST", "Glass Top Steel Table", "coffee_table", 1100, 600, 400, "15000",
     "12 mm tempered glass top, brushed stainless steel base"),
    ("CT-LIVEEDGE", "Rustic Live-Edge Table", "coffee_table", 1200, 700, 430, "24000",
     "Natural live-edge acacia slab on black hairpin legs"),
    ("CT-LIFT-ST", "Lift-Top Storage Table", "coffee_table", 1050, 600, 450, "17500",
     "Grey oak lift-top with concealed storage compartment"),
    ("CT-CANE-DRUM", "Cane Drum Table", "coffee_table", 700, 700, 450, "8500",
     "Round cane-wrapped drum table, natural finish, wooden top"),
    ("CT-HEX-TWIN", "Hexagon Twin Tables", "coffee_table", 800, 700, 410, "12000",
     "Two hexagonal tables at staggered heights, marble and brass"),
    ("CT-SLAT-OAK", "Slatted Oak Table", "coffee_table", 1150, 580, 400, "16500",
     "Light oak top with slatted base and rounded corners"),

    # ---------------------------------------------------- side tables (10)
    ("ST-NOVA-C", "Nova C-Table", "side_table", 450, 400, 550, "4500",
     "C-shaped side table, walnut top, black metal frame, slides over a sofa"),
    ("ST-MARBLE-P", "Marble Pedestal Table", "side_table", 400, 400, 550, "7800",
     "Round white marble top on a fluted white pedestal"),
    ("ST-CANE-RD", "Cane Round Side Table", "side_table", 450, 450, 500, "3800",
     "Woven cane cylinder table, natural finish"),
    ("ST-TRAY-MT", "Metal Tray Table", "side_table", 400, 400, 580, "4200",
     "Removable brass tray top on a folding black frame"),
    ("ST-NEST-PR", "Nesting Pair Side Tables", "side_table", 500, 450, 550, "6500",
     "Two-tier nesting tables, oak tops, slim white legs"),
    ("ST-TEAK-BLK", "Teak Block Table", "side_table", 400, 400, 450, "5900",
     "Solid teak block side table, chunky square profile"),
    ("ST-WIRE-FR", "Wire Frame Table", "side_table", 380, 380, 520, "3200",
     "Black wire frame with a round glass top"),
    ("ST-DRUM-CER", "Ceramic Drum Table", "side_table", 420, 420, 480, "6800",
     "Glazed teal ceramic drum table, glossy finish"),
    ("ST-CONSOLE-S", "Slim Console Table", "side_table", 900, 300, 750, "12500",
     "Narrow console in dark walnut with two drawers"),
    ("ST-BEDSIDE-D", "Bedside Drawer Table", "side_table", 450, 400, 600, "7500",
     "Two-drawer bedside table, white lacquer, brass handles"),

    # -------------------------------------------------- dining tables (10)
    ("DT-COMPACT-4", "Compact 4-Seater Table", "dining_table", 1200, 750, 760, "22000",
     "Rectangular four-seater in mango wood, straight legs"),
    ("DT-EXT-46", "Extendable 4-6 Seater", "dining_table", 1400, 850, 760, "34000",
     "Butterfly-leaf extendable table, oak veneer, extends to 1800 mm"),
    ("DT-CLASSIC-6", "Classic 6-Seater Table", "dining_table", 1650, 900, 760, "42000",
     "Solid sheesham six-seater, chamfered edges, honey finish"),
    ("DT-MARBLE-6", "Marble 6-Seater Table", "dining_table", 1800, 900, 760, "68000",
     "Italian marble top on a sculpted white pedestal base"),
    ("DT-ROUND-4", "Round 4-Seater Table", "dining_table", 1100, 1100, 750, "26000",
     "Circular four-seater, white top, natural beech legs"),
    ("DT-FAMILY-8", "Family 8-Seater Table", "dining_table", 2200, 1000, 760, "78000",
     "Long eight-seater in dark walnut with trestle legs"),
    ("DT-GLASS-6", "Glass 6-Seater Table", "dining_table", 1600, 900, 750, "38000",
     "Tempered glass top on a chrome X-frame"),
    ("DT-BENCH-6", "Bench Dining Set 6", "dining_table", 1700, 850, 760, "46000",
     "Rustic pine table with two matching benches"),
    ("DT-LIVEEDGE-8", "Live-Edge 8-Seater", "dining_table", 2400, 1000, 770, "95000",
     "Single acacia slab with natural edge on black steel legs"),
    ("DT-BISTRO-2", "Bistro 2-Seater Table", "dining_table", 800, 800, 750, "14500",
     "Small square bistro table, black metal base, oak top"),

    # ------------------------------------------------------ tv units (10)
    ("TV-LINEA-18", "Linea TV Unit 1.8m", "tv_unit", 1800, 420, 500, "22000",
     "Walnut laminate console with two soft-close drawers"),
    ("TV-FLOAT-15", "Floating Wall Unit 1.5m", "tv_unit", 1500, 350, 300, "16500",
     "Wall-mounted floating unit in matte white, no visible supports"),
    ("TV-STORE-20", "Storage Console 2.0m", "tv_unit", 2000, 450, 550, "28000",
     "Grey oak console with four drawers and open central bay"),
    ("TV-COMPACT-12", "Compact TV Unit 1.2m", "tv_unit", 1200, 400, 450, "13500",
     "Small two-shelf unit in light beech, open back for cables"),
    ("TV-SWIVEL-16", "Swivel Media Unit", "tv_unit", 1600, 450, 600, "32000",
     "Console with a swivelling TV mount and closed cabinets"),
    ("TV-RUSTIC-17", "Rustic Sheesham Unit", "tv_unit", 1750, 450, 520, "36000",
     "Solid sheesham with carved fronts and iron handles"),
    ("TV-GLASS-14", "Glass Shelf Unit", "tv_unit", 1400, 400, 480, "18000",
     "Black glass shelves on a slim chrome frame"),
    ("TV-CORNER-11", "Corner TV Unit", "tv_unit", 1100, 550, 500, "15500",
     "Triangular corner unit in white with two shelves"),
    ("TV-PANEL-24", "Full Wall Media Panel", "tv_unit", 2400, 400, 2100, "68000",
     "Floor-to-ceiling panelled media wall with integrated lighting"),
    ("TV-SLAB-16", "Minimal Slab Console", "tv_unit", 1650, 380, 420, "19500",
     "Single-slab console in pale ash, push-open doors"),

    # ---------------------------------------------------------- beds (10)
    ("BED-QUEEN-STR", "Queen Bed with Storage", "bed", 1600, 2050, 900, "42000",
     "Queen frame with hydraulic under-bed storage, grey upholstered headboard"),
    ("BED-KING-HYD", "King Hydraulic Bed", "bed", 1900, 2100, 950, "58000",
     "King size with full hydraulic lift storage, walnut finish"),
    ("BED-SINGLE-CM", "Single Compact Bed", "bed", 1000, 2000, 850, "22000",
     "Single bed in white engineered wood, slatted headboard"),
    ("BED-DOUBLE-SH", "Double Sheesham Bed", "bed", 1450, 2050, 900, "36000",
     "Solid sheesham double bed, carved headboard, honey finish"),
    ("BED-QUEEN-UPH", "Queen Upholstered Bed", "bed", 1650, 2100, 1100, "52000",
     "Queen bed with tall buttoned beige fabric headboard"),
    ("BED-KING-POST", "King Four-Poster Bed", "bed", 1950, 2150, 1800, "88000",
     "Teak four-poster with slim square posts and canopy rails"),
    ("BED-QUEEN-PLT", "Queen Platform Bed", "bed", 1580, 2020, 450, "38000",
     "Low Japandi platform bed in light oak, floating base"),
    ("BED-DOUBLE-ST", "Double Storage Bed", "bed", 1500, 2050, 880, "34000",
     "Double bed with two side drawers, matte grey laminate"),
    ("BED-KING-WING", "King Wingback Bed", "bed", 1980, 2150, 1250, "76000",
     "King bed with a wing-sided velvet headboard in deep blue"),
    ("BED-TRUNDLE-S", "Single Trundle Bed", "bed", 1050, 2000, 700, "26000",
     "Single bed with a pull-out trundle underneath, white finish"),

    # ----------------------------------------------------- wardrobes (10)
    ("WR-2D-SLIDE", "2-Door Sliding Wardrobe", "wardrobe", 1500, 600, 2200, "48000",
     "Two sliding doors, one mirrored, light laminate carcass"),
    ("WR-3D-HINGE", "3-Door Hinged Wardrobe", "wardrobe", 1800, 600, 2100, "56000",
     "Three hinged doors with a central drawer bank, walnut finish"),
    ("WR-4D-MASTER", "4-Door Master Wardrobe", "wardrobe", 2400, 620, 2400, "82000",
     "Full-height four-door wardrobe with loft storage above"),
    ("WR-2D-COMPACT", "Compact 2-Door Wardrobe", "wardrobe", 1200, 580, 2000, "34000",
     "Two-door wardrobe in white with a single hanging rail"),
    ("WR-MIRROR-SL", "Mirror Sliding Wardrobe", "wardrobe", 1600, 620, 2200, "54000",
     "Full-length mirrored sliding doors, aluminium profile"),
    ("WR-WALKIN-MD", "Walk-in Wardrobe Module", "wardrobe", 2700, 650, 2400, "125000",
     "Open walk-in system with rails, shelves and drawer towers"),
    ("WR-CORNER-1", "Corner Wardrobe", "wardrobe", 1200, 1200, 2200, "62000",
     "L-shaped corner wardrobe maximising an internal angle"),
    ("WR-KIDS-2D", "Kids 2-Door Wardrobe", "wardrobe", 1000, 550, 1800, "26000",
     "Child-height wardrobe in pastel mint with rounded edges"),
    ("WR-LOFT-ST", "Loft Storage Wardrobe", "wardrobe", 1800, 600, 2600, "68000",
     "Wardrobe with an integrated loft box reaching the ceiling"),
    ("WR-OPEN-CL", "Open Closet System", "wardrobe", 1500, 550, 2000, "29000",
     "Doorless open shelving closet on a black metal frame"),

    # ---------------------------------------------------- bookshelves (10)
    ("BS-LADDER-5", "Ladder Bookshelf 5-Tier", "bookshelf", 600, 350, 1800, "12500",
     "Leaning ladder shelf, five tapering tiers, oak and black steel"),
    ("BS-WIDE-4", "Wide 4-Shelf Unit", "bookshelf", 1200, 320, 1500, "18000",
     "Low wide bookcase in white with four fixed shelves"),
    ("BS-TALL-NR", "Tall Narrow Bookshelf", "bookshelf", 400, 300, 2000, "9500",
     "Slim six-tier tower in dark laminate for tight corners"),
    ("BS-CUBE-9", "9-Cube Storage Unit", "bookshelf", 1100, 350, 1100, "14500",
     "Three-by-three cube unit in light wood, open both sides"),
    ("BS-CORNER-TW", "Corner Bookshelf Tower", "bookshelf", 500, 500, 1800, "11000",
     "Five-tier corner tower with angled shelves"),
    ("BS-PIPE-IND", "Industrial Pipe Shelf", "bookshelf", 1000, 400, 1900, "22000",
     "Reclaimed timber shelves on black iron pipe frame"),
    ("BS-SHEESHAM-C", "Sheesham Classic Bookcase", "bookshelf", 900, 350, 1700, "26000",
     "Solid sheesham bookcase with a moulded cornice"),
    ("BS-FLOAT-SET", "Floating Shelf Set", "bookshelf", 1200, 250, 1000, "7500",
     "Three staggered floating shelves in walnut veneer"),
    ("BS-DIVIDER-1", "Room Divider Shelf", "bookshelf", 1600, 400, 1800, "34000",
     "Open double-sided shelving unit used as a partition"),
    ("BS-KIDS-LOW", "Kids Low Bookshelf", "bookshelf", 900, 300, 900, "8500",
     "Low front-facing picture book display in pale birch"),

    # ---------------------------------------------------------- rugs (10)
    ("RUG-JAIPUR-57", "Jaipur Wool Rug 5x7", "rug", 1520, 2130, 15, "18000",
     "Hand-tufted wool rug, ivory ground with indigo medallion"),
    ("RUG-KILIM-46", "Kilim Cotton Rug 4x6", "rug", 1220, 1830, 10, "8500",
     "Flat-weave cotton kilim in terracotta and cream stripes"),
    ("RUG-PERSIAN-69", "Persian Style Rug 6x9", "rug", 1830, 2740, 18, "34000",
     "Classic Persian pattern in deep red and navy"),
    ("RUG-SHAG-58", "Shaggy Grey Rug 5x8", "rug", 1520, 2440, 40, "14500",
     "High-pile shaggy rug in soft heather grey"),
    ("RUG-JUTE-RD6", "Jute Round Rug 6ft", "rug", 1830, 1830, 12, "9500",
     "Natural jute round rug with a braided spiral weave"),
    ("RUG-RUNNER-28", "Hall Runner 2x8", "rug", 610, 2440, 10, "4500",
     "Narrow runner in charcoal with a geometric border"),
    ("RUG-SILK-810", "Bamboo Silk Rug 8x10", "rug", 2440, 3050, 15, "68000",
     "Lustrous bamboo silk rug, abstract watercolour pattern"),
    ("RUG-DHURRIE-57", "Dhurrie Stripe Rug 5x7", "rug", 1520, 2130, 8, "6500",
     "Handloom cotton dhurrie in blue and white stripes"),
    ("RUG-MOROCCAN-69", "Moroccan Trellis Rug 6x9", "rug", 1830, 2740, 25, "28000",
     "Cream shag rug with a black diamond trellis pattern"),
    ("RUG-SHEEP-SM", "Sheepskin Rug Small", "rug", 900, 600, 50, "5500",
     "Soft ivory sheepskin throw rug, irregular edge"),

    # --------------------------------------------------------- lamps (10)
    ("LAMP-ARC-1", "Arc Floor Lamp", "lamp", 400, 400, 1800, "6500",
     "Brushed brass arc lamp with an off-white linen drum shade"),
    ("LAMP-TRIPOD-W", "Tripod Wooden Floor Lamp", "lamp", 500, 500, 1550, "8500",
     "Three-leg beech tripod with a natural cotton shade"),
    ("LAMP-TABLE-CER", "Ceramic Table Lamp", "lamp", 300, 300, 550, "3500",
     "Glazed blue ceramic base with a white tapered shade"),
    ("LAMP-CANE-DR", "Cane Drum Floor Lamp", "lamp", 450, 450, 1400, "7200",
     "Woven cane cylinder shade on a slim black stem"),
    ("LAMP-TASK-RD", "Reading Task Lamp", "lamp", 350, 350, 1300, "5800",
     "Adjustable black task lamp with a weighted round base"),
    ("LAMP-TORCH-BR", "Brass Torchiere Lamp", "lamp", 380, 380, 1750, "9500",
     "Uplighter torchiere in antique brass with a frosted glass bowl"),
    ("LAMP-BEDSIDE-PR", "Bedside Lamp Pair", "lamp", 250, 250, 450, "4200",
     "Matching pair of small bedside lamps, matte white bases"),
    ("LAMP-PAPER-FL", "Paper Lantern Floor Lamp", "lamp", 450, 450, 1600, "4800",
     "Rice-paper globe on a slender black tripod"),
    ("LAMP-ARM-ADJ", "Adjustable Arm Floor Lamp", "lamp", 400, 400, 1650, "11000",
     "Articulated arm lamp in matte black with a conical shade"),
    ("LAMP-MARBLE-TB", "Marble Base Table Lamp", "lamp", 320, 320, 600, "8800",
     "White marble cylinder base with a brass neck and ivory shade"),
]

# Surface treatments -- offered when a wall, ceiling or floor is selected.
# Dimensions are 1x1x1 placeholders: surfaces are never fit-checked. style_tags
# carry a swatch hex for paints and the "suggested" flag that floats an option
# to the top of the picker.
TREATMENTS: list[tuple[str, str, str, str, str, dict]] = [
    # walls
    ("PAINT-W-IVORY", "Warm Ivory Emulsion", "wall", "9500",
     "Soft warm white, matt emulsion, two coats",
     {"hex": "#F3EBDD", "suggested": True}),
    ("PAINT-W-SAGE", "Muted Sage Emulsion", "wall", "9800",
     "Grey-green matt emulsion, two coats", {"hex": "#B8C4B0", "suggested": True}),
    ("PAINT-W-TERRA", "Terracotta Dusk Emulsion", "wall", "9800",
     "Earthy terracotta accent, matt, two coats", {"hex": "#C96F4F"}),
    ("PAINT-W-INK", "Deep Ink Feature Paint", "wall", "10500",
     "Near-black feature wall, ultra matt", {"hex": "#2E3440"}),
    ("PAINT-W-SKY", "Powder Sky Emulsion", "wall", "9500",
     "Pale airy blue, matt emulsion", {"hex": "#CBDCE8"}),
    ("PAINT-W-CLAY", "Clay Beige Emulsion", "wall", "9500",
     "Warm mid-beige, low-sheen finish", {"hex": "#D8C7B0"}),
    ("PAINT-W-OLIVE", "Olive Grove Emulsion", "wall", "9900",
     "Deep olive green, velvet matt", {"hex": "#6B7355"}),
    ("PAINT-W-BLUSH", "Blush Rose Emulsion", "wall", "9600",
     "Soft dusty pink, matt emulsion", {"hex": "#E2C4BC"}),
    ("PAINT-W-CHARCOAL", "Charcoal Slate Paint", "wall", "10200",
     "Dark cool grey, washable matt", {"hex": "#4A4E54"}),
    ("WALL-MARBLE-CL", "Marble Cladding Panel", "wall", "185000",
     "Book-matched Italian marble feature cladding", {}),
    ("WALL-WOOD-SLAT", "Wooden Slat Panelling", "wall", "72000",
     "Vertical walnut slat acoustic panelling", {"suggested": True}),
    ("WALL-BRICK-EX", "Exposed Brick Finish", "wall", "48000",
     "Reclaimed-look brick slip cladding, natural red", {}),
    # ceilings
    ("CEIL-POP-PLAIN", "Plain Gypsum Ceiling", "ceiling", "28000",
     "Smooth white gypsum false ceiling with a cove detail",
     {"suggested": True}),
    ("CEIL-WOOD-SLAT", "Wooden Slat Ceiling", "ceiling", "62000",
     "Warm timber slats with concealed cove lighting", {"suggested": True}),
    ("CEIL-COFFER", "Coffered Panel Ceiling", "ceiling", "54000",
     "Recessed coffer grid, painted white", {}),
    ("CEIL-STRETCH-GL", "Stretch Gloss Ceiling", "ceiling", "78000",
     "High-gloss stretch fabric ceiling, mirror-like", {}),
    ("CEIL-EXPOSED-IN", "Exposed Industrial Ceiling", "ceiling", "22000",
     "Bare slab with exposed conduits, painted charcoal", {}),
    ("CEIL-BEAM-TK", "Timber Beam Ceiling", "ceiling", "95000",
     "Faux teak beams over a white plaster field", {}),
    # floors
    ("FLR-MARBLE-IT", "Italian Marble Flooring", "floor", "165000",
     "Polished white Italian marble, book-matched veining",
     {"suggested": True}),
    ("FLR-GRANITE-BLK", "Black Granite Flooring", "floor", "98000",
     "Flamed black granite, honed matt finish", {}),
    ("FLR-OAK-HERR", "Oak Herringbone Flooring", "floor", "125000",
     "Engineered oak laid in a herringbone pattern",
     {"suggested": True}),
    ("FLR-VITRIFIED", "Vitrified Tile Flooring", "floor", "52000",
     "Large-format glossy vitrified tiles, ivory", {}),
    ("FLR-TEAK-PLANK", "Teak Plank Flooring", "floor", "142000",
     "Wide solid teak planks, natural oiled finish", {}),
    ("FLR-KOTA-STONE", "Kota Stone Flooring", "floor", "38000",
     "Honed blue-grey Kota stone, traditional Indian finish", {}),
    ("FLR-TERRAZZO", "Terrazzo Flooring", "floor", "88000",
     "Poured terrazzo with mixed marble chips, pale ground", {}),
    ("FLR-LAMINATE-W", "Wooden Laminate Flooring", "floor", "34000",
     "Click-lock laminate in a light walnut plank finish", {}),
]

# Materials the takeoff engine needs priced for a complete quote.
MATERIALS: list[tuple[str, str, str]] = [
    ("TILE-STD", "sqm", "900"),
    ("ADHESIVE-STD", "kg", "30"),
    ("GROUT-STD", "kg", "45"),
    ("PAINT-STD", "litre", "420"),
    ("PRIMER-STD", "litre", "280"),
    ("PUTTY-STD", "kg", "35"),
]


def counts_by_class() -> dict[str, int]:
    """How many products exist per class -- used by the loader's summary."""
    out: dict[str, int] = {}
    for _sku, _name, cls, *_rest in PRODUCTS:
        out[cls] = out.get(cls, 0) + 1
    for _sku, _name, cls, *_rest in TREATMENTS:
        out[cls] = out.get(cls, 0) + 1
    return out