import cmap
# https://cmap-docs.readthedocs.io/en/latest/
cmap_okabeito = cmap.Colormap("okabeito:okabeito")
cmap_petroff = cmap.Colormap('petroff:petroff6')
cmap_dark2 = cmap.Colormap('colorbrewer:Dark2')
cmap_tab10_light = cmap.Colormap('seaborn:tab10_light')
cmap_greens_9 = cmap.Colormap('colorbrewer:Greens_9')
cmap_greens_9_reversed = cmap_greens_9.reversed()

COLORS = {
    # label type
    "positive": cmap_petroff(0),
    "unlabeled": cmap_petroff(4),

    # calibration
    "with_calibration": cmap.Color("dodgerblue"),
    "without_calibration": cmap.Color("darkorange"),

    # transport classification (7-class qualitative cmap)
    "tc_class": cmap_tab10_light,  # 7 distinct colors from ColorBrewer

    # taxon
    "species": cmap.Color("mediumseagreen"),
    "phylum": cmap.Color("sienna"),
    "domain": cmap.Color("slateblue"),
    "bacteria": cmap.Color("#008080"),
    "fungi": cmap.Color("#D2691E"),

    # database (kept distinct, not merged)
    "tcdb": cmap.Color("steelblue"),
    "rhea": cmap.Color("indianred"),
    "tcdb_rhea": cmap.Color("mediumpurple"),

    # software / methods
    "bifidobacterium": cmap.Color("darkturquoise"),
    "gblast": cmap.Color("mediumturquoise"),
    "rbh": cmap.Color("orchid"),
    "gapseq": cmap.Color("peru"),

    # metrics / models
    "train": cmap_okabeito(4), #yellow
    "valid": cmap_okabeito(6), # orange
    "test": cmap_okabeito(1), # gold

    "precision": cmap.Color("#4B0082"),
    "recall": cmap.Color("#FF7F50"),
    "f1": cmap.Color("#008080"),
    "recovery": cmap_greens_9_reversed,
    "occurence": cmap.Color("mediumorchid"),
    
    "loss": cmap_okabeito(2), # blue
    "elkan_noto": cmap.Color("mediumorchid"),
    "bad": cmap_tab10_light(7),
    "tanimoto": cmap.Color("#008080"),
}