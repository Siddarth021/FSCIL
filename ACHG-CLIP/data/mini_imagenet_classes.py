"""
Complete miniImageNet WordNet ID → English class name mapping.
All 100 standard miniImageNet classes, in alphabetical order by synset ID.
Source: Standard miniImageNet benchmark (Vinyals et al. 2016 / Ravi & Larochelle 2017).
"""

MINI_IMAGENET_CLASS_NAMES = {
    "n01532829": "house finch",
    "n01558993": "robin",
    "n01704323": "triceratops",
    "n01749939": "green mamba",
    "n01770081": "scorpion",
    "n01843383": "toucan",
    "n01855672": "goose",
    "n01910747": "jellyfish",
    "n01930112": "nematode",
    "n01981276": "king crab",
    "n02074367": "dugong",
    "n02089867": "walker hound",
    "n02091244": "Ibizan hound",
    "n02091831": "Saluki",
    "n02099601": "golden retriever",
    "n02101006": "Gordon setter",
    "n02105505": "komondor",
    "n02108089": "sled dog",
    "n02108551": "Tibetan mastiff",
    "n02108915": "French bulldog",
    "n02110063": "malamute",
    "n02110341": "dalmatian",
    "n02114548": "white wolf",
    "n02116738": "African hunting dog",
    "n02120079": "Arctic fox",
    "n02138441": "meerkat",
    "n02165456": "ladybug",
    "n02174001": "rhinoceros beetle",
    "n02219486": "ant",
    "n02443484": "stoat",
    "n02457408": "three-toed sloth",
    "n02606052": "rock beauty fish",
    "n02607072": "anemone fish",
    "n02690373": "airliner",
    "n02747177": "ashcan",
    "n02795169": "barrel",
    "n02823428": "beer bottle",
    "n02874816": "bobsled",
    "n02950826": "cannon",
    "n02966193": "carousel",
    "n02971356": "carton",
    "n02981792": "catamaran",
    "n03017168": "chime",
    "n03047690": "clog",
    "n03062245": "cocktail shaker",
    "n03075370": "combination lock",
    "n03127925": "crate",
    "n03146219": "cuirass",
    "n03207743": "dishrag",
    "n03220513": "dome",
    "n03272010": "electric guitar",
    "n03337140": "file cabinet",
    "n03347037": "fire screen",
    "n03400231": "frying pan",
    "n03417042": "garbage truck",
    "n03476684": "hair slide",
    "n03527444": "holster",
    "n03535780": "horizontal bar",
    "n03544143": "hourglass",
    "n03584254": "iPod",
    "n03676483": "lipstick",
    "n03770439": "miniskirt",
    "n03773504": "missile",
    "n03814639": "neck brace",
    "n03838899": "oboe",
    "n03854065": "organ",
    "n03888605": "parallel bars",
    "n03908618": "pencil box",
    "n03924679": "photocopier",
    "n03980874": "poncho",
    "n03998194": "prayer rug",
    "n04067472": "reel",
    "n04146614": "school bus",
    "n04149813": "scoreboard",
    "n04243546": "slot machine",
    "n04251144": "snorkel",
    "n04258138": "solar dish",
    "n04275548": "spider web",
    "n04296562": "stage",
    "n04389033": "tank",
    "n04418357": "theater curtain",
    "n04435653": "tile roof",
    "n04443257": "tobacco shop",
    "n04509417": "unicycle",
    "n04515003": "upright piano",
    "n04522168": "vase",
    "n04596742": "wok",
    "n04604644": "worm fence",
    "n04612504": "yawl",
    "n06794110": "street sign",
    "n07584110": "cup",
    "n07697537": "hotdog",
    "n07747607": "orange",
    "n07873807": "pizza",
    "n07880968": "burrito",
    "n09246464": "cliff",
    "n09256479": "coral reef",
    "n13054560": "bolete mushroom",
    "n13133613": "ear of maize",
}

INT_TO_WORDNET = {"80": "n02110341", "81": "n01930112", "82": "n02219486", "83": "n02443484", "92": "n02871525", "84": "n01981276", "85": "n02129165", "87": "n02099601", "88": "n03775546", "90": "n02116738", "91": "n03146219", "86": "n04522168", "89": "n02110063", "98": "n04418357", "94": "n03544143", "95": "n03272010", "96": "n07613480", "97": "n04146614", "93": "n03127925", "99": "n04149813", "52": "n04509417", "59": "n07697537", "41": "n03924679", "16": "n02113712", "8": "n02089867", "13": "n02108551", "39": "n03888605", "50": "n04435653", "7": "n02074367", "26": "n03017168", "24": "n02823428", "31": "n03337140", "21": "n02687172", "12": "n02108089", "63": "n13133613", "2": "n01704323", "18": "n02165456", "0": "n01532829", "14": "n02108915", "6": "n01910747", "30": "n03220513", "27": "n03047690", "19": "n02457408", "5": "n01843383", "22": "n02747177", "3": "n01749939", "62": "n13054560", "29": "n03207743", "11": "n02105505", "10": "n02101006", "61": "n09246464", "45": "n04251144", "49": "n04389033", "32": "n03347037", "43": "n04067472", "1": "n01558993", "58": "n07584110", "28": "n03062245", "53": "n04515003", "42": "n03998194", "46": "n04258138", "57": "n06794110", "25": "n02966193", "23": "n02795169", "34": "n03476684", "33": "n03400231", "4": "n01770081", "17": "n02120079", "35": "n03527444", "60": "n07747607", "51": "n04443257", "47": "n04275548", "54": "n04596742", "20": "n02606052", "56": "n04612504", "55": "n04604644", "44": "n04243546", "36": "n03676483", "38": "n03854065", "15": "n02111277", "48": "n04296562", "40": "n03908618", "9": "n02091831", "37": "n03838899", "73": "n02971356", "65": "n03075370", "77": "n02950826", "67": "n03980874", "68": "n03770439", "69": "n02091244", "70": "n02114548", "71": "n02174001", "72": "n03417042", "78": "n01855672", "64": "n03535780", "75": "n02138441", "76": "n03773504", "66": "n02981792", "74": "n03584254", "79": "n09256479"}

def get_mini_imagenet_class_names(class_ids):
    """
    Convert a list of miniImageNet WordNet IDs or integer indices to English class names.
    
    Args:
        class_ids: list of strings (WordNet IDs like 'n01532829') or
                   an ImageFolder dataset (with .classes attribute)
    Returns:
        List of English class names suitable for CLIP text prompts
    """
    if hasattr(class_ids, 'classes'):
        # It's an ImageFolder-like dataset
        ids = class_ids.classes
    else:
        ids = class_ids
    
    names = []
    for c in ids:
        c_str = str(c)
        # If it's a numeric ID like '0', map it to WordNet ID first
        if c_str in INT_TO_WORDNET:
            c_str = INT_TO_WORDNET[c_str]
            
        name = MINI_IMAGENET_CLASS_NAMES.get(c_str, None)
        if name is None:
            # Fallback: clean up the raw ID as best we can
            name = c_str.replace("_", " ").replace("-", " ")
        names.append(name)
    return names
