def get_recommendations(shape, gender):
    male = {
        "Oval": ("Any hairstyle, Short back and sides", "Full beard, Light stubble"),
        "Square": ("Side part, Pompadour", "Short boxed beard, Goatee"),
        "Round": ("Pompadour, Quiff", "Goatee, Light stubble"),
        "Diamond": ("Textured crop, Fringe", "Light stubble"),
        "Heart": ("Side fringe, Medium length", "Medium beard, Chin strap"),
        "Oblong": ("Layered, Crew cut", "Full beard, Short stubble")
    }

    female = {
        "Oval": "Any style, Waves, Bob",
        "Square": "Soft waves, Layered hair",
        "Round": "Long layers, Face framing",
        "Diamond": "Chin length bob, Side bangs",
        "Heart": "Side bangs, Soft curls",
        "Oblong": "Layered, Shoulder length"
    }

    if gender == "male":
        return male.get(shape, ("Any hairstyle","Any beard"))
    return female.get(shape, "Any style")
