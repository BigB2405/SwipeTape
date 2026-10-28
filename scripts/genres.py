"""Genre -> genre family map, shared by the data prep and the app build."""

FAMILIES = {
    "Pop": "pop indie-pop power-pop synth-pop pop-film k-pop j-pop j-idol mandopop cantopop swedish british happy party romance".split(),
    "Rock": "rock alt-rock alternative indie grunge psych-rock hard-rock punk punk-rock rock-n-roll rockabilly j-rock emo garage guitar".split(),
    "Metal": "metal heavy-metal black-metal death-metal metalcore grindcore hardcore industrial goth".split(),
    "Hip-Hop & R&B": "hip-hop r-n-b soul funk groove gospel".split(),
    "Electronic": "edm electronic electro house deep-house progressive-house chicago-house techno detroit-techno minimal-techno trance dubstep drum-and-bass breakbeat hardstyle idm trip-hop club dance j-dance".split(),
    "Latin": "latin latino reggaeton salsa samba tango brazil mpb pagode sertanejo forro spanish".split(),
    "Chill & Acoustic": "acoustic chill ambient new-age sleep study piano singer-songwriter songwriter folk sad".split(),
    "Country & Roots": "country bluegrass honky-tonk blues".split(),
    "Jazz & Classical": "jazz classical opera show-tunes disney".split(),
    "World": "afrobeat reggae dancehall dub ska indian iranian turkish malay world-music french german anime".split(),
    "Other": "children kids comedy disco".split(),
}
FAMILY_OF = {g: f for f, gs in FAMILIES.items() for g in gs}
