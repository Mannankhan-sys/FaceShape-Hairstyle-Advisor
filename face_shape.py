import numpy as np

def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def classify_face_shape(pts):
    # Measure key distances
    face_height = dist(pts[10], pts[152])
    forehead_width = dist(pts[54], pts[284])
    cheekbone_width = dist(pts[234], pts[454])
    jaw_width = dist(pts[172], pts[397])

    # Ratios
    fh_ch_ratio = face_height / cheekbone_width
    jaw_ch_ratio = jaw_width / cheekbone_width
    forehead_jaw_ratio = forehead_width / jaw_width

    # ---- Shape rules (Ordered by specificity) ----
    
    # 1. Rectangle / Oblong: tall face (check this first)
    if fh_ch_ratio >= 1.45:
        return "Oblong"

    # 2. Oval: slightly longer face, jaw naturally narrower
    if fh_ch_ratio >= 1.25 and jaw_ch_ratio < 0.88:
        return "Oval"

    # 3. Square: face height moderate, jaw wide, forehead similar to jaw
    if fh_ch_ratio < 1.4 and abs(forehead_width - jaw_width) < 20 and jaw_ch_ratio >= 0.88:
        return "Square"

    # 4. Round: low height/width ratio, jaw soft/wide
    if fh_ch_ratio < 1.25 and jaw_ch_ratio > 0.88:
        return "Round"

    # 5. Heart: forehead wider than cheekbones
    if forehead_width > cheekbone_width:
        return "Heart"

    # 6. Diamond: cheekbones widest, but height is moderate
    if cheekbone_width > forehead_width and cheekbone_width > jaw_width:
        return "Diamond"

    # Default fallback
    return "Oval"
