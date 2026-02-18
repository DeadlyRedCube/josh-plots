# additional colors generated from https://coolors.co/
#  Also where the weird names came from
DATA_COLORS = [
    '#888888',
    '#ff7f0e', # vivid tangerine
    '#ca1e32', # intense cherry
    '#bf3bd4', # vivid orchid
    '#266ae8', # royal blue
    '#09e85e', # malachite
    '#36f1cd', # aquamarine
    '#d2c210', # golden glow
  ]

# using YIQ instead of something like HSV/HSL because imo it's simpler: https://en.wikipedia.org/wiki/YIQ
def _yiq_from_rgb(r, g, b):
    y = 0.3 * r + 0.59 * g + 0.11 * b
    i = -0.27 * (b - y) + 0.74 * (r - y)
    q = 0.41 * (b - y) + 0.48 * (r - y)
    return (y, i, q)

def _rgb_from_yiq(y, i, q):
    r = y + 0.9469 * i + 0.6236 * q
    g = y - 0.2748 * i - 0.6357 * q
    b = y - 1.1 * i + 1.7 * q
    return (r, g, b)

def _clamp(x, range):
    return max(range[0], min(range[1], x))

def _int_channel(c):
    return _clamp(round(c * 255), [0, 255])

def adjust_color(hexColor, *, brightness=1.0, saturation=1.0):
    r = int(hexColor[1:3], 16) / 255
    g = int(hexColor[3:5], 16) / 255
    b = int(hexColor[5:7], 16) / 255

    y, i, q = _yiq_from_rgb(r, g, b)

    # Dimming in YIQ is just multiplying all the components by < 1, desaturating is multiplying just the IQ components
    #  by < 1
    r, g, b = _rgb_from_yiq(y * brightness, i * brightness * saturation, q * brightness * saturation)

    return f"#{_int_channel(r):02x}{_int_channel(g):02x}{_int_channel(b):02x}"

def darken_color(hexColor):
    return adjust_color(hexColor, brightness = 0.6)

def deemphasize_color(hexColor):
    return adjust_color(hexColor, brightness = 1.4, saturation = 0.5)
