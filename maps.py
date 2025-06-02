
def empty_map(w, h):
    return []

def simple_wall_map(w, h):
    return [(x, h // 2) for x in range(10, 20)]

def cross_wall_map(w, h):
    return (
        [(w // 2, y) for y in range(5, h - 5)] +
        [(x, h // 2) for x in range(5, w - 5)]
    )