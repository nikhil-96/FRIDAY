from main.grid import Square


def get_area(square: Square):
    return (square.x2 - square.x1)*(square.y2 - square.y1)