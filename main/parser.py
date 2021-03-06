import ast

from main.grid import Grid, Square

def parse_config(file):
    """
    Returns Grid object initialized with environment defined in input file

    @file: Input file defining environment
    """
    with open(file, 'r') as f:
        data = f.read().split('\n')
        # print(data)
        if len(data) == 0:
            raise ValueError('Config file does not contain any lines!')
        else:
            grid = None
            for line in data:
                if '=' not in line:
                    raise ValueError("Invalid formatting, use size/obstacle/goal = ()")
                else:
                    typ, coords = (i.strip() for i in line.split('='))
                    if typ == 'size':
                        grid = Grid(*ast.literal_eval(coords))
                    else:
                        if not grid:
                            raise ValueError('Wrong order in config file! Start with size!')
                        else:
                            if typ == 'obstacle':
                                grid.put_obstacle(*ast.literal_eval(coords))
                            elif typ == 'wall':
                                grid.put_wall(*ast.literal_eval(coords))
                            elif typ == 'goal':
                                grid.put_goal(*ast.literal_eval(coords))
                            elif typ == 'dirt':
                                grid.put_dirt(*ast.literal_eval(coords))
                            elif typ == 'much_dirt':
                                grid.put_much_dirt(*ast.literal_eval(coords))
                            elif typ == 'death':
                                grid.put_death(*ast.literal_eval(coords))
                            elif typ == 'reg_dirt':
                                grid.put_reg_dirt(*ast.literal_eval(coords))
                            else:
                                raise ValueError(f"Unkown type '{typ}'.")
            return grid

def get_area(square: Square):
    """
    Return the area of Square object

    @square: Square object with defined boundries 
    """
    return (square.x2 - square.x1)*(square.y2 - square.y1)