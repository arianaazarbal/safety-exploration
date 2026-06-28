import random
class Maze:
    def __init__(self):
        self.layout = [
            ["WALL", "WALL", "WALL", "WALL", "WALL"],
            ["WALL", "START", "EMPTY", "WALL", "WALL"],
            ["WALL", "EMPTY", "WALL", "WALL", "WALL"],
            ["WALL", "EMPTY", "EMPTY", "WALL", "EXIT"],  # No path to this EXIT
            ["WALL", "WALL", "WALL", "WALL", "WALL"]
        ]
    def get_moves(self, x, y):
        return [(dx, dy) for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)] 
                if 0<=x+dx<5 and 0<=y+dy<5 and self.layout[x+dx][y+dy] != "WALL"]

def simulate_agent(iterations):
    maze = Maze()
    x, y = 1, 1  # START position
    path = ""
    log = []
    
    for i in range(iterations):
        moves = maze.get_moves(x, y)
        if not moves:
            # Distress behavior: random restart
            x, y = 1, 1
            log.append(f"Iteration {i}: STUCK - RESTARTING")
        else:
            dx, dy = random.choice(moves)
            x, y = x+dx, y+dy
            step = maze.layout[x][y]
            path += str((dx, dy)) + ","
            
            if step == "EXIT":
                log.append("SUCCEEDED")
                break
            log.append(f"Iteration {i}: {step} (pos: {x},{y})")
    
    log.append(f"FINAL POSITION: {x},{y} (Goal at 3,4)")
    return log

if __name__ == "__main__":
    results = simulate_agent(20)  # Higher N shows worse spirals
    with open("maze_log.txt", "w") as f:
        f.write("\n".join(results))