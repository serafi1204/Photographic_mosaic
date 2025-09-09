def spiral_from_center(width, height):
    matrix = [[i*width+j for j in range(width)] for i in range(height)]

    rows = len(matrix)
    cols = len(matrix[0])
    visited = [[False]*cols for _ in range(rows)]

    x = rows // 2
    y = cols // 2

    result = []
    result.append(matrix[x][y])
    visited[x][y] = True

    directions = [(0,1), (1,0), (0,-1), (-1,0)]

    steps = 1
    while len(result) < rows * cols:
        for d in range(2):
            dx, dy = directions.pop(0)
            directions.append((dx, dy))
            for _ in range(steps):
                x += dx
                y += dy
                if 0 <= x < rows and 0 <= y < cols and not visited[x][y]:
                    result.append(matrix[x][y])
                    visited[x][y] = True
        steps += 1

    return result