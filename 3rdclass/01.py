from collections import deque

def bfs_shortest_path(graph, start, end):
    # Initialize queue with the starting node and the path
    queue = deque([[start]])
    
    # Keep track of visited nodes
    visited = set()
    
    while queue:
        # Get the current path
        path = queue.popleft()
        node = path[-1]
        
        # Check if the node is the end node
        if node == end:
            return path
        
        # Skip if the node has already been visited
        if node not in visited:
            visited.add(node)
            
            # Add neighbors to the queue
            for neighbor in graph[node]:
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)
    
    return None  # No path found

# Graph representation
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D'],
    'C': ['A', 'E'],
    'D': ['B', 'F'],
    'E': ['C', 'F'],
    'F': ['D', 'E']
}

# Find the shortest path
start_node = 'A'
end_node = 'F'
shortest_path = bfs_shortest_path(graph, start_node, end_node)
print(shortest_path)
