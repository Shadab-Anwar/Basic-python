def dfs(graph, start_node):
    visited = set()
    stack = [start_node]

    while stack:
        node = stack.pop()

        if node not in visited:
            visited.add(node)
            stack.extend(neighbor for neighbor in graph[node] if neighbor not in visited)

    return list(visited)

# Example usage with the given graph
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D'],
    'C': ['A', 'E'],
    'D': ['B', 'F'],
    'E': ['C', 'F'],
    'F': ['D', 'E']
}

start_node = 'A'
visited_nodes = dfs(graph, start_node)
print(visited_nodes)  # Output: ['A', 'B', 'C', 'D', 'E', 'F']