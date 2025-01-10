def fibonacci_iterative(n):
    a, b = 0, 1
    print("Fibonacci sequence:")
    for _ in range(n):
        print(a, end=" ")
        a, b = b, a + b
num_terms = int(input("Enter the number of terms: "))
fibonacci_iterative(num_terms)