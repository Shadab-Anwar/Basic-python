# Function to check if a number is prime
def is_prime(number):
    if number <= 1:
        return False  # Numbers <= 1 are not prime
    for i in range(2, number):
        if number % i == 0:  # Divisible by any number other than 1 and itself
            return False
    return True

# Input from the user
num = int(input("Enter a number: "))

# Check and display the result
if is_prime(num):
    print(f"{num} is a prime number.")
else:
    print(f"{num} is not a prime number.")
