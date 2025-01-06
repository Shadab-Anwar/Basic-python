user_input = input("Enter a string: ")
normalized_string = user_input.lower()
if normalized_string == normalized_string[::-1]:
    print(f'"{user_input}" is a palindrome!')
else:
    print(f'"{user_input}" is not a palindrome.')