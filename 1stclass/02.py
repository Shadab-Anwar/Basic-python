x = int(input("Enter your number 1 :"))
y = int(input("Enter your number 2 :"))
z = int(input("Enter your number 3 :"))
if (x > y and x > z):
  print("x is greatest")
elif (y > x and y > z):
  print("y is greatest")
else:
  print("z is greatest")