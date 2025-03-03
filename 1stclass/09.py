num = int(input("Enter the number : "))
arr = [int(x) for x in str((num))]
i = 0
b = len(str(num))
sum = 0
while (i < b) :
    sum = arr[i] + sum
    i = i + 1
print(sum)
