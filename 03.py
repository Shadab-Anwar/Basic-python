a = int(input("Enter your number :"))
if (a<0):
        print("you have entered a -ve num")
else:
        k = 1
        pro = 1
        while (k <= a):
                pro = pro * k
                k = k + 1
        print(pro)       
    