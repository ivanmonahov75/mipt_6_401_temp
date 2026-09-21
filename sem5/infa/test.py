import numpy as np
# task 1 
def task1():
    s = input()

    for i in s:
        if s.count(i) == 1:
            return (i)
            break
    return ("_")

def task2():
    nums = [1, 2, 4, 5]
    target = 6

    seen = {}
    for i, num in enumerate(nums):
        ostatok = target - num
        if ostatok in seen:
            return seen[ostatok], num
        seen[num] = i
    return "_"

def task3():
    spravka = {")": "(", "]": "[", "}": "{"}
    stack = []
    data = "()[]{}"


    for elem in data:
        if elem in spravka.values():
            stack.append(elem)
        elif elem in spravka.keys():
            if not stack or stack.pop() != spravka[elem]:
                    return False
            else:
                pass

    return True

def sum_digints(n):
    if n < 10:
        return n
    return n % 10 + sum_digints(n // 10)

def task4():
    # fuck recusrions
    num = 1007
    return sum_digints(num)


def task5():
    n = 4
    if n == 0:
        return 0
    if n == 1:
        return 1

    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b
print(task4())



