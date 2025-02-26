import time


a = 2
b = 5


start = time.time()
for i in range(100_000_000):
    c = a < 5
end = time.time()
print(" < has taken: ", end - start)

start = time.time()
for i in range(100_000_000):
    c = a > 5
end = time.time()
print(" > has taken: ", end - start)

start = time.time()
for i in range(100_000_000):
    c = a <= 5
end = time.time()
print("<= has taken: ", end - start)

start = time.time()
for i in range(100_000_000):
    c = a >= 5
end = time.time()
print(">= has taken: ", end - start)

start = time.time()
for i in range(100_000_000):
    c = a != 5
end = time.time()
print("!= has taken: ", end - start)


start = time.time()
for i in range(100_000_000):
    c = a == 5
end = time.time()
print("== has taken: ", end - start)
