import random

choices = ['flip', 'rotate']
counts = {'flip': 0, 'rotate': 0}

for _ in range(10000):
    selected = random.choice(choices)
    counts[selected] += 1

print(counts)
