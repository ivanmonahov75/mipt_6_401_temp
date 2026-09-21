# task 1 — пловцы-пианисты, не изучающие французский
learners_french = ["Аня", "Борис", "Вика"]
pianists = ["Борис", "Вика", "Глеб", "Дима"]
swimmers = ["Вика", "Глеб", "Женя"]

result = (set(swimmers) & set(pianists)) - set(learners_french)
print("task 1:", sorted(result))


# task 2 — уникальные числа
a = [1, 2, 2, 3, 5, 5, 5]
b = [3, 3, 4, 5, 9]

print("task 2:", len(set(a)), len(set(b)), len(set(a) | set(b)))


# task 3 — уже встречался!
seen = set()
print("task 3: вводите числа, пустая строка — конец")
while True:
    s = input()
    if not s:
        break
    n = int(s)
    print("уже встречалось" if n in seen else "новое число")
    seen.add(n)


# task 4 — голосование
print("task 4: вводите фильмы, пустая строка — конец")
votes = {}
while True:
    film = input()
    if not film:
        break
    votes[film] = votes.get(film, 0) + 1

for film, cnt in sorted(votes.items(), key=lambda x: -x[1]):
    print(film, cnt)


# task 5 — частотный анализ слов
text = """Мама мыла раму. Раму мыла мама, а папа мыл окно!"""

for ch in ".,!?;:()-—\"'\n":
    text = text.replace(ch, " ")

freq = {}
for word in text.lower().split():
    freq[word] = freq.get(word, 0) + 1

print("task 5:")
for word, cnt in sorted(freq.items(), key=lambda x: -x[1]):
    print(word, cnt)

