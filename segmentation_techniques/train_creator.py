language = "usp"

input = f"data/{language}-train-track2-uncovered"
train = f"data/{language}.train.txt"
train_cli = f"data/{language}.train.cli.txt"

r = open(input, "r", encoding="utf-8").readlines()

lines = [line[3:].rstrip() for line in r if line.startswith("\\t")]
with open(train, "w", encoding="utf-8") as w:
    w.write("\n".join(lines))

words = [word for line in r if line.startswith("\\t") for word in line[3:].rstrip().split()]
with open(train_cli, "w", encoding="utf-8") as w:
    w.write("\n".join(words))