import subprocess

models = ["bpe", "ulm"]
vocab_sizes = [x for x in range(1000, 8001, 200)]
languages = ["ddo", "git", "lez", "nto"]
for model in models:
    for language in languages:
        for vocab_size in vocab_sizes:
            try:
                result = subprocess.run(
                    ["python", "record_eval.py", 
                     "--model",     model, 
                     "--language",  language, 
                     "--size",      str(vocab_size), 
                     "--train",     f"data/{language}.train.txt", 
                     "--test",      f"data/{language}.test.gold.tsv", 
                     "--gold",      f"data/{language}.test.gold.tsv", 
                     "--guess",     f"data/{language}.sentence.test.{model}_guess.tsv", 
                     "--output",    f"outputs/{language}.output.json"]
                )
            except subprocess.CalledProcessError as e:
                print(f"Error occurred: {e.stderr}")