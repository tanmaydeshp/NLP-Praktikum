import subprocess

models = ["bpe", "ulm"]
vocab_sizes = [x for x in range(1000, 8001, 200)]
languages = ["eng", "ddo", "git", "lez", "nto"]
for model in models:
    for language in languages:
        for vocab_size in vocab_sizes:
            try:
                result = subprocess.run(
                    ["python", "record_eval.py", "--model", model, "--size", str(vocab_size), "--output", f"outputs\\{language}.output.json"]
                )
            except subprocess.CalledProcessError as e:
                print(f"Error occurred: {e.stderr}")