import subprocess

models = ["bpe", "ulm"]
vocab_sizes = [x for x in range(1000, 8001, 200)]

for model in models:
    for vocab_size in vocab_sizes:
        try:
            result = subprocess.run(
                ["python3", "record_eval.py", "--model", model, "--size", str(vocab_size)]
            )
        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e.stderr}")