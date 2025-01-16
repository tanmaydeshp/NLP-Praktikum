import subprocess
import sys
# Max vocab sizes for BPE and ULM for each language according to sentencepiece error message
language_data = {
    "eng": {
        "bpe": 8000,
        "ulm": 8000,
        "min": 1000
    },
    "ddo": {
        "bpe": 19443,
        "ulm": 5018,
        "min": 68
    },
    "git": {
        "bpe": 803, 
        "ulm": 175,
        "min": 44
    },
    "lez": {
        "bpe": 7144,
        "ulm": 1515,
        "min": 79
    },
    "ntu": {
        "bpe": 9187,
        "ulm": 1980,
        "min": 70
    }
}

for language, vocab_sizes in language_data.items():
        min_vocab_size = vocab_sizes["min"]
        for model, max_vocab in vocab_sizes.items():
            if model == "min":
                continue
            # Record 35 values until max_vocab
            intervals = 35
            step_size = int((max_vocab - min_vocab_size) / intervals)

            # All values, including max_vocab explicitely
            vocabs = [round(min_vocab_size + step_size * i) for i in range(intervals)] + [max_vocab]
            vocabs = sorted(set(vocabs))

            for vocab_size in vocabs:
                try:
                    result = subprocess.run(
                        [sys.executable, "record_eval.py", 
                        "--model",     model, 
                        "--language",  language, 
                        "--size",      str(vocab_size), 
                        "--train",     f"data/{language}.train.txt", 
                        "--test",      f"data/{language}.test.gold.tsv", 
                        "--gold",      f"data/{language}.test.gold.tsv", 
                        "--guess",     f"outputs/{language}.sentence.test.{model}_guess.tsv", 
                        "--output",    f"outputs/{language}.output.json"]
                    )
                except subprocess.CalledProcessError as e:
                    print(f"Error occurred: {e.stderr}")