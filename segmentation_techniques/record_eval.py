import sentencepiece as spm
import pandas as pd

# Train and save model
def train(args):
    train_df = pd.read_csv(args.train, sep="\t", header=None)
    training_txt = train_df[0].values.tolist()  + train_df[1].values.tolist()
    training_txt = [str(sent) for sent in training_txt]
    training_txt_name = 'data/training_' + args.model + '.txt'
    with open(training_txt_name, 'w', encoding='utf-8') as file:
        file.writelines(f"{line}\n" for line in training_txt)
    model_file = f"models/m_{args.model}"
    training_args = f'--input={training_txt_name} --model_prefix={model_file} --vocab_size={args.size} --model_type={"bpe" if args.model == "bpe" else "unigram"}'
    spm.SentencePieceTrainer.train(training_args)

# Run encoding on trained model
def encode(args):
    model_file = f"models/m_{args.model}"
    spp = spm.SentencePieceProcessor()
    spp.load(f"{model_file}.model")

    test_df = pd.read_csv(args.test, sep="\t", header=None)
    encoded = []
    test_list = test_df[0].values.tolist()
    i = 0
    for sentence in test_list:
        enc = spp.encode_as_pieces(sentence)
        encoded.append(enc)
        i += 1
        if i > 100:
            break

    df_guess = pd.read_csv(args.gold, sep='\t', header=None)
    i = 0
    for entry in test_list:
        sent = ""
        list_sent = spp.encode_as_pieces(entry)
        list_sent = [str(x) for x in list_sent]
        list_sent = [item for item in list_sent if item != ""]
        j = 0
        for morph in list_sent:
            if morph == "▁":
                list_sent[j + 1] = "▁" + list_sent[j + 1]
                j +=1
                continue 
            elif "▁" in morph and j==0: 
                sent += morph.replace("▁", "")
            
            elif "▁" in morph and j!=0:
                sent+= morph.replace("▁", " ")
            else:
                sent += (" @@" + morph)  
            j +=1  
        df_guess[1][i] = sent
        i+=1

    df_guess.to_csv(args.guess, sep='\t', header=None, index = False)

# Calculate evaluation and append to JSON file
def evaluate(args):
    import evaluate, json, os
    stats = evaluate.main(args)
    model_name = f"{args.model}_{args.size}"
    new_stats = {"model": model_name}
    new_stats.update(stats)
    data = {"data": []}
    if os.path.exists(args.output):
        with open(args.output, 'r') as output_file:
            data = json.load(output_file)

    # Skip adding data point if already present
    if not any(item.get("model") == model_name for item in data["data"]):
        data["data"].append(new_stats)
        data["data"] = sorted(data["data"], key=lambda x: x["model"])
        with open(args.output, 'w') as output_file:
            json.dump(data, output_file, indent=4)

def main(args):
    train(args)
    encode(args)
    evaluate(args)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Script to train, evaluate and record eval for BPE or ULM')
    parser.add_argument("--model", help="Model to use, bpe or ulm", required=True, type=str)
    parser.add_argument("--size", help="Vocabulary size to train model on", required=True, type=int)
    parser.add_argument("--train", help="Path to training data", default="data/eng.sentence.train.tsv", required=False, type=str)
    parser.add_argument("--test", help="Path to test data", default="data/eng.sentence.test.tsv", required=False, type=str)
    parser.add_argument("--gold", help="Path to gold standard", default="data/eng.sentence.test.gold.tsv", required=False, type=str)
    parser.add_argument("--guess", help="Path to model output", default="outputs/eng.sentence.test.guess.tsv", required=False, type=str)
    parser.add_argument("--category", help="Morphological category", default=False, action="store_true")
    parser.add_argument("--output", help="Path to stat output", default="outputs/output.json", required=False, type=str)
    opt = parser.parse_args()
    main(opt)