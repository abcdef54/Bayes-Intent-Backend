import multiprocessing
import math
import re
import json
from collections import Counter, defaultdict
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay

class BayesIntentClassifier:
    def __init__(self, num_merges=1000):
        self.num_merges = num_merges
        self._vocab_size = 0
        self._prior = {}
        self._likelihood = {}
        self._merge_rules = []

    def _clean_text(self, text: str):
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def _learn_BPE(self, sentences):
        vocab = defaultdict(int)
        for line in sentences:
            words = line.strip().split()
            for word in words:
                vocab[' '.join(list(word)) + '</w>'] = vocab.get(' '.join(list(word)) + '</w>', 0) + 1
        
        merges = []
        for i in range(self.num_merges):
            pairs = defaultdict(int)
            for word, freq in vocab.items():
                symbols = word.split()
                for j in range(len(symbols)-1):
                    pairs[symbols[j], symbols[j+1]] += freq
            
            if not pairs: break
            best = max(pairs, key=pairs.get)
            merges.append(best)

            new_vocab = {}
            bigram = re.escape(' '.join(best))
            p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
            new_pair = ''.join(best)
            for word in vocab:
                w_out = p.sub(new_pair, word)
                new_vocab[w_out] = new_vocab.get(w_out, 0) + vocab[word]
            vocab = new_vocab
            
            if (i+1) % 200 == 0: print(f"BPE: Đã gộp {i+1} cặp...")
            
        self._merge_rules = merges
        self._compiled_merges = []
        for pair in merges:
            bigram = re.escape(' '.join(pair))
            # Pattern này sẽ được dùng lại cho mọi câu
            pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
            replacement = ''.join(pair)
            self._compiled_merges.append((pattern, replacement))
        
        self._merge_rules = merges
        return merges

    def _tokenize(self, sentence: str):
        words = self._clean_text(sentence).split()
        final_tokens = []
        
        for word in words:
            word_formatted = ' '.join(list(word)) + '</w>'
            for pattern, replacement in self._compiled_merges:
                word_formatted = pattern.sub(replacement, word_formatted)
            final_tokens.extend(word_formatted.split())
        return final_tokens

    def fit(self, df):
        df = df.copy()
        print("--- Bắt đầu huấn luyện BayesIntent ---")
        cleaned_sentences = df["Sentence"].apply(self._clean_text)
        self._learn_BPE(cleaned_sentences)
        
        num_cores = multiprocessing.cpu_count() // 2
        df["tokens"] = Parallel(n_jobs=num_cores)(delayed(self._tokenize)(s) for s in tqdm(cleaned_sentences))

        # Tính Prior P(I)
        counts = Counter(df["intent"])
        total = sum(counts.values())
        self._prior = {intent: count / total for intent, count in counts.items()}

        # Tính Likelihood P(Wi | I) với Laplace Smoothing
        all_tokens = [t for sublist in df["tokens"].to_list() for t in sublist]
        unique_tokens = set(all_tokens)
        self._vocab_size = len(unique_tokens)

        self._likelihood = {}
        for intent in self._prior.keys():
            intent_tokens = [t for sublist in df[df["intent"] == intent]["tokens"].to_list() for t in sublist]
            token_counts = Counter(intent_tokens)
            total_tokens_in_intent = len(intent_tokens)
            
            self._likelihood[intent] = {
                token: (token_counts[token] + 1) / (total_tokens_in_intent + self._vocab_size)
                for token in unique_tokens
            }
        print("--- Huấn luyện hoàn tất! ---")

    def predict(self, sentence: str | list):
        if isinstance(sentence, list):
            return [self.predict(s) for s in sentence]
            
        tokens = self._tokenize(sentence)
        scores = {}

        for intent in self._prior.keys():
            score = math.log(self._prior[intent])
            for token in tokens:
                if token in self._likelihood[intent]:
                    score += math.log(self._likelihood[intent][token])
                else:
                    score += math.log(1 / (self._vocab_size + 1))
            scores[intent] = score
        
        best_intent = max(scores, key=scores.get)
        
        # Tính Confidence (Softmax)
        max_log = scores[best_intent]
        exp_scores = {k: math.exp(v - max_log) for k, v in scores.items()}
        confidence = exp_scores[best_intent] / sum(exp_scores.values())

        return {"intent": best_intent, "confidence": round(confidence * 100, 2)}

    def predict_with_steps(self, sentence: str):
        tokens = self._tokenize(sentence)
        scores = {}
        breakdown_details = {}

        for intent in self._prior.keys():
            prior_log = math.log(self._prior[intent])
            score = prior_log
            
            intent_breakdown = {
                "prior": round(prior_log, 2),
                "tokens": {}
            }

            for token in tokens:
                if token in self._likelihood[intent]:
                    val = math.log(self._likelihood[intent][token])
                else:
                    val = math.log(1 / (self._vocab_size + 1))
                
                score += val
                intent_breakdown["tokens"][token] = round(val, 2)

            scores[intent] = score
            breakdown_details[intent] = intent_breakdown
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_intent, max_log = sorted_scores[0]
        
        exp_scores = {k: math.exp(v - max_log) for k, v in scores.items()}
        total_exp = sum(exp_scores.values())
        confidence = exp_scores[best_intent] / total_exp

        top_3_breakdowns = {item[0]: breakdown_details[item[0]] for item in sorted_scores[:3]}

        return {
            "tokens": tokens,
            "intent": best_intent,
            "confidence": round(confidence * 100, 2),
            "top_3": sorted_scores[:3],
            "breakdowns": top_3_breakdowns
        }

    def to_json(self, path):
        data = {
            "prior": self._prior,
            "likelihood": self._likelihood,
            "vocab_size": self._vocab_size,
            "merge_rules": [list(pair) for pair in self._merge_rules]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    @classmethod
    def from_json(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        instance = cls(num_merges=len(data["merge_rules"]))
        
        instance._prior = data["prior"]
        instance._likelihood = data["likelihood"]
        instance._vocab_size = data["vocab_size"]
        instance._merge_rules = [tuple(pair) for pair in data["merge_rules"]]
        
        instance._compiled_merges = []
        for pair in instance._merge_rules:
            bigram = re.escape(' '.join(pair))
            pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
            replacement = ''.join(pair)
            instance._compiled_merges.append((pattern, replacement))
            
        return instance

    def _perform_k_fold(self, df):
        df = df.copy()
        X = df["Sentence"]
        y = df["intent"]

        k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        metrics_result = {"f1": [], "accuracy": [], "precision": [], "recall": []}
        num_cores = multiprocessing.cpu_count() // 2

        all_y_true = []
        all_y_pred = []

        for fold, (train_idx, test_idx) in enumerate(k_fold.split(X, y)):
            train_data = df.iloc[train_idx].copy()
            test_data = df.iloc[test_idx].copy()

            self.fit(train_data)

            y_true = test_data["intent"].to_list()
            results = Parallel(n_jobs=num_cores)(delayed(self.predict)(s) for s in tqdm(test_data["Sentence"]))
            y_pred = [res["intent"] for res in results]

            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)

            f1 = f1_score(y_true, y_pred, average="weighted")
            acc = accuracy_score(y_true, y_pred)
            pre = precision_score(y_true, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)

            metrics_result["f1"].append(f1)
            metrics_result["accuracy"].append(acc)
            metrics_result["precision"].append(pre)
            metrics_result["recall"].append(rec)
            
            print(f"\nFold {fold+1}: F1={f1:.4f} | Acc={acc:.4f} | Pre={pre:.4f} | Rec={rec:.4f}\n")

        print("\n" + "="*30)
        print("KẾT QUẢ TRUNG BÌNH CUỐI CÙNG:")
        for metric, values in metrics_result.items():
            print(f"{metric.upper()}: {np.mean(values):.4f} (+/- {np.std(values):.4f})")
        print("="*30)

        print("\n" + "="*30)
        print("CONFUSION MATRIX TOÀN BỘ 5 FOLDS:")
        
        labels = list(self._prior.keys())
        
        cm = confusion_matrix(all_y_true, all_y_pred, labels=labels)
        display = ConfusionMatrixDisplay(cm, display_labels=labels)
        
        fig, ax = plt.subplots(figsize=(12, 12))
        display.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')
        plt.tight_layout()
        plt.savefig("confusion_matrix.svg", format="svg", bbox_inches="tight")
        plt.show()