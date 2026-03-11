"""
Evaluation pipeline for zero-shot LLM extraction of chemical and radiological events.

This script reproduces the entity-level precision, recall, and F1 scores reported
in the manuscript using a 200-article gold-standard dataset annotated by human reviewers.

Author: Damian Honeyman et al.
"""
import argparse
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from scipy.stats import beta


class NEREvaluation:
    def __init__(self, input_path: str, embedding_model: str):
        self.input_path = input_path

        if input_path.lower().endswith(".xlsx"):
            self.df = pd.read_excel(input_path)
        else:
            self.df = pd.read_csv(input_path)

        self.embedding_model = SentenceTransformer(embedding_model, trust_remote_code=True)

    def normalize_label(self, label: str):

        cleaned = str(label).strip()

        mapping = {
            "chemical substance": "CHEMICAL SUBSTANCE",
            "radiological substance": "RADIOLOGICAL SUBSTANCE",
            "fatality count": "FATALITY COUNT",
            "case number": "CASE NUMBER",
            "state": "STATE",
            "county": "COUNTY",
            "city": "CITY",
            "location": "LOCATION",
            "timeframe": "TIMEFRAME",
            "answer": "Answer",
            "adverbs": "Adverbs",
            "years": "Years",
            "months": "Months",
            "dates": "Dates",
            "date": "date",
            "dates or days of the week": "Dates or days of the week",
        }

        return mapping.get(cleaned.lower(), cleaned)

    def convert_entities_to_dictionary(self, text: str):

        if pd.isna(text) or str(text).strip() == "":
            return {}

        lines = str(text).splitlines()
        entity_dict = {}
        current_key = None

        for raw_line in lines:

            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("-"):
                line = line[1:].strip()

            if ":" in line:

                key_part, value_part = line.split(":", 1)
                key = self.normalize_label(key_part)
                value = value_part.strip()

                entity_dict.setdefault(key, [])
                current_key = key

                if value != "":
                    entity_dict[key].append(value)

            else:

                if current_key is not None:
                    entity_dict.setdefault(current_key, []).append(line)

        # deduplicate
        for key, values in entity_dict.items():

            seen = set()
            deduped = []

            for value in values:

                v = str(value).strip()

                if v.lower() not in seen:
                    deduped.append(v)
                    seen.add(v.lower())

            entity_dict[key] = deduped

        return entity_dict

    def get_embedding_sentence_transformers(self, texts):

        embeddings = self.embedding_model.encode(texts)

        if embeddings.ndim == 2 and embeddings.shape[1] > 768:
            embeddings = embeddings[:, :768]

        return embeddings

    def fuzzy_match(self, prediction, actual, threshold=1.0):

        actual_lower = [str(i).lower() for i in actual]
        predicted_lower = [str(i).lower() for i in prediction]

        common_lower = set(actual_lower).intersection(predicted_lower)

        actual_filtered = [i for i in actual if str(i).lower() not in common_lower]
        predicted_filtered = [i for i in prediction if str(i).lower() not in common_lower]

        if not predicted_filtered or not actual_filtered:
            return prediction

        prediction_embedding = self.get_embedding_sentence_transformers(predicted_filtered)
        actual_embedding = self.get_embedding_sentence_transformers(actual_filtered)

        cosine_sim_matrix = cosine_similarity(prediction_embedding, actual_embedding)

        for r in range(len(cosine_sim_matrix)):
            for c in range(len(cosine_sim_matrix[r])):
                if cosine_sim_matrix[r][c] >= threshold:
                    cosine_sim_matrix[r][c] = 1

        row_idx, col_idx = linear_sum_assignment(-cosine_sim_matrix)

        term_arr = []

        for i in range(len(row_idx)):
            if cosine_sim_matrix[row_idx[i], col_idx[i]] == 1:
                term_arr.append(actual_filtered[col_idx[i]])
            else:
                term_arr.append(predicted_filtered[row_idx[i]])

        remaining_idx = set(range(len(predicted_filtered))) - set(row_idx)

        for idx in remaining_idx:
            term_arr.append(predicted_filtered[idx])

        common_original = [i for i in actual if str(i).lower() in common_lower]

        term_arr.extend(common_original)

        return term_arr

    def merge_entity_labels(self, ner_dict):

        if not ner_dict:
            return ner_dict

        merged = {}

        for key, values in ner_dict.items():

            key_clean = str(key).strip().lower()

            if key_clean == "chemical substance":
                key_norm = "CHEMICAL SUBSTANCE"

            elif key_clean == "radiological substance":
                key_norm = "RADIOLOGICAL SUBSTANCE"

            elif key_clean == "timeframe":
                key_norm = "TIMEFRAME"

            elif key_clean == "date":
                key_norm = "date"

            else:
                key_norm = self.normalize_label(key)

            merged.setdefault(key_norm, []).extend(values)

        for key, values in merged.items():

            seen = set()
            deduped = []

            for value in values:

                v = str(value).strip()

                if v.lower() not in seen:
                    deduped.append(v)
                    seen.add(v.lower())

            merged[key] = deduped

        return merged

    def evaluate_ner(self, predicted_ner, label_ner):

        predicted_ner = self.merge_entity_labels(predicted_ner)
        label_ner = self.merge_entity_labels(label_ner)

        considered_entities = list(set(predicted_ner.keys()) | set(label_ner.keys()))

        metrics = {}

        for entity_type in considered_entities:

            tp = fp = fn = tn = 0

            if entity_type not in predicted_ner and entity_type not in label_ner:
                tn += 1

            elif entity_type not in predicted_ner:
                fn += len(label_ner[entity_type])

            elif entity_type not in label_ner:
                fp += len(predicted_ner[entity_type])

            else:

                pred = predicted_ner[entity_type]
                lab = label_ner[entity_type]

                fuzzy = self.fuzzy_match(pred, lab)

                pred_set = {i.lower() for i in fuzzy}
                lab_set = {i.lower() for i in lab}

                tp = len(pred_set & lab_set)
                fp = len(pred_set - lab_set)
                fn = len(lab_set - pred_set)

            metrics[entity_type] = {
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn
            }

        return metrics

    def calculate_precision(self, tp, fp):
        return 0 if tp + fp == 0 else tp / (tp + fp)

    def calculate_recall(self, tp, fn):
        return 0 if tp + fn == 0 else tp / (tp + fn)

    def calculate_f_measure(self, p, r):
        return 0 if p + r == 0 else 2 * (p * r) / (p + r)

    def clopper_pearson_ci(self, successes, trials, alpha=0.05):

        if trials == 0:
            return 0, 1

        x = int(successes)
        n = int(trials)

        low = 0 if x == 0 else beta.ppf(alpha / 2, x, n - x + 1)
        high = 1 if x == n else beta.ppf(1 - alpha / 2, x + 1, n - x)

        return float(low), float(high)

    def fit(self):

        required_columns = {"feedback", "gpt4"}

        if not required_columns.issubset(self.df.columns):
            raise ValueError("Dataset must contain 'gpt4' and 'feedback' columns")

        full_eval = {"gpt4": {}}

        for _, row in self.df.iterrows():

            actual = row["feedback"]
            prediction = row["gpt4"]

            actual_dict = self.convert_entities_to_dictionary(actual)
            pred_dict = self.convert_entities_to_dictionary(prediction)

            evaluation = self.evaluate_ner(pred_dict, actual_dict)

            for entity, scores in evaluation.items():

                if entity not in full_eval["gpt4"]:
                    full_eval["gpt4"][entity] = scores.copy()

                else:
                    for k in scores:
                        full_eval["gpt4"][entity][k] += scores[k]

        allowed_entities = {
            "Answer",
            "CHEMICAL SUBSTANCE",
            "RADIOLOGICAL SUBSTANCE",
            "FATALITY COUNT",
            "CASE NUMBER",
            "STATE",
            "COUNTY",
            "CITY",
            "LOCATION",
            "TIMEFRAME",
            "Years",
            "Months",
            "Dates",
            "date",
            "Dates or days of the week",
            "Adverbs",
        }

        # Reporting rename
        reporting_names = {
            "LOCATION": "COUNTRY",
            "Answer": "Event identification"
        }

        rows = []

        for entity, m in full_eval["gpt4"].items():

            if entity not in allowed_entities:
                continue

            p = self.calculate_precision(m["TP"], m["FP"])
            r = self.calculate_recall(m["TP"], m["FN"])
            f = self.calculate_f_measure(p, r)

            p_low, p_high = self.clopper_pearson_ci(m["TP"], m["TP"] + m["FP"])
            r_low, r_high = self.clopper_pearson_ci(m["TP"], m["TP"] + m["FN"])

            entity_name = reporting_names.get(entity, entity)

            rows.append({
                "ENTITY_TYPE": entity_name,
                "precision": p,
                "precision_ci_low": p_low,
                "precision_ci_high": p_high,
                "recall": r,
                "recall_ci_low": r_low,
                "recall_ci_high": r_high,
                "fmeasure": f,
                "TP": m["TP"],
                "FP": m["FP"],
                "FN": m["FN"],
                "TN": m["TN"],
            })

        return pd.DataFrame(rows)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="INSERT_INPUT_FILE"
    )

    parser.add_argument(
        "--output",
        default="NAME_YOUR_OUTPUT_FILE"
    )

    parser.add_argument(
        "--embedding_model",
        default="sentence-transformers/all-MiniLM-L6-v2"
    )

    args = parser.parse_args()

    evaluator = NEREvaluation(args.input, args.embedding_model)

    output_df = evaluator.fit()

    output_df.to_csv(args.output, index=False)

    print("Saved evaluation results to:", args.output)


if __name__ == "__main__":
    main()
