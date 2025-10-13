# Based on seminar materials


def _levenshtein_distance(seq_a, seq_b):
    len_a = len(seq_a)
    len_b = len(seq_b)

    if len_a == 0:
        return len_b
    if len_b == 0:
        return len_a

    prev_row = list(range(len_b + 1))
    for i, char_a in enumerate(seq_a, start=1):
        current_row = [i]
        for j, char_b in enumerate(seq_b, start=1):
            substitution_cost = 0 if char_a == char_b else 1
            current_row.append(
                min(
                    prev_row[j] + 1,
                    current_row[j - 1] + 1,
                    prev_row[j - 1] + substitution_cost,
                )
            )
        prev_row = current_row
    return prev_row[-1]


def calc_cer(target_text, predicted_text) -> float:
    if not target_text:
        return float(predicted_text != "")
    distance = _levenshtein_distance(target_text, predicted_text)
    return distance / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    target_words = target_text.split()
    predicted_words = predicted_text.split()

    if not target_words:
        return float(bool(predicted_words))

    distance = _levenshtein_distance(target_words, predicted_words)
    return distance / len(target_words)
