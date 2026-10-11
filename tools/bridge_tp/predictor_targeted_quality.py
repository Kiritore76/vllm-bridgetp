"""Deterministic completion diagnostics, not official semantic benchmark scores."""

import csv
import io
import re
from collections import Counter


def _lines(text):
    return [line.strip() for line in text.split("\n") if line.strip()]


def repetition(token_ids):
    """Report repeated nonoverlapping 32-token blocks without rejecting prose."""
    blocks = [tuple(token_ids[i : i + 32]) for i in range(0, len(token_ids) - 31, 32)]
    counts = Counter(blocks)
    return {
        "full_32_token_blocks": len(blocks),
        "duplicate_32_token_block_fraction": (
            (len(blocks) - len(counts)) / len(blocks) if blocks else 0.0
        ),
        "maximum_32_token_block_occurrences": max(counts.values(), default=0),
    }


def _table(text, spec):
    match = re.search(r"```tsv\s*\n(.*?)(?:```|\Z)", text, re.S)
    body = match.group(1) if match else text

    def parse(value):
        return [
            tuple(cell.strip() for cell in row)
            for row in csv.reader(io.StringIO(value), delimiter="\t")
            if any(cell.strip() for cell in row)
        ]

    expected = parse(spec["reference_output"])
    actual = parse(body)
    header = expected[0]
    actual = [row for row in actual if row != header]
    reference = Counter(expected[1:])
    predicted = Counter(actual)
    matched = sum((reference & predicted).values())
    # First-cell coverage is separate from stricter whole-row agreement.
    keys = Counter(row[0] for row in expected[1:])
    output_keys = Counter(row[0] for row in actual)
    return {
        "expected_rows": sum(reference.values()),
        "observed_rows": len(actual),
        "exact_rows_matched": matched,
        "exact_row_recall": matched / sum(reference.values()),
        "first_cell_rows_matched": sum((keys & output_keys).values()),
        "missing_first_cells": list((keys - output_keys).elements()),
        "extra_or_duplicate_first_cells": list((output_keys - keys).elements()),
        "malformed_column_rows": sum(len(row) != len(header) for row in actual),
        "complete_exact_rows": predicted == reference,
    }


def _route(text, spec):
    match = re.search(r"<Route>(.*?)(?:</Route>|\Z)", text, re.S)
    actual = _lines(match.group(1) if match else text)
    expected = _lines(spec["reference_output"])
    common = 0
    for truth, prediction in zip(expected, actual):
        if truth != prediction:
            break
        common += 1
    matched = sum((Counter(expected) & Counter(actual)).values())
    return {
        "expected_steps": len(expected),
        "observed_lines": len(actual),
        "exact_steps_matched_any_order": matched,
        "correct_prefix_steps": common,
        "correct_prefix_fraction": common / len(expected),
        "duplicate_lines": len(actual) - len(set(actual)),
        "closing_route_tag": "</Route>" in text,
        "complete_exact_route": actual == expected and "</Route>" in text,
    }


def _weekly(text, spec):
    kind = re.escape(spec["type"])
    pattern = rf"#\*#\s*{kind}\s+(\d+)\b[^\n]*\n"
    matches = list(re.finditer(pattern, text))
    implicit_first = (
        bool(text.strip())
        and (not matches or matches[0].start() > 0)
        and (not matches or int(matches[0].group(1)) != 1)
    )
    # The published user prompt ends in the Week-1 heading. A model may
    # continue its body directly rather than emit that heading again.
    if implicit_first:
        text = spec["prefix"] + "\n" + text
        matches = list(re.finditer(pattern, text))
    entries = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = re.split(r"\*\*\*\s*finished", text[match.end() : end], flags=re.I)[0]
        entries.append((int(match.group(1)), body.strip()))
    numbers = Counter(number for number, _ in entries)
    expected = set(range(1, spec["number"] + 1))
    words = [len(re.findall(r"\b[\w]+(?:['’-][\w]+)*\b", body)) for _, body in entries]
    normalized = [re.sub(r"\s+", " ", body).lower() for _, body in entries if body]
    bodies = {number: body.lower() for number, body in entries}
    hints = []
    for category in ("checks_once", "checks_range", "checks_periodic"):
        for number, hint in spec[category].items():
            hints.append(
                {
                    "category": category,
                    "entry": int(number),
                    "literal_hint": hint,
                    "literal_phrase_found": hint.lower() in bodies.get(int(number), ""),
                }
            )
    return {
        "expected_entries": spec["number"],
        "observed_entries": len(entries),
        "entry_numbers": [number for number, _ in entries],
        "missing_entries": sorted(expected - numbers.keys()),
        "duplicate_entries": sorted(n for n, count in numbers.items() if count > 1),
        "out_of_range_entries": sorted(numbers.keys() - expected),
        "in_order": [number for number, _ in entries] == sorted(numbers.elements()),
        "word_counts": words,
        "minimum_words_requested": spec["minimum_words_per_entry"],
        "entries_meeting_word_minimum": sum(
            count >= spec["minimum_words_per_entry"] for count in words
        ),
        "identical_nonempty_entry_bodies": len(normalized) - len(set(normalized)),
        "completion_marker": bool(re.search(r"\*\*\*\s*finished", text, re.I)),
        "complete_entry_number_sequence": (
            [number for number, _ in entries] == list(range(1, spec["number"] + 1))
        ),
        "first_heading_inferred_from_prompt": implicit_first,
        "literal_constraint_hints_not_semantic_scores": hints,
    }


def evaluate(kind, text, spec, token_ids):
    """Measure explicit rows/steps/entries and repetition from a saved response."""
    evaluator = {"html_to_tsv": _table, "path_traversal": _route}
    metrics = evaluator.get(kind, _weekly)(text, spec)
    return {
        "diagnostic_only_not_official_benchmark_score": True,
        **metrics,
        **repetition(token_ids),
    }
