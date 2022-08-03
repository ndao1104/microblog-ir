import re
import string


def output_ranks(result_file_name, ranks, run_id):
    result = open(result_file_name, "w", encoding="utf-8")

    result.write("{0:<6}\t{1:<2}\t{2:<16}\t{3:<4}\t{4:<4}\t{5:<}\n".format(
        "qry_id", "Q0", "doc_id", "rank", "score", "run_id"))

    for query, relevant_documents in ranks.items():
        for rank, (document, score) in enumerate(relevant_documents.items(), start=1):
            if score <= 0 or rank > 1000:
                break
            result.write("{0:<6}\t{1:<2}\t{2:<16}\t{3:<4}\t{4:<.3f}\t{5:<}\n".format(
                query, "Q0", document, rank, score, run_id))

    result.close()


def remove_url(text):
    return re.sub("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "", text)


def remove_punct(text):
    table = text.maketrans({key: None for key in string.punctuation})

    return text.translate(table)


def remove_tag(text, tag):
    text = text.replace('<' + tag + '>', '')
    text = text.replace('</' + tag + '>', '')

    return text.strip()
