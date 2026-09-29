import bz2
import html
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

import mwparserfromhell
from tqdm import tqdm


# ============================================================
# FILES
# ============================================================

DUMP_PATH = Path("elwiki-latest-pages-articles.xml.bz2")
METADATA_PATH = Path("greek_people_metadata_full.json")

BIOGRAPHIES_OUTPUT = Path("greek_biographies.jsonl")
QA_OUTPUT = Path("greek_biographies_qa.jsonl")
NOT_FOUND_OUTPUT = Path("not_found_titles.json")
# Use 100 for a small test.
# Use None to process every matched biography.
MAX_BIOGRAPHIES = None

MIN_LEAD_LENGTH = 10

MALE_QID = "Q6581097"
FEMALE_QID = "Q6581072"


# These sections do not contain biographical information.
EXCLUDED_SECTIONS = {
    "παραπομπες",
    "αναφορες",
    "σημειωσεις",
    "πηγες",
    "βιβλιογραφια",
    "εξωτερικοι συνδεσμοι",
    "δειτε επισης",
}


# Readable names for the Wikidata fields.
FACT_LABELS = {
    "gender_details": "Φύλο",
    "countries_of_citizenship": "Υπηκοότητα",
    "ethnic_groups": "Εθνοτική ομάδα",

    "spouses": "Σύζυγοι",
    "unmarried_partners": "Σύντροφοι",
    "children": "Παιδιά",
    "number_of_children": "Αριθμός παιδιών",
    "father": "Πατέρας",
    "mother": "Μητέρα",
    "siblings": "Αδέλφια",

    "dates_of_birth": "Ημερομηνία γέννησης",
    "places_of_birth": "Τόπος γέννησης",
    "dates_of_death": "Ημερομηνία θανάτου",
    "places_of_death": "Τόπος θανάτου",
    "causes_of_death": "Αιτία θανάτου",
    "manners_of_death": "Τρόπος θανάτου",
    "places_of_burial": "Τόπος ταφής",

    "occupations": "Επαγγέλματα",
    "educated_at": "Σπουδές",
    "employers": "Εργοδότες",
    "positions_held": "Θέσεις που κατείχε",
    "fields_of_work": "Τομείς εργασίας",
    "work_locations": "Τόποι εργασίας",

    "political_parties": "Πολιτικά κόμματα",
    "memberships": "Οργανισμοί στους οποίους ήταν μέλος",

    "awards": "Βραβεία και διακρίσεις",
    "notable_works": "Σημαντικά έργα",
    "movements": "Καλλιτεχνικά ή ιδεολογικά κινήματα",
    "genres": "Είδη",
    "instruments": "Μουσικά όργανα",

    "military_branches": "Κλάδοι ενόπλων δυνάμεων",
    "military_ranks": "Στρατιωτικοί βαθμοί",
    "conflicts": "Πόλεμοι και συγκρούσεις",

    "sports_teams": "Αθλητικές ομάδες",
    "playing_positions": "Αγωνιστικές θέσεις",

    "residences": "Τόποι κατοικίας",
    "religions_or_worldviews": "Θρησκεία ή κοσμοθεωρία",
    "native_languages": "Μητρικές γλώσσες",
    "languages_spoken": "Γλώσσες",
    "participants_in": "Συμμετοχές",
}


# ============================================================
# TEXT CLEANING
# ============================================================

def normalize_title(title):
    """Make Wikipedia and Wikidata titles easier to match."""
    return unicodedata.normalize(
        "NFC",
        title.replace("_", " ").strip(),
    )


def normalize_heading(text):
    """Lowercase text and remove Greek accents."""
    text = unicodedata.normalize("NFD", text.lower())

    text = "".join(
        character
        for character in text
        if unicodedata.category(character) != "Mn"
    )

    return re.sub(r"\s+", " ", text).strip()


def clean_wikitext(wikitext):
    """Convert Wikipedia markup into readable plain text."""
    if not wikitext:
        return ""

    # Remove comments.
    wikitext = re.sub(
        r"<!--.*?-->",
        " ",
        wikitext,
        flags=re.DOTALL,
    )

    # Remove references.
    wikitext = re.sub(
        r"<ref\b[^>]*>.*?</ref\s*>",
        " ",
        wikitext,
        flags=re.IGNORECASE | re.DOTALL,
    )

    wikitext = re.sub(
        r"<ref\b[^>]*/\s*>",
        " ",
        wikitext,
        flags=re.IGNORECASE,
    )

    # Remove Wikipedia tables.
    wikitext = re.sub(
        r"\{\|.*?\|\}",
        " ",
        wikitext,
        flags=re.DOTALL,
    )

    # Remove categories, files, and images.
    wikitext = re.sub(
        r"\[\[(?:Κατηγορία|Category|Αρχείο|File|Εικόνα|Image):.*?\]\]",
        " ",
        wikitext,
        flags=re.IGNORECASE | re.DOTALL,
    )

    wikicode = mwparserfromhell.parse(wikitext)

    # Remove templates from the prose.
    for template in list(
        wikicode.filter_templates(recursive=True)
    ):
        try:
            wikicode.remove(template)
        except ValueError:
            pass

    plain_text = wikicode.strip_code(
        normalize=True,
        collapse=False,
        keep_template_params=False,
    )

    plain_text = html.unescape(plain_text)

    # Remove remaining HTML tags.
    plain_text = re.sub(
        r"<[^>]+>",
        " ",
        plain_text,
    )

    # Normalize whitespace.
    plain_text = re.sub(r"[ \t]+", " ", plain_text)
    plain_text = re.sub(r" *\n *", "\n", plain_text)
    plain_text = re.sub(r"\n{3,}", "\n\n", plain_text)

    return plain_text.strip()


# ============================================================
# ARTICLE EXTRACTION
# ============================================================

def extract_article(raw_text):
    """
    Extract:
    - introduction
    - sections
    - infobox
    - complete cleaned article
    """
    wikicode = mwparserfromhell.parse(raw_text)

    # --------------------------------------------------------
    # Extract the infobox before templates are removed.
    # --------------------------------------------------------

    infobox = {}

    for template in wikicode.filter_templates(
        recursive=False
    ):
        template_name = normalize_heading(
            str(template.name)
        )

        is_infobox = (
            template_name.startswith("πληροφορι")
            or "infobox" in template_name
        )

        if not is_infobox:
            continue

        fields = {}

        for parameter in template.params:
            key = clean_wikitext(
                str(parameter.name)
            )

            value = clean_wikitext(
                str(parameter.value)
            )

            if key and value:
                fields[key] = value

        infobox = {
            "template": str(template.name).strip(),
            "fields": fields,
        }

        # Use only the first probable infobox.
        break

    # --------------------------------------------------------
    # Extract introduction and level-two sections.
    # --------------------------------------------------------

    section_nodes = wikicode.get_sections(
        include_lead=True,
        include_headings=True,
        levels=[2],
        flat=True,
    )

    if not section_nodes:
        lead = clean_wikitext(raw_text)

        return lead, [], infobox, lead

    lead = clean_wikitext(
        str(section_nodes[0])
    )

    sections = []
    full_text_parts = [lead]

    for section_node in section_nodes[1:]:
        section_code = mwparserfromhell.parse(
            str(section_node)
        )

        headings = section_code.filter_headings(
            recursive=False
        )

        if not headings:
            continue

        heading_node = headings[0]

        heading = clean_wikitext(
            str(heading_node.title)
        )

        try:
            section_code.remove(heading_node)
        except ValueError:
            pass

        section_text = clean_wikitext(
            str(section_code)
        )

        if not heading or not section_text:
            continue

        normalized_heading = normalize_heading(
            heading
        )

        if normalized_heading in EXCLUDED_SECTIONS:
            continue

        sections.append(
            {
                "heading": heading,
                "text": section_text,
            }
        )

        full_text_parts.append(
            f"{heading}\n{section_text}"
        )

    full_text = "\n\n".join(full_text_parts)

    return lead, sections, infobox, full_text


# ============================================================
# WIKIDATA FORMATTING
# ============================================================

def format_value(item):
    """Convert one Wikidata value into readable text."""
    if item is None:
        return ""

    if isinstance(item, (str, int, float)):
        return str(item)

    if not isinstance(item, dict):
        return str(item)

    # A full statement may contain another dictionary in "value".
    if "value" in item and isinstance(
        item["value"],
        dict,
    ):
        text = format_value(item["value"])

        qualifiers = item.get("qualifiers", {})
        qualifier_text = []

        if isinstance(qualifiers, dict):
            for qualifier_name, qualifier_values in qualifiers.items():

                if not isinstance(
                    qualifier_values,
                    list,
                ):
                    qualifier_values = [
                        qualifier_values
                    ]

                values = [
                    format_value(value)
                    for value in qualifier_values
                ]

                values = [
                    value
                    for value in values
                    if value
                ]

                if values:
                    qualifier_text.append(
                        f"{qualifier_name}: "
                        + ", ".join(values)
                    )

        if qualifier_text:
            return (
                f"{text} "
                f"({'; '.join(qualifier_text)})"
            )

        return text

    # Entity with Greek or English label.
    if item.get("label"):
        return str(item["label"])

    # Wikidata time value.
    if item.get("type") == "time":
        raw_date = str(item.get("time", ""))

    elif (
        "value" in item
        and isinstance(item["value"], str)
        and "T" in item["value"]
    ):
        raw_date = item["value"]

    else:
        raw_date = ""

    if raw_date:
        raw_date = raw_date.lstrip("+")

        match = re.match(
            r"(-?\d+)-(\d{2})-(\d{2})T",
            raw_date,
        )

        if match:
            year, month, day = match.groups()

            if month == "00":
                return year

            if day == "00":
                return f"{month}/{year}"

            return f"{day}/{month}/{year}"

        return raw_date

    # Wikidata quantity.
    if item.get("type") == "quantity":
        return str(
            item.get("amount", "")
        ).lstrip("+")

    # Simple value from the SPARQL metadata script.
    if "value" in item:
        return str(item["value"]).lstrip("+")

    # Entity without a label.
    if item.get("qid"):
        return str(item["qid"])

    if item.get("time"):
        return str(item["time"])

    if item.get("amount") is not None:
        return str(item["amount"]).lstrip("+")

    return json.dumps(
        item,
        ensure_ascii=False,
    )


def build_complete_answer(full_text, facts):
    answer_parts = []

    if full_text:
        answer_parts.append(full_text)

    fact_lines = []

    for field_name, values in facts.items():
        if not values:
            continue

        if not isinstance(values, list):
            values = [values]

        readable_values = []

        for value in values:
            text = format_value(value)

            if text and text not in readable_values:
                readable_values.append(text)

        if readable_values:
            readable_name = FACT_LABELS.get(
                field_name,
                field_name.replace("_", " "),
            )

            fact_lines.append(
                f"{readable_name}: "
                + ", ".join(readable_values)
            )

    if fact_lines:
        answer_parts.append(
            "Δομημένα στοιχεία από το Wikidata\n"
            + "\n".join(fact_lines)
        )

    return "\n\n".join(answer_parts)


# ============================================================
# MAIN EXTRACTION
# ============================================================

def main():
    if not DUMP_PATH.exists():
        raise FileNotFoundError(
            f"Δεν βρέθηκε το Wikipedia dump: "
            f"{DUMP_PATH.resolve()}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Δεν βρέθηκε το metadata file: "
            f"{METADATA_PATH.resolve()}"
        )

    # Load the Greek-person metadata.
    with METADATA_PATH.open(
        encoding="utf-8"
    ) as metadata_file:
        people = json.load(metadata_file)

    # Create a fast lookup using the article title.
    people_by_title = {
        normalize_title(person["title"]): person
        for person in people
    }

    print(
        f"Loaded metadata for "
        f"{len(people_by_title):,} people"
    )

    biography_count = 0
    matched_titles = set()

    # Open the compressed XML directly.
    with (
        bz2.open(DUMP_PATH, "rb") as xml_file,
        BIOGRAPHIES_OUTPUT.open(
            "w",
            encoding="utf-8",
        ) as biography_file,
        QA_OUTPUT.open(
            "w",
            encoding="utf-8",
        ) as qa_file,
        tqdm(
            desc="Scanning Wikipedia pages",
            unit=" pages",
        ) as progress,
    ):
        context = ET.iterparse(
            xml_file,
            events=("start", "end"),
        )

        _, root = next(context)

        for event, page in context:

            if (
                event != "end"
                or not page.tag.endswith("page")
            ):
                continue

            progress.update(1)

            title_element = page.find("./{*}title")
            namespace_element = page.find("./{*}ns")
            redirect_element = page.find("./{*}redirect")
            revision_element = page.find("./{*}revision")

            # Skip incomplete XML pages.
            if (
                title_element is None
                or namespace_element is None
                or revision_element is None
            ):
                root.clear()
                continue

            # Namespace 0 contains normal Wikipedia articles.
            if namespace_element.text != "0":
                root.clear()
                continue

            # Skip redirects.
            if redirect_element is not None:
                root.clear()
                continue

            title = normalize_title(
                title_element.text or ""
            )

            # Skip people not found in the Wikidata whitelist.
            if title not in people_by_title:
                root.clear()
                continue

            text_element = revision_element.find(
                "./{*}text"
            )

            if (
                text_element is None
                or not text_element.text
            ):
                root.clear()
                continue

            revision_id_element = revision_element.find(
                "./{*}id"
            )

            timestamp_element = revision_element.find(
                "./{*}timestamp"
            )

            metadata = people_by_title[title]
            raw_text = text_element.text

            lead, sections, infobox, full_text = (
                extract_article(raw_text)
            )

            if len(lead) < MIN_LEAD_LENGTH:
                root.clear()
                continue

            facts = metadata.get("facts", {})

            revision_id = (
                revision_id_element.text
                if revision_id_element is not None
                else None
            )

            revision_timestamp = (
                timestamp_element.text
                if timestamp_element is not None
                else None
            )

            # ------------------------------------------------
            # Structured biography output.
            # ------------------------------------------------

            biography_record = {
                "title": title,
                "wikidata_id": metadata.get(
                    "wikidata_id"
                ),
                "greek_criteria": metadata.get(
                    "criteria",
                    [],
                ),
                "gender": metadata.get(
                    "gender",
                    [],
                ),
                "birth": metadata.get(
                    "birth",
                    [],
                ),
                "death": metadata.get(
                    "death",
                    [],
                ),
                # "facts": facts,
                "summary_counts": metadata.get(
                    "summary_counts",
                    {},
                ),
                "lead": lead,
                "sections": sections,
                "infobox": infobox,
                "full_text": full_text,
                "revision_id": revision_id,
                "revision_timestamp": revision_timestamp,
                "source_dump": DUMP_PATH.name,
            }

            biography_file.write(
                json.dumps(
                    biography_record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            # ------------------------------------------------
            # Create the complete QA response.
            # ------------------------------------------------

            gender = set(
                metadata.get("gender", [])
            )

            if FEMALE_QID in gender:
                question = (
                    f"Ποια είναι η {title}; "
                    f"Δώσε όλες τις διαθέσιμες πληροφορίες "
                    f"για τη ζωή, την οικογένεια, "
                    f"τη σταδιοδρομία και το έργο της."
                )

            elif MALE_QID in gender:
                question = (
                    f"Ποιος είναι ο {title}; "
                    f"Δώσε όλες τις διαθέσιμες πληροφορίες "
                    f"για τη ζωή, την οικογένεια, "
                    f"τη σταδιοδρομία και το έργο του."
                )

            else:
                question = (
                    f"Τι γνωρίζουμε για το πρόσωπο {title}; "
                    f"Δώσε όλες τις διαθέσιμες "
                    f"βιογραφικές πληροφορίες."
                )

            complete_answer = build_complete_answer(
                full_text,
                infobox,
                #facts,
            )

            qa_record = {
                "instruction": question,
                "response": complete_answer,
                "title": title,
                "wikidata_id": metadata.get(
                    "wikidata_id"
                ),
                "qa_type": "complete_biography",

                # Keep the structured information as well.
                # "facts": facts,
                "infobox": infobox,
                "sections": sections,

                "revision_id": revision_id,
                "revision_timestamp": revision_timestamp,
                "source_dump": DUMP_PATH.name,
            }

            qa_file.write(
                json.dumps(
                    qa_record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            biography_count += 1
            matched_titles.add(title)

            if (
                MAX_BIOGRAPHIES is not None
                and biography_count >= MAX_BIOGRAPHIES
            ):
                root.clear()
                break

            # Prevent excessive memory use.
            root.clear()
    

    print()
    print(
        f"Extracted biographies: "
        f"{biography_count:,}"
    )

    print(
        f"Metadata titles not matched: "
        f"{len(people_by_title) - len(matched_titles):,}"
    )
    
    not_found = list(set(people_by_title.keys())-matched_titles)
    with open(NOT_FOUND_OUTPUT, "w",encoding="utf-8") as not_found_file:
        for title in not_found:
            not_found_file.write(json.dumps(title, ensure_ascii=False) + "\n")

    print(
        f"Biographies saved to: "
        f"{BIOGRAPHIES_OUTPUT}"
    )

    print(
        f"Complete QA pairs saved to: "
        f"{QA_OUTPUT}"
    )


if __name__ == "__main__":
    main()
