# build_greek_people_metadata_full.py

import json
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
# 16,084

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

# A new filename protects your existing file.
# Change this to "greek_people_metadata.json" to replace it.
OUTPUT_PATH = Path("greek_people_metadata_full.json")

PAGE_SIZE = 1000

# Number of people included in each facts query.
# Reduce this to 50 if Wikidata frequently times out.
FACT_BATCH_SIZE = 50

REQUEST_DELAY = 0.5

# Replace this with a real contact email.
USER_AGENT = (
    "GreekBiographyDatasetBot/1.0 "
    "(Wikipedia user: Christiechris)"
)


# ----------------------------------------------------------------------
# Stage 1: identify Greek humans with a Greek Wikipedia article
# ----------------------------------------------------------------------

PEOPLE_QUERY = """
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX schema: <http://schema.org/>

SELECT DISTINCT
    ?person
    ?article
    ?gender
    ?birth
    ?death
    ?criterion
WHERE {{
    ?person wdt:P31 wd:Q5 .

    {{
        ?person wdt:P27 wd:Q41 .
        BIND("citizenship:Greece" AS ?criterion)
    }}
    UNION
    {{
        ?person wdt:P172/wdt:P279* wd:Q539051 .
        BIND("ethnic_group:Greek" AS ?criterion)
    }}

    ?article schema:about ?person ;
             schema:isPartOf <https://el.wikipedia.org/> .

    OPTIONAL {{
        ?person wdt:P21 ?gender .
    }}

    OPTIONAL {{
        ?person wdt:P569 ?birth .
    }}

    OPTIONAL {{
        ?person wdt:P570 ?death .
    }}
}}
ORDER BY ?person
LIMIT {limit}
OFFSET {offset}
"""


# ----------------------------------------------------------------------
# Stage 2: properties to retrieve for each person
# ----------------------------------------------------------------------

PROPERTY_MAP = {
    # Identity
    "P21": "gender_details",
    "P27": "countries_of_citizenship",
    "P172": "ethnic_groups",

    # Family and relationships
    "P26": "spouses",
    "P451": "unmarried_partners",
    "P40": "children",
    "P1971": "number_of_children",
    "P22": "father",
    "P25": "mother",
    "P3373": "siblings",

    # Birth and death
    "P569": "dates_of_birth",
    "P19": "places_of_birth",
    "P570": "dates_of_death",
    "P20": "places_of_death",
    "P509": "causes_of_death",
    "P1196": "manners_of_death",
    "P119": "places_of_burial",

    # Education and career
    "P106": "occupations",
    "P69": "educated_at",
    "P108": "employers",
    "P39": "positions_held",
    "P101": "fields_of_work",
    "P937": "work_locations",

    # Political and organizational activity
    "P102": "political_parties",
    "P463": "memberships",

    # Cultural and professional information
    "P166": "awards",
    "P800": "notable_works",
    "P135": "movements",
    "P136": "genres",
    "P1303": "instruments",

    # Military information
    "P241": "military_branches",
    "P410": "military_ranks",
    "P607": "conflicts",

    # Sports information
    "P54": "sports_teams",
    "P413": "playing_positions",

    # Personal background
    "P551": "residences",
    "P140": "religions_or_worldviews",
    "P103": "native_languages",
    "P1412": "languages_spoken",

    # Participation
    "P1344": "participants_in",
}


def qid(uri: str) -> str:
    """Extract Q123 or P123 from a Wikidata URI."""
    return uri.rsplit("/", 1)[-1]


def title_from_article(uri: str) -> str:
    """Convert a Greek Wikipedia URL to its article title."""
    path = urlparse(uri).path

    return (
        unquote(path.rsplit("/", 1)[-1])
        .replace("_", " ")
        .strip()
    )


def append_unique(
    record: dict,
    key: str,
    value,
) -> None:
    """Append a value only when it is not empty or duplicated."""
    if value is not None and value not in record[key]:
        record[key].append(value)


def make_session() -> requests.Session:
    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "application/sparql-results+json",
        }
    )

    return session


def run_sparql(
    session: requests.Session,
    query: str,
    maximum_attempts: int = 6,
) -> list[dict]:
    """
    Run a SPARQL query with retries for temporary Wikidata errors.
    """
    last_error = None

    for attempt in range(maximum_attempts):
        try:
            response = session.get(
                SPARQL_ENDPOINT,
                params={
                    "query": query,
                    "format": "json",
                },
                timeout=240,
            )

            response.raise_for_status()

            data = response.json()

            return data["results"]["bindings"]

        except (
            requests.RequestException,
            ValueError,
            KeyError,
        ) as error:
            last_error = error

            if attempt == maximum_attempts - 1:
                break

            wait_seconds = min(60, 2 ** (attempt + 1))

            print(
                f"Temporary query error: {error}. "
                f"Retrying in {wait_seconds} seconds."
            )

            time.sleep(wait_seconds)

    raise RuntimeError(
        "Wikidata query failed after several attempts."
    ) from last_error


def build_people_whitelist(
    session: requests.Session,
) -> dict[str, dict]:
    """
    Retrieve Greek people who have articles in Greek Wikipedia.
    The returned dictionary is indexed by Wikipedia title.
    """
    records: dict[str, dict] = {}
    offset = 0

    print("Stage 1: finding Greek people...")

    while True:
        query = PEOPLE_QUERY.format(
            limit=PAGE_SIZE,
            offset=offset,
        )

        rows = run_sparql(session, query)

        if not rows:
            break

        for row in rows:
            title = title_from_article(
                row["article"]["value"]
            )

            person_qid = qid(
                row["person"]["value"]
            )

            record = records.setdefault(
                title,
                {
                    "title": title,
                    "wikidata_id": person_qid,
                    "criteria": [],
                    "gender": [],
                    "birth": [],
                    "death": [],
                    "facts": {
                        field_name: []
                        for field_name
                        in PROPERTY_MAP.values()
                    },
                },
            )

            append_unique(
                record,
                "criteria",
                row.get(
                    "criterion",
                    {},
                ).get("value"),
            )

            if "gender" in row:
                append_unique(
                    record,
                    "gender",
                    qid(row["gender"]["value"]),
                )

            if "birth" in row:
                append_unique(
                    record,
                    "birth",
                    row["birth"]["value"],
                )

            if "death" in row:
                append_unique(
                    record,
                    "death",
                    row["death"]["value"],
                )

        print(
            f"Fetched {offset + len(rows):,} "
            f"whitelist rows"
        )

        if len(rows) < PAGE_SIZE:
            break

        offset += PAGE_SIZE
        time.sleep(1)

    print(
        f"Found {len(records):,} unique people."
    )

    return records


def chunks(
    values: list[str],
    size: int,
):
    """Yield fixed-size portions of a list."""
    for start in range(0, len(values), size):
        yield values[start:start + size]


def create_facts_query(
    person_qids: list[str],
) -> str:
    """
    Build a query returning one row per person/property/value.

    This avoids the Cartesian multiplication caused by placing
    every spouse, child, award, occupation, etc. in separate
    OPTIONAL clauses in the original query.
    """
    person_values = " ".join(
        f"wd:{person_qid}"
        for person_qid in person_qids
    )

    property_values = " ".join(
        f"wdt:{property_id}"
        for property_id in PROPERTY_MAP
    )

    return f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX bd: <http://www.bigdata.com/rdf#>

SELECT DISTINCT
    ?person
    ?property
    ?value
    ?valueLabel
WHERE {{
    VALUES ?person {{
        {person_values}
    }}

    VALUES ?property {{
        {property_values}
    }}

    ?person ?property ?value .

    SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "el,en" .
    }}
}}
"""


def parse_fact_value(
    row: dict,
) -> dict:
    """
    Convert a SPARQL value into a readable JSON object.
    """
    binding = row["value"]

    value_type = binding.get("type")
    raw_value = binding.get("value")

    label = row.get(
        "valueLabel",
        {},
    ).get("value")

    if value_type == "uri":
        if raw_value.startswith(
            "http://www.wikidata.org/entity/"
        ):
            entity_qid = qid(raw_value)

            result = {
                "qid": entity_qid,
                "label": label or entity_qid,
            }

            return result

        return {
            "uri": raw_value,
            "label": label or raw_value,
        }

    result = {
        "value": raw_value,
    }

    datatype = binding.get("datatype")

    if datatype:
        result["datatype"] = datatype.rsplit(
            "#",
            1,
        )[-1].rsplit("/", 1)[-1]

    language = binding.get("xml:lang")

    if language:
        result["language"] = language

    return result


def enrich_people(
    session: requests.Session,
    records_by_title: dict[str, dict],
) -> None:
    """
    Retrieve structured life facts for every person.
    """
    records_by_qid = {
        record["wikidata_id"]: record
        for record in records_by_title.values()
    }

    person_qids = sorted(records_by_qid)

    total_batches = (
        len(person_qids)
        + FACT_BATCH_SIZE
        - 1
    ) // FACT_BATCH_SIZE

    print()
    print(
        "Stage 2: fetching family, life, "
        "death, education and career facts..."
    )

    for batch_number, qid_batch in enumerate(
        chunks(
            person_qids,
            FACT_BATCH_SIZE,
        ),
        start=1,
    ):
        query = create_facts_query(qid_batch)
        rows = run_sparql(session, query)

        for row in rows:
            person_qid = qid(
                row["person"]["value"]
            )

            property_id = qid(
                row["property"]["value"]
            )

            field_name = PROPERTY_MAP.get(
                property_id
            )

            if field_name is None:
                continue

            person_record = records_by_qid.get(
                person_qid
            )

            if person_record is None:
                continue

            fact = parse_fact_value(row)

            append_unique(
                person_record["facts"],
                field_name,
                fact,
            )

        processed = min(
            batch_number * FACT_BATCH_SIZE,
            len(person_qids),
        )

        print(
            f"Facts batch "
            f"{batch_number:,}/{total_batches:,}: "
            f"{processed:,}/{len(person_qids):,} people"
        )

        time.sleep(REQUEST_DELAY)


def add_summary_counts(
    records_by_title: dict[str, dict],
) -> None:
    """
    Add useful counts without claiming that Wikidata is complete.
    """
    for record in records_by_title.values():
        facts = record["facts"]

        record["summary_counts"] = {
            "listed_spouses": len(
                facts["spouses"]
            ),
            "listed_unmarried_partners": len(
                facts["unmarried_partners"]
            ),
            "listed_children": len(
                facts["children"]
            ),
            "reported_number_of_children": (
                facts["number_of_children"]
            ),
            "listed_siblings": len(
                facts["siblings"]
            ),
            "listed_awards": len(
                facts["awards"]
            ),
            "listed_notable_works": len(
                facts["notable_works"]
            ),
        }


def save_output(
    records_by_title: dict[str, dict],
) -> None:
    output = sorted(
        records_by_title.values(),
        key=lambda item: item["title"],
    )

    temporary_path = OUTPUT_PATH.with_suffix(
        OUTPUT_PATH.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            output,
            output_file,
            ensure_ascii=False,
            indent=2,
        )

    # Replace the final file only after the JSON was
    # written successfully.
    temporary_path.replace(OUTPUT_PATH)

    print()
    print(
        f"Saved {len(output):,} enriched people "
        f"to {OUTPUT_PATH}"
    )


def main() -> None:
    session = make_session()

    people = build_people_whitelist(
        session
    )

    enrich_people(
        session,
        people,
    )

    add_summary_counts(
        people
    )

    save_output(
        people
    )


if __name__ == "__main__":
    main()